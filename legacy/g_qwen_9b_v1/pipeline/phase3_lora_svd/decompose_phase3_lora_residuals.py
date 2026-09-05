# -*- coding: utf-8 -*-
"""Decompose Phase 2 Residuals into LoRA-Residual Factors (r_Delta = 64).

Reduces residual tensor footprint from 9.6 GB (4-bit RVQ) to ~645 MB (FP16).
Replaces _res_q and _res_scale with _res_A and _res_B in models/g_qwen_9b_phase3.
"""

import os
import sys
import glob
import time
import torch
import safetensors.torch as st

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight tensor to float32 on CUDA."""
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32))


def compute_lora_factors(R: torch.Tensor, rank: int = 64) -> tuple:
    """Computes balanced truncated SVD factors: A in R^{d_out x r}, B in R^{d_in x r}."""
    U_r, S_r, V_r = torch.pca_lowrank(R, q=rank, center=False, niter=4)
    sqrt_S = torch.sqrt(torch.clamp(S_r, min=0.0))
    A = (U_r * sqrt_S.unsqueeze(0)).to(torch.float16).contiguous().cpu()
    B = (V_r * sqrt_S.unsqueeze(0)).to(torch.float16).contiguous().cpu()
    return A, B


def run_phase3_decomposition(target_rank: int = 64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print(f"    FASE 3: FATORACAO LORA-RESIDUAL SVD (RANK r_Delta = {target_rank})                            ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    phase2_dir = os.path.abspath("models/g_qwen_9b_phase2")
    phase3_dir = os.path.abspath("models/g_qwen_9b_phase3")
    os.makedirs(phase3_dir, exist_ok=True)

    t0_total = time.time()
    total_original_res_mb = 0.0
    total_lora_res_mb = 0.0

    for g in range(16):
        t0_chart = time.time()
        cpath_p2 = os.path.join(phase2_dir, f"chart_{g}.safetensors")
        cpath_p3 = os.path.join(phase3_dir, f"chart_{g}.safetensors")

        print(f"\n--- [Carta {g:02d}/16] Carregando bases e calibracoes da Fase 2...", end="", flush=True)
        cd2 = st.load_file(cpath_p2, device="cpu")

        # Dicionario para salvar na Fase 3
        cd3 = {}

        # 1. Copiar Bases Compartilhadas intactas
        for k in ["basis_mixer_U", "basis_mixer_V", "basis_ffn_gate_U", "basis_ffn_gate_V", 
                  "basis_ffn_up_U", "basis_ffn_up_V", "basis_ffn_down_U", "basis_ffn_down_V"]:
            cd3[k] = cd2[k].contiguous()

        U_mix = cd2["basis_mixer_U"].to(device=device, dtype=torch.float32)
        V_mix = cd2["basis_mixer_V"].to(device=device, dtype=torch.float32)
        U_gate = cd2["basis_ffn_gate_U"].to(device=device, dtype=torch.float32)
        V_gate = cd2["basis_ffn_gate_V"].to(device=device, dtype=torch.float32)
        U_up = cd2["basis_ffn_up_U"].to(device=device, dtype=torch.float32)
        V_up = cd2["basis_ffn_up_V"].to(device=device, dtype=torch.float32)
        U_down = cd2["basis_ffn_down_U"].to(device=device, dtype=torch.float32)
        V_down = cd2["basis_ffn_down_V"].to(device=device, dtype=torch.float32)

        print(" Concluido! Fatorando 4 camadas...", flush=True)

        for idx in range(4):
            l = 4 * g + idx
            is_attn = (l % 4 == 3)
            layer_f = os.path.join(snapshot_dir, f"layers-{l}.safetensors")

            # Copiar matrizes de coordenadas S_l
            for proj in ["mixer", "ffn_gate", "ffn_up", "ffn_down"]:
                cd3[f"layer_{l}_{proj}_S"] = cd2[f"layer_{l}_{proj}_S"].contiguous()

            with st.safe_open(layer_f, framework="pt") as lf:
                # 1. Mixer (Attention o_proj ou DeltaNet out_proj)
                proj_key = f"model.language_model.layers.{l}.self_attn.o_proj.weight" if is_attn else f"model.language_model.layers.{l}.linear_attn.out_proj.weight"
                scale_key = proj_key + "_scale_inv"
                W_orig = dequant_fp8(lf.get_tensor(proj_key).to(device), lf.get_tensor(scale_key).to(device))
                S_mix = cd2[f"layer_{l}_mixer_S"].to(device=device, dtype=torch.float32)
                R_mix = W_orig - (U_mix @ S_mix @ V_mix.t())
                A_mix, B_mix = compute_lora_factors(R_mix, rank=target_rank)
                cd3[f"layer_{l}_mixer_res_A"] = A_mix
                cd3[f"layer_{l}_mixer_res_B"] = B_mix

                # 2. FFN Gate
                W_orig = dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight").to(device), 
                                     lf.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv").to(device))
                S_gate = cd2[f"layer_{l}_ffn_gate_S"].to(device=device, dtype=torch.float32)
                R_gate = W_orig - (U_gate @ S_gate @ V_gate.t())
                A_gate, B_gate = compute_lora_factors(R_gate, rank=target_rank)
                cd3[f"layer_{l}_ffn_gate_res_A"] = A_gate
                cd3[f"layer_{l}_ffn_gate_res_B"] = B_gate

                # 3. FFN Up
                W_orig = dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight").to(device), 
                                     lf.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv").to(device))
                S_up = cd2[f"layer_{l}_ffn_up_S"].to(device=device, dtype=torch.float32)
                R_up = W_orig - (U_up @ S_up @ V_up.t())
                A_up, B_up = compute_lora_factors(R_up, rank=target_rank)
                cd3[f"layer_{l}_ffn_up_res_A"] = A_up
                cd3[f"layer_{l}_ffn_up_res_B"] = B_up

                # 4. FFN Down
                W_orig = dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight").to(device), 
                                     lf.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv").to(device))
                S_down = cd2[f"layer_{l}_ffn_down_S"].to(device=device, dtype=torch.float32)
                R_down = W_orig - (U_down @ S_down @ V_down.t())
                A_down, B_down = compute_lora_factors(R_down, rank=target_rank)
                cd3[f"layer_{l}_ffn_down_res_A"] = A_down
                cd3[f"layer_{l}_ffn_down_res_B"] = B_down

        # Salvar Carta compacta da Fase 3
        st.save_file(cd3, cpath_p3)
        sz_p2 = os.path.getsize(cpath_p2) / 1e6
        sz_p3 = os.path.getsize(cpath_p3) / 1e6
        total_original_res_mb += sz_p2
        total_lora_res_mb += sz_p3
        print(f"  -> Carta {g:02d} concluida em {time.time() - t0_chart:.2f}s! Tamanho: {sz_p2:.1f} MB -> {sz_p3:.1f} MB (-{(1.0 - sz_p3/sz_p2)*100:.1f}%)", flush=True)

    t_total = time.time() - t0_total
    print("\n" + "=" * 105)
    print("    FASE 3: DECOMPOSICAO LORA-RESIDUAL CONCLUIDA COM SUCESSO!                          ", flush=True)
    print("=" * 105)
    print(f"Tempo Total de Processamento : {t_total:.2f}s ({t_total/60:.2f} minutos)")
    print(f"Volume Original das Cartas   : {total_original_res_mb / 1e3:.2f} GB (Fase 2)")
    print(f"Volume Compactado LoRA-SVD   : {total_lora_res_mb / 1e3:.2f} GB (Fase 3)")
    print(f"Economia Liquida de Disco/RAM: {(1.0 - total_lora_res_mb / total_original_res_mb)*100:.1f}% de reducao!")
    print(f"Pico de VRAM Utilizado       : {torch.cuda.max_memory_allocated(0)/1e6:.1f} MB")
    print(f"Diretorio de Saida           : {phase3_dir}")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_phase3_decomposition(target_rank=64)
