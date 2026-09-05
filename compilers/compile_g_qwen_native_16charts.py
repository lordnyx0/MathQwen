"""Compilador do Modelo G-Qwen 9B Nativo Completo (16 Macro-Cartas / 64 Camadas Fisicas).

Objetivo:
- Processar todas as 16 cartas (camadas 0 a 63)
- Bases Stiefel (U_g, V_g) em FP8 (float8_e4m3fn)
- Coordenadas S em FP8 com escala por tensor
- Projecoes de Mixer e Residuos de Mixer em 4-bit
- Residuos densos FFN (gate, up, down) em 2-bit com bloco 128
- Projecoes exclusivas (DeltaNet e Attention) em 4-bit
- Embeddings e LM Head em 4-bit
- Saida: models/g_qwen_9b_native_16charts.safetensors (~10.5 GB)
- Permite execucao 100% residente na RTX 3060 (12 GB) com ~1.7 GB de margem VRAM.
"""

import os
import sys
import glob
import time
import psutil
import torch
import safetensors.torch as st
from typing import Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def quantize_4bit(w: torch.Tensor, block_size: int = 128):
    """Quantizacao uniforme simetrica de 4 bits empacotada em uint8."""
    shape = w.shape
    w_flat = w.contiguous().view(-1, block_size)
    max_val = torch.amax(torch.abs(w_flat), dim=1, keepdim=True)
    scale = (max_val / 7.0).clamp(min=1e-6).to(torch.float16)
    q = torch.clamp(torch.round(w_flat / scale.to(w.dtype)), -7, 7).to(torch.int8) + 7
    q = q.view(-1, 2)
    packed = (q[:, 0] | (q[:, 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(1), torch.tensor(list(shape), dtype=torch.int32)


def quantize_2bit_from_4bit(packed_4bit: torch.Tensor, scale_4bit: torch.Tensor, shape: torch.Size):
    """Requantiza residuo 4-bit em 2-bit uniforme (bloco 128) empacotado em uint8 (4 valores/byte)."""
    low = (packed_4bit & 0x0F).to(torch.int8) - 7
    high = ((packed_4bit >> 4) & 0x0F).to(torch.int8) - 7
    unp = torch.stack([low, high], dim=1).view(-1, 128).to(torch.float32) * scale_4bit.view(-1, 1).to(torch.float32)
    
    max_val = torch.amax(torch.abs(unp), dim=1, keepdim=True)
    s2 = (max_val / 1.5).clamp(min=1e-6).to(torch.float16)
    q2 = torch.clamp(torch.round(unp / s2.to(torch.float32)), -2, 1).to(torch.int8) + 2
    q2 = q2.view(-1, 4)
    packed2 = (q2[:, 0] | (q2[:, 1] << 2) | (q2[:, 2] << 4) | (q2[:, 3] << 6)).to(torch.uint8)
    return packed2, s2.squeeze(1)


def quantize_fp8_tensor(t: torch.Tensor):
    """Converte tensor para float8_e4m3fn com fator de escala."""
    t_f = t.to(torch.float32)
    max_abs = torch.amax(torch.abs(t_f)).clamp(min=1e-6)
    scale = (max_abs / 448.0).to(torch.float16)
    t_fp8 = (t_f / scale.to(torch.float32)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return t_fp8, scale


def dequant_fp8_weight(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.float16)


def compile_native_16charts():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("    COMPILADOR G-QWEN 9B NATIVO COMPLETO (16 CARTAS / 64 CAMADAS -> 100% VRAM RESIDENTE)   ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    charts_dir = os.path.abspath("models/g_qwen_9b_phase1")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    out_safetensors_path = os.path.abspath("models/g_qwen_9b_native_16charts.safetensors")

    vram_dict: Dict[str, torch.Tensor] = {}

    # 1. Outside Tensors: Output Norm, Token Embeddings (4-bit), LM Head (4-bit)
    print("\n[1/3] Processando Embeddings, LM Head e Final Norm (Quantizacao 4-bit)...", flush=True)
    t0_outside = time.time()
    with st.safe_open(outside_path, framework="pt") as f:
        vram_dict["output_norm.weight"] = f.get_tensor("model.language_model.norm.weight").to(torch.float16).cpu()

        print("  - Quantizando token_embd para 4-bit...", end="", flush=True)
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight").to(device=device, dtype=torch.float16)
        p_emb, s_emb, sh_emb = quantize_4bit(embed_w)
        vram_dict["token_embd.q"] = p_emb.cpu()
        vram_dict["token_embd.s"] = s_emb.cpu()
        vram_dict["token_embd.shape"] = sh_emb.cpu()
        del embed_w, p_emb, s_emb, sh_emb
        torch.cuda.empty_cache()
        print(" Concluido!", flush=True)

        print("  - Quantizando lm_head para 4-bit...", end="", flush=True)
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.float16)
        p_head, s_head, sh_head = quantize_4bit(lm_head_w)
        vram_dict["lm_head.q"] = p_head.cpu()
        vram_dict["lm_head.s"] = s_head.cpu()
        vram_dict["lm_head.shape"] = sh_head.cpu()
        del lm_head_w, p_head, s_head, sh_head
        torch.cuda.empty_cache()
        print(f" Concluido em {time.time()-t0_outside:.2f}s!", flush=True)

    # 2. Processar todas as 16 Macro-Cartas (64 Camadas Fisicas)
    print("\n[2/3] Processando as 16 Macro-Cartas Nativas e Camadas Exclusivas...", flush=True)
    t0_charts = time.time()

    basis_names = [
        "basis_mixer_U", "basis_mixer_V",
        "basis_ffn_gate_U", "basis_ffn_gate_V",
        "basis_ffn_up_U", "basis_ffn_up_V",
        "basis_ffn_down_U", "basis_ffn_down_V",
    ]

    for g in range(16):
        t_chart = time.time()
        cpath = os.path.join(charts_dir, f"chart_{g}.safetensors")
        cd = st.load_file(cpath)

        for b_name in basis_names:
            vram_dict[f"chart.{g}.{b_name}"] = cd[b_name].to(torch.float8_e4m3fn).cpu()

        for idx in range(4):
            l = 4 * g + idx
            is_attn = (l % 4 == 3)

            s_mix_fp8, s_mix_scale = quantize_fp8_tensor(cd[f"layer_{l}_mixer_S"])
            vram_dict[f"blk.{l}.mixer_S.fp8"] = s_mix_fp8.cpu()
            vram_dict[f"blk.{l}.mixer_S.scale"] = s_mix_scale.cpu()

            s_gate_fp8, s_gate_scale = quantize_fp8_tensor(cd[f"layer_{l}_ffn_gate_S"])
            vram_dict[f"blk.{l}.ffn_gate_S.fp8"] = s_gate_fp8.cpu()
            vram_dict[f"blk.{l}.ffn_gate_S.scale"] = s_gate_scale.cpu()

            s_up_fp8, s_up_scale = quantize_fp8_tensor(cd[f"layer_{l}_ffn_up_S"])
            vram_dict[f"blk.{l}.ffn_up_S.fp8"] = s_up_fp8.cpu()
            vram_dict[f"blk.{l}.ffn_up_S.scale"] = s_up_scale.cpu()

            s_down_fp8, s_down_scale = quantize_fp8_tensor(cd[f"layer_{l}_ffn_down_S"])
            vram_dict[f"blk.{l}.ffn_down_S.fp8"] = s_down_fp8.cpu()
            vram_dict[f"blk.{l}.ffn_down_S.scale"] = s_down_scale.cpu()

            vram_dict[f"blk.{l}.mixer_res_q"] = cd[f"layer_{l}_mixer_res_q"].cpu()
            vram_dict[f"blk.{l}.mixer_res_scale"] = cd[f"layer_{l}_mixer_res_scale"].to(torch.float16).cpu()

            p2_gate, s2_gate = quantize_2bit_from_4bit(
                cd[f"layer_{l}_ffn_gate_res_q"],
                cd[f"layer_{l}_ffn_gate_res_scale"],
                torch.Size([17408, 5120])
            )
            vram_dict[f"blk.{l}.ffn_gate_res_2q"] = p2_gate.cpu()
            vram_dict[f"blk.{l}.ffn_gate_res_2scale"] = s2_gate.cpu()

            p2_up, s2_up = quantize_2bit_from_4bit(
                cd[f"layer_{l}_ffn_up_res_q"],
                cd[f"layer_{l}_ffn_up_res_scale"],
                torch.Size([17408, 5120])
            )
            vram_dict[f"blk.{l}.ffn_up_res_2q"] = p2_up.cpu()
            vram_dict[f"blk.{l}.ffn_up_res_2scale"] = s2_up.cpu()

            p2_down, s2_down = quantize_2bit_from_4bit(
                cd[f"layer_{l}_ffn_down_res_q"],
                cd[f"layer_{l}_ffn_down_res_scale"],
                torch.Size([5120, 17408])
            )
            vram_dict[f"blk.{l}.ffn_down_res_2q"] = p2_down.cpu()
            vram_dict[f"blk.{l}.ffn_down_res_2scale"] = s2_down.cpu()

            layer_f = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(layer_f, framework="pt") as lf:
                vram_dict[f"blk.{l}.attn_norm.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.input_layernorm.weight").to(torch.float16).cpu()
                vram_dict[f"blk.{l}.ffn_norm.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.post_attention_layernorm.weight").to(torch.float16).cpu()

                if is_attn:
                    for proj_name in ["q_proj", "k_proj", "v_proj"]:
                        w_fp8 = lf.get_tensor(f"model.language_model.layers.{l}.self_attn.{proj_name}.weight").to(device)
                        s_inv = lf.get_tensor(f"model.language_model.layers.{l}.self_attn.{proj_name}.weight_scale_inv").to(device)
                        w_deq = dequant_fp8_weight(w_fp8, s_inv)
                        p, s, sh = quantize_4bit(w_deq)
                        vram_dict[f"blk.{l}.attn_{proj_name[0]}.q"] = p.cpu()
                        vram_dict[f"blk.{l}.attn_{proj_name[0]}.s"] = s.cpu()
                        vram_dict[f"blk.{l}.attn_{proj_name[0]}.shape"] = sh.cpu()
                        del w_fp8, s_inv, w_deq, p, s, sh

                    vram_dict[f"blk.{l}.attn_q_norm.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.self_attn.q_norm.weight").to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.attn_k_norm.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.self_attn.k_norm.weight").to(torch.float16).cpu()
                else:
                    qkv_fp8 = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight").to(device)
                    qkv_s_inv = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv").to(device)
                    qkv_deq = dequant_fp8_weight(qkv_fp8, qkv_s_inv)
                    p_qkv, s_qkv, sh_qkv = quantize_4bit(qkv_deq)
                    vram_dict[f"blk.{l}.attn_qkv.q"] = p_qkv.cpu()
                    vram_dict[f"blk.{l}.attn_qkv.s"] = s_qkv.cpu()
                    vram_dict[f"blk.{l}.attn_qkv.shape"] = sh_qkv.cpu()
                    del qkv_fp8, qkv_s_inv, qkv_deq, p_qkv, s_qkv, sh_qkv

                    z_fp8 = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight").to(device)
                    z_s_inv = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv").to(device)
                    z_deq = dequant_fp8_weight(z_fp8, z_s_inv)
                    p_z, s_z, sh_z = quantize_4bit(z_deq)
                    vram_dict[f"blk.{l}.attn_gate.q"] = p_z.cpu()
                    vram_dict[f"blk.{l}.attn_gate.s"] = s_z.cpu()
                    vram_dict[f"blk.{l}.attn_gate.shape"] = sh_z.cpu()
                    del z_fp8, z_s_inv, z_deq, p_z, s_z, sh_z

                    vram_dict[f"blk.{l}.ssm_alpha.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_a.weight").to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_beta.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_b.weight").to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_conv1d.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.conv1d.weight").to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_dt.bias"] = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.dt_bias").to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_a.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.A_log").to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_norm.weight"] = lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.norm.weight").to(torch.float16).cpu()

                torch.cuda.empty_cache()

        del cd
        proc = psutil.Process()
        print(f"  - Carta {g+1:2d}/16 (Camadas {4*g:2d}..{4*g+3:2d}) processada em {time.time()-t_chart:.2f}s | RAM RSS: {proc.memory_info().rss/1e9:.2f} GB", flush=True)

    print(f"\nTodas as 16 macro-cartas processadas em {time.time()-t0_charts:.2f}s!", flush=True)

    # 3. Gravar Safetensors Unificado
    print(f"\n[3/3] Gravando {out_safetensors_path}...", end="", flush=True)
    t0_save = time.time()
    st.save_file(vram_dict, out_safetensors_path)
    file_size_gb = os.path.getsize(out_safetensors_path) / (1024**3)
    print(f" Concluido em {time.time()-t0_save:.2f}s!", flush=True)

    print("\n" + "=" * 105)
    print("COMPILACAO G-QWEN 9B NATIVO (16 CARTAS / 64 CAMADAS) FINALIZADA COM SUCESSO!")
    print(f"Arquivo Unificado : {out_safetensors_path}")
    print(f"Tamanho Final     : {file_size_gb:.2f} GiB ({file_size_gb * 1.074:.2f} GB)")
    print(f"Margem Livre VRAM : {12.0 - file_size_gb:.2f} GiB LIVRES na RTX 3060 (12 GB)!")
    print("Zero Offload Ciclado | 64 Camadas Integrais | 100% Residente em VRAM")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    compile_native_16charts()
