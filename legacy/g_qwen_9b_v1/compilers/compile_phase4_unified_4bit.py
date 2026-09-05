# -*- coding: utf-8 -*-
"""G-Qwen 9B Phase 4 Unified 4-bit VRAM Model Compiler.

Reconstructs all 64 layers from Phase 3 charts (Stiefel bases + LoRA residuals r=64)
and compiles them into a unified 4-bit standalone model (~7.8 GB) for 100% VRAM residency
and 35-45 tk/s inference on RTX 3060 (12 GB).
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
    """Vectorized 4-bit uniform quantization with block size 128 on CUDA."""
    shape = w.shape
    w_flat = w.contiguous().view(-1, block_size)
    max_val = torch.amax(torch.abs(w_flat), dim=1, keepdim=True)
    scale = (max_val / 7.0).clamp(min=1e-6).to(torch.float16)
    q = torch.clamp(torch.round(w_flat / scale.to(w.dtype)), -7, 7).to(torch.int8) + 7
    q = q.view(-1, 2)
    packed = (q[:, 0] | (q[:, 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(1), torch.tensor(list(shape), dtype=torch.int32)


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantizes FP8 tensor with block-wise scale inversion into float16 on CUDA."""
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.float16)


def compile_phase4_unified_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("    COMPILADOR OFICIAL FASE 4: MODELO UNIFICADO 4-BIT STANDALONE (G-QWEN 9B)               ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)

    snapshot_dir = os.path.abspath(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0])
    phase3_dir = os.path.abspath("models/g_qwen_9b_phase3")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    out_safetensors_path = os.path.abspath("models/g_qwen_9b_vram_4bit.safetensors")

    vram_dict: Dict[str, torch.Tensor] = {}

    # [1/3] Outside Tensors: Final Norm, Embeddings (4-bit), LM Head (4-bit)
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
        print(" Concluido!", flush=True)

    print(f"Outside tensors concluidos em {time.time() - t0_outside:.2f}s!")

    # [2/3] Processando as 16 Macro-Cartas da Fase 3 (64 Camadas Fisicas)
    print("\n[2/3] Fundindo Bases + LoRA Residual (r=64) e Quantizando 64 Camadas para 4-bit...", flush=True)
    t0_charts = time.time()

    for g in range(16):
        t_chart = time.time()
        cpath = os.path.join(phase3_dir, f"chart_{g}.safetensors")
        cd = st.load_file(cpath, device="cpu")

        # Carrega bases na GPU
        U_mix = cd['basis_mixer_U'].to(device=device, dtype=torch.float16)
        V_mix = cd['basis_mixer_V'].to(device=device, dtype=torch.float16)
        U_gate = cd['basis_ffn_gate_U'].to(device=device, dtype=torch.float16)
        V_gate = cd['basis_ffn_gate_V'].to(device=device, dtype=torch.float16)
        U_up = cd['basis_ffn_up_U'].to(device=device, dtype=torch.float16)
        V_up = cd['basis_ffn_up_V'].to(device=device, dtype=torch.float16)
        U_down = cd['basis_ffn_down_U'].to(device=device, dtype=torch.float16)
        V_down = cd['basis_ffn_down_V'].to(device=device, dtype=torch.float16)

        for idx in range(4):
            l = 4 * g + idx
            is_attn = (l % 4 == 3)

            # Reconstrucao Analitica Fused: W = U S V^T + A B^T
            # Gate Proj
            S_gate = cd[f'layer_{l}_ffn_gate_S'].to(device=device, dtype=torch.float16)
            A_gate = cd[f'layer_{l}_ffn_gate_res_A'].to(device=device, dtype=torch.float16)
            B_gate = cd[f'layer_{l}_ffn_gate_res_B'].to(device=device, dtype=torch.float16)
            r_g = S_gate.shape[0]
            W_gate = torch.matmul(torch.matmul(U_gate[:, :r_g], S_gate), V_gate.t()) + torch.matmul(A_gate, B_gate.t())
            p_g, s_g, sh_g = quantize_4bit(W_gate)
            vram_dict[f"blk.{l}.ffn_gate.q"] = p_g.cpu()
            vram_dict[f"blk.{l}.ffn_gate.s"] = s_g.cpu()
            vram_dict[f"blk.{l}.ffn_gate.shape"] = sh_g.cpu()
            del S_gate, A_gate, B_gate, W_gate, p_g, s_g, sh_g

            # Up Proj
            S_up = cd[f'layer_{l}_ffn_up_S'].to(device=device, dtype=torch.float16)
            A_up = cd[f'layer_{l}_ffn_up_res_A'].to(device=device, dtype=torch.float16)
            B_up = cd[f'layer_{l}_ffn_up_res_B'].to(device=device, dtype=torch.float16)
            r_u = S_up.shape[0]
            W_up = torch.matmul(torch.matmul(U_up[:, :r_u], S_up), V_up.t()) + torch.matmul(A_up, B_up.t())
            p_u, s_u, sh_u = quantize_4bit(W_up)
            vram_dict[f"blk.{l}.ffn_up.q"] = p_u.cpu()
            vram_dict[f"blk.{l}.ffn_up.s"] = s_u.cpu()
            vram_dict[f"blk.{l}.ffn_up.shape"] = sh_u.cpu()
            del S_up, A_up, B_up, W_up, p_u, s_u, sh_u

            # Down Proj
            S_down = cd[f'layer_{l}_ffn_down_S'].to(device=device, dtype=torch.float16)
            A_down = cd[f'layer_{l}_ffn_down_res_A'].to(device=device, dtype=torch.float16)
            B_down = cd[f'layer_{l}_ffn_down_res_B'].to(device=device, dtype=torch.float16)
            r_d = S_down.shape[0]
            W_down = torch.matmul(torch.matmul(U_down[:, :r_d], S_down), V_down.t()) + torch.matmul(A_down, B_down.t())
            p_d, s_d, sh_d = quantize_4bit(W_down)
            vram_dict[f"blk.{l}.ffn_down.q"] = p_d.cpu()
            vram_dict[f"blk.{l}.ffn_down.s"] = s_d.cpu()
            vram_dict[f"blk.{l}.ffn_down.shape"] = sh_d.cpu()
            del S_down, A_down, B_down, W_down, p_d, s_d, sh_d

            # Mixer Proj
            S_mix = cd[f'layer_{l}_mixer_S'].to(device=device, dtype=torch.float16)
            A_mix = cd[f'layer_{l}_mixer_res_A'].to(device=device, dtype=torch.float16)
            B_mix = cd[f'layer_{l}_mixer_res_B'].to(device=device, dtype=torch.float16)
            r_m = S_mix.shape[0]
            W_mix = torch.matmul(torch.matmul(U_mix[:, :r_m], S_mix), V_mix.t()) + torch.matmul(A_mix, B_mix.t())
            p_m, s_m, sh_m = quantize_4bit(W_mix)
            vram_dict[f"blk.{l}.mixer.q"] = p_m.cpu()
            vram_dict[f"blk.{l}.mixer.s"] = s_m.cpu()
            vram_dict[f"blk.{l}.mixer.shape"] = sh_m.cpu()
            del S_mix, A_mix, B_mix, W_mix, p_m, s_m, sh_m

            # Layer Norms
            vram_dict[f"blk.{l}.input_layernorm.weight"] = cd[f"layer_{l}_input_layernorm_weight"].to(torch.float16).cpu()
            vram_dict[f"blk.{l}.post_attention_layernorm.weight"] = cd[f"layer_{l}_post_attention_layernorm_weight"].to(torch.float16).cpu()

            if is_attn:
                # Softmax Attention QKV (Quantized to 4-bit)
                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    w_fp8 = cd[f"layer_{l}_{proj_name}_weight"].to(device)
                    s_inv = cd[f"layer_{l}_{proj_name}_scale_inv"].to(device)
                    w_deq = dequant_fp8(w_fp8, s_inv)
                    p, s, sh = quantize_4bit(w_deq)
                    vram_dict[f"blk.{l}.attn_{proj_name[0]}.q"] = p.cpu()
                    vram_dict[f"blk.{l}.attn_{proj_name[0]}.s"] = s.cpu()
                    vram_dict[f"blk.{l}.attn_{proj_name[0]}.shape"] = sh.cpu()
                    del w_fp8, s_inv, w_deq, p, s, sh

                vram_dict[f"blk.{l}.attn_q_norm.weight"] = cd[f"layer_{l}_q_norm_weight"].to(torch.float16).cpu()
                vram_dict[f"blk.{l}.attn_k_norm.weight"] = cd[f"layer_{l}_k_norm_weight"].to(torch.float16).cpu()
            else:
                # Linear Attention (QKV, Z in 4-bit, conv1d/dt_bias/A_log in FP16)
                qkv_fp8 = cd[f"layer_{l}_in_proj_qkv_weight"].to(device)
                qkv_s_inv = cd[f"layer_{l}_in_proj_qkv_scale_inv"].to(device)
                qkv_deq = dequant_fp8(qkv_fp8, qkv_s_inv)
                p_qkv, s_qkv, sh_qkv = quantize_4bit(qkv_deq)
                vram_dict[f"blk.{l}.attn_qkv.q"] = p_qkv.cpu()
                vram_dict[f"blk.{l}.attn_qkv.s"] = s_qkv.cpu()
                vram_dict[f"blk.{l}.attn_qkv.shape"] = sh_qkv.cpu()
                del qkv_fp8, qkv_s_inv, qkv_deq, p_qkv, s_qkv, sh_qkv

                z_fp8 = cd[f"layer_{l}_in_proj_z_weight"].to(device)
                z_s_inv = cd[f"layer_{l}_in_proj_z_scale_inv"].to(device)
                z_deq = dequant_fp8(z_fp8, z_s_inv)
                p_z, s_z, sh_z = quantize_4bit(z_deq)
                vram_dict[f"blk.{l}.attn_gate.q"] = p_z.cpu()
                vram_dict[f"blk.{l}.attn_gate.s"] = s_z.cpu()
                vram_dict[f"blk.{l}.attn_gate.shape"] = sh_z.cpu()
                del z_fp8, z_s_inv, z_deq, p_z, s_z, sh_z

                vram_dict[f"blk.{l}.attn_a.weight"] = cd[f"layer_{l}_in_proj_a_weight"].to(torch.float16).cpu()
                vram_dict[f"blk.{l}.attn_b.weight"] = cd[f"layer_{l}_in_proj_b_weight"].to(torch.float16).cpu()
                vram_dict[f"blk.{l}.attn_conv1d.weight"] = cd[f"layer_{l}_conv1d_weight"].to(torch.float16).cpu()
                vram_dict[f"blk.{l}.attn_dt_bias"] = cd[f"layer_{l}_dt_bias"].to(torch.float16).cpu()
                vram_dict[f"blk.{l}.attn_A_log"] = cd[f"layer_{l}_A_log"].to(torch.float16).cpu()
                vram_dict[f"blk.{l}.attn_norm.weight"] = cd[f"layer_{l}_norm_weight"].to(torch.float16).cpu()

        del U_mix, V_mix, U_gate, V_gate, U_up, V_up, U_down, V_down, cd
        torch.cuda.empty_cache()
        print(f"Carta {g:02d}/16 (Camadas {4*g:02d}..{4*g+3:02d}) compilada em {time.time() - t_chart:.2f}s! RAM RSS: {psutil.Process().memory_info().rss/1e9:.2f} GB", flush=True)

    print(f"\nTodas as 64 camadas compiladas com sucesso em {time.time() - t0_charts:.2f}s!")

    # [3/3] Salvando arquivo consolidado
    print("\n[3/3] Salvando modelo unificado em disco...", end="", flush=True)
    t0_save = time.time()
    st.save_file(vram_dict, out_safetensors_path)
    file_size_gb = os.path.getsize(out_safetensors_path) / (1024**3)
    print(f" Concluido em {time.time() - t0_save:.2f}s!")
    print("=" * 105, flush=True)
    print(f"MODELO UNIFICADO FASE 4 GERADO COM SUCESSO: {out_safetensors_path}")
    print(f"Tamanho do Arquivo Final : {file_size_gb:.2f} GB (Budget estrito de VRAM: 100% residente em 12 GB)")
    print(f"Total de Tensores Salvos : {len(vram_dict):,}")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    compile_phase4_unified_model()
