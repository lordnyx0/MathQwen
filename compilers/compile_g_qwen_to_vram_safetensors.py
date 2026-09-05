"""Compile G-Qwen 9B into a unified standalone VRAM Safetensors model.

Features:
- Complete Qwen 3.5 architecture (64 layers: 48 Gated DeltaNet + 16 Softmax Attention)
- 16 Grassmannian Macro-Charts reconstructed via GPU cuBLAS and quantized to uniform 4-bit
- Layer exclusive projections and attention weights packaged in FP8/FP16
- Embeddings and LM Head quantized to 4-bit
- Total file size: ~8.0 - 8.5 GB (100% VRAM resident for 12 GB GPUs)
- Output: models/g_qwen_9b_vram.safetensors
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


def dequant_chart_residual_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size, block_size: int = 128) -> torch.Tensor:
    """Vectorized 4-bit unpacking from chart file directly into float16 on CUDA."""
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, block_size)
    return (unpacked.to(torch.float16) * scale.view(-1, 1).to(torch.float16)).view(shape)


def compile_to_vram_safetensors():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 100, flush=True)
    print("    COMPILADOR OFICIAL G-QWEN 9B -> VRAM SAFETENSORS (100% GPU RESIDENT, PARA 12 GB VRAM)    ", flush=True)
    print("=" * 100, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    charts_dir = os.path.abspath("models/g_qwen_9b_phase1")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    out_safetensors_path = os.path.abspath("models/g_qwen_9b_vram.safetensors")

    vram_dict: Dict[str, torch.Tensor] = {}

    # 1. Outside Tensors: Final Norm, Embeddings (4-bit), LM Head (4-bit)
    print("\n[1/3] Processando Embeddings, LM Head e Final Norm...", flush=True)
    t0_outside = time.time()
    with st.safe_open(outside_path, framework="pt") as f:
        # Final Norm
        vram_dict["output_norm.weight"] = f.get_tensor("model.language_model.norm.weight").to(torch.float16).cpu()

        # Token Embeddings (Quantize to 4-bit)
        print("  - Quantizando token_embd para 4-bit...", end="", flush=True)
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight").to(device=device, dtype=torch.float16)
        p_emb, s_emb, sh_emb = quantize_4bit(embed_w)
        vram_dict["token_embd.q"] = p_emb.cpu()
        vram_dict["token_embd.s"] = s_emb.cpu()
        vram_dict["token_embd.shape"] = sh_emb.cpu()
        del embed_w, p_emb, s_emb, sh_emb
        torch.cuda.empty_cache()
        print(" Concluido!", flush=True)

        # Output LM Head (Quantize to 4-bit)
        print("  - Quantizando lm_head para 4-bit...", end="", flush=True)
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.float16)
        p_head, s_head, sh_head = quantize_4bit(lm_head_w)
        vram_dict["lm_head.q"] = p_head.cpu()
        vram_dict["lm_head.s"] = s_head.cpu()
        vram_dict["lm_head.shape"] = sh_head.cpu()
        del lm_head_w, p_head, s_head, sh_head
        torch.cuda.empty_cache()
        print(f" Concluido em {time.time()-t0_outside:.2f}s!", flush=True)

    # 2. Reconstruct & Quantize all 64 layers across 16 charts
    print("\n[2/3] Reconstruindo e Quantizando as 64 Camadas (16 Macro-Cartas)...", flush=True)
    t0_charts = time.time()

    for g in range(16):
        t_chart = time.time()
        cpath = os.path.join(charts_dir, f"chart_{g}.safetensors")
        cd = st.load_file(cpath, device=str(device))

        U_mix = cd['basis_mixer_U'].to(torch.float16)
        V_mix = cd['basis_mixer_V'].to(torch.float16)
        U_gate = cd['basis_ffn_gate_U'].to(torch.float16)
        V_gate = cd['basis_ffn_gate_V'].to(torch.float16)
        U_up = cd['basis_ffn_up_U'].to(torch.float16)
        V_up = cd['basis_ffn_up_V'].to(torch.float16)
        U_down = cd['basis_ffn_down_U'].to(torch.float16)
        V_down = cd['basis_ffn_down_V'].to(torch.float16)

        for idx in range(4):
            l = 4 * g + idx
            is_attn = (l % 4 == 3)

            # Reconstruct 4 dense matrices via cuBLAS on GPU
            R_mix = dequant_chart_residual_4bit(cd[f'layer_{l}_mixer_res_q'], cd[f'layer_{l}_mixer_res_scale'], torch.Size([5120, 6144]))
            W_mix = torch.matmul(U_mix, torch.matmul(cd[f'layer_{l}_mixer_S'].to(torch.float16), V_mix.t())) + R_mix
            p_mix, s_mix, sh_mix = quantize_4bit(W_mix)
            vram_dict[f"blk.{l}.mix.q"] = p_mix.cpu()
            vram_dict[f"blk.{l}.mix.s"] = s_mix.cpu()
            vram_dict[f"blk.{l}.mix.shape"] = sh_mix.cpu()
            del R_mix, W_mix, p_mix, s_mix, sh_mix

            R_gate = dequant_chart_residual_4bit(cd[f'layer_{l}_ffn_gate_res_q'], cd[f'layer_{l}_ffn_gate_res_scale'], torch.Size([17408, 5120]))
            W_gate = torch.matmul(U_gate, torch.matmul(cd[f'layer_{l}_ffn_gate_S'].to(torch.float16), V_gate.t())) + R_gate
            p_gate, s_gate, sh_gate = quantize_4bit(W_gate)
            vram_dict[f"blk.{l}.ffn_gate.q"] = p_gate.cpu()
            vram_dict[f"blk.{l}.ffn_gate.s"] = s_gate.cpu()
            vram_dict[f"blk.{l}.ffn_gate.shape"] = sh_gate.cpu()
            del R_gate, W_gate, p_gate, s_gate, sh_gate

            R_up = dequant_chart_residual_4bit(cd[f'layer_{l}_ffn_up_res_q'], cd[f'layer_{l}_ffn_up_res_scale'], torch.Size([17408, 5120]))
            W_up = torch.matmul(U_up, torch.matmul(cd[f'layer_{l}_ffn_up_S'].to(torch.float16), V_up.t())) + R_up
            p_up, s_up, sh_up = quantize_4bit(W_up)
            vram_dict[f"blk.{l}.ffn_up.q"] = p_up.cpu()
            vram_dict[f"blk.{l}.ffn_up.s"] = s_up.cpu()
            vram_dict[f"blk.{l}.ffn_up.shape"] = sh_up.cpu()
            del R_up, W_up, p_up, s_up, sh_up

            R_down = dequant_chart_residual_4bit(cd[f'layer_{l}_ffn_down_res_q'], cd[f'layer_{l}_ffn_down_res_scale'], torch.Size([5120, 17408]))
            W_down = torch.matmul(U_down, torch.matmul(cd[f'layer_{l}_ffn_down_S'].to(torch.float16), V_down.t())) + R_down
            p_down, s_down, sh_down = quantize_4bit(W_down)
            vram_dict[f"blk.{l}.ffn_down.q"] = p_down.cpu()
            vram_dict[f"blk.{l}.ffn_down.s"] = s_down.cpu()
            vram_dict[f"blk.{l}.ffn_down.shape"] = sh_down.cpu()
            del R_down, W_down, p_down, s_down, sh_down

            # Layer exclusive weights
            layer_f = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(layer_f, framework="pt") as lf:
                vram_dict[f"blk.{l}.attn_norm.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.input_layernorm.weight').to(torch.float16).cpu()
                vram_dict[f"blk.{l}.ffn_norm.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.post_attention_layernorm.weight').to(torch.float16).cpu()

                if is_attn:
                    # Softmax attention
                    vram_dict[f"blk.{l}.attn_q.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.q_proj.weight').cpu()
                    vram_dict[f"blk.{l}.attn_q.scale"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.attn_k.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.k_proj.weight').cpu()
                    vram_dict[f"blk.{l}.attn_k.scale"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.attn_v.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.v_proj.weight').cpu()
                    vram_dict[f"blk.{l}.attn_v.scale"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.attn_q_norm.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.q_norm.weight').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.attn_k_norm.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.self_attn.k_norm.weight').to(torch.float16).cpu()
                else:
                    # Linear attention (DeltaNet)
                    vram_dict[f"blk.{l}.attn_qkv.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight').cpu()
                    vram_dict[f"blk.{l}.attn_qkv.scale"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.attn_gate.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_z.weight').cpu()
                    vram_dict[f"blk.{l}.attn_gate.scale"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_alpha.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_a.weight').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_beta.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_b.weight').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_conv1d.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.conv1d.weight').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_dt.bias"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.dt_bias').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_a.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.A_log').to(torch.float16).cpu()
                    vram_dict[f"blk.{l}.ssm_norm.weight"] = lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.norm.weight').to(torch.float16).cpu()

        del cd, U_mix, V_mix, U_gate, V_gate, U_up, V_up, U_down, V_down
        torch.cuda.empty_cache()
        proc = psutil.Process()
        print(f"Carta {g+1:2d}/16 (Camadas {4*g:2d}..{4*g+3:2d}) compilada em {(time.time()-t_chart):.2f}s | RAM RSS: {proc.memory_info().rss/1e9:.2f} GB | VRAM: {torch.cuda.memory_allocated(0)/1e6:.1f} MB", flush=True)

    print(f"\nTodas as 64 camadas reconstruidas e quantizadas em {time.time()-t0_charts:.2f}s!", flush=True)

    # 3. Save Unified Safetensors
    print("\n[3/3] Gravando models/g_qwen_9b_vram.safetensors...", end="", flush=True)
    t0_save = time.time()
    st.save_file(vram_dict, out_safetensors_path)
    file_size_gb = os.path.getsize(out_safetensors_path) / (1024**3)
    print(f" Concluido em {time.time()-t0_save:.2f}s!", flush=True)

    print("\n" + "=" * 100)
    print("COMPILACAO VRAM SAFETENSORS FINALIZADA COM SUCESSO!")
    print(f"Arquivo Unificado : {out_safetensors_path}")
    print(f"Tamanho Final     : {file_size_gb:.2f} GB (100% RESIDENTE NA RTX 3060!)")
    print(f"Margem Livre VRAM : {12.0 - file_size_gb:.2f} GB livres na sua GPU de 12 GB")
    print("=" * 100, flush=True)


if __name__ == "__main__":
    compile_to_vram_safetensors()
