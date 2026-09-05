# -*- coding: utf-8 -*-
"""Módulo de exportação do modelo Atlas compactado e autônomo em FP8 bloco-128."""
import os
import sys
import json
import time
import shutil
import torch
import safetensors.torch as st

from .config import AtlasConfig
from .projection import compute_chart_atlas_bases
from reference.loader import get_snapshot_dir, dequant

def quantize_fp8_block128(w: torch.Tensor):
    """Quantiza um tensor 2D (múltiplo de 128) para torch.float8_e4m3fn em blocos de 128x128."""
    M, N = w.shape
    w_blocks = w.view(M // 128, 128, N // 128, 128).permute(0, 2, 1, 3)
    max_abs = w_blocks.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12)
    scale = max_abs / 448.0
    w_fp8 = (w_blocks / scale).to(torch.float8_e4m3fn).permute(0, 2, 1, 3).reshape(M, N)
    scale_inv = scale.squeeze(-1).squeeze(-1).to(torch.bfloat16)
    return w_fp8, scale_inv

def export_autonomous_atlas_model(
    output_dir: str = "checkpoints/atlas_autonomous",
    stabilizers_path: str = "checkpoints/atlas_stabilizers_adaptive.pt",
    r_base: int = 2048,
    device = None
):
    """
    Exporta o modelo Atlas compactado completo em FP8 bloco-128 (~13.8 GB).
    O runtime de inferência subsequente consome exclusivamente este diretório,
    sem tocar no checkpoint original do HuggingFace.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 90)
    print("EXPORTAÇÃO DO MODELO ATLAS COMPRIMIDO AUTÔNOMO (FP8 BLOCO-128)")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Diretório de Destino: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    snapshot_dir = get_snapshot_dir()

    # 1. Copiar/Vincular outside.safetensors via NTFS hardlink (0 bytes adicionais)
    src_outside = os.path.join(snapshot_dir, "outside.safetensors")
    dst_outside = os.path.join(output_dir, "outside.safetensors")
    if not os.path.exists(dst_outside):
        print("\n[1/4] Vinculando outside.safetensors via NTFS hardlink...")
        try:
            os.link(src_outside, dst_outside)
            print(f"  -> outside.safetensors vinculado com sucesso via hardlink (0 bytes adicionais).")
        except Exception as e:
            shutil.copy2(src_outside, dst_outside)
            print(f"  -> outside.safetensors copiado ({os.path.getsize(dst_outside)/(1024*1024):.1f} MB).")
    else:
        print("\n[1/4] outside.safetensors já presente no destino.")

    # 2. Exportar config.json canônico do Atlas
    print("\n[2/4] Gerando config.json canônico do Atlas...")
    c = AtlasConfig()
    cfg_dict = {
        "model_type": "mathqwen_atlas_autonomous",
        "format": "atlas_factorized_fp8_block128",
        "vocab_size": c.vocab_size,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_hidden_layers": c.num_hidden_layers,
        "num_charts": c.num_charts,
        "layers_per_chart": c.layers_per_chart,
        "r_base": r_base,
        "r_corr": c.r_corr,
        "tie_word_embeddings": c.tie_word_embeddings,
        "rms_norm_eps": c.rms_norm_eps
    }
    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"  -> config.json salvo em {cfg_path}.")

    # 3. Exportar as 16 cartas comprimidas em FP8 bloco-128
    print("\n[3/4] Comprimindo e exportando as 16 cartas do Atlas em FP8 bloco-128...")
    t_start = time.time()

    # Reutilizar bases já persistidas se disponíveis
    bases_cached_path = "checkpoints/atlas_bases.pt"
    cached_bases_dict = torch.load(bases_cached_path, map_location=device) if os.path.exists(bases_cached_path) else None
    if cached_bases_dict:
        print(f"  -> Bases Atlas carregadas do cache ({len(cached_bases_dict)} cartas).")

    for chart_idx in range(16):
        start_l = chart_idx * 4
        end_l = start_l + 4
        chart_file = os.path.join(charts_dir, f"chart_{chart_idx:02d}.pt")

        chart_weights = {}
        native_submodules = {}

        for l in range(start_l, end_l):
            lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(lp, framework="pt") as f:
                is_attn = (l % 4 == 3)
                if is_attn:
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight_scale_inv"))
                else:
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight_scale_inv"))
                gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv"))
                up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv"))
                down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv"))
                chart_weights[l] = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}

                # Preservar tensores não-MLP em seu formato compacto nativo FP8/BF16
                in_ln = f.get_tensor(f"model.language_model.layers.{l}.input_layernorm.weight").to(dtype=torch.bfloat16)
                post_ln = f.get_tensor(f"model.language_model.layers.{l}.post_attention_layernorm.weight").to(dtype=torch.bfloat16)

                sub = {"input_ln": in_ln, "post_ln": post_ln, "is_attn": is_attn}
                if is_attn:
                    sub["q_w"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight")
                    sub["q_s"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv")
                    sub["k_w"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight")
                    sub["k_s"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv")
                    sub["v_w"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight")
                    sub["v_s"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv")
                    sub["q_norm"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_norm.weight")
                    sub["k_norm"] = f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_norm.weight")
                else:
                    sub["qkv_w"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight")
                    sub["qkv_s"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv")
                    sub["z_w"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight")
                    sub["z_s"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv")
                    sub["in_proj_a"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_a.weight")
                    sub["in_proj_b"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_b.weight")
                    sub["conv1d"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.conv1d.weight")
                    sub["dt_bias"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.dt_bias")
                    sub["A_log"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.A_log")
                    sub["norm"] = f.get_tensor(f"model.language_model.layers.{l}.linear_attn.norm.weight")
                native_submodules[l] = sub

        if cached_bases_dict and chart_idx in cached_bases_dict:
            bases = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in cached_bases_dict[chart_idx].items()}
        else:
            bases = compute_chart_atlas_bases(chart_weights, r_base=r_base, device=device)

        U_m = bases["U_mix"]
        U_d = bases["U_down"]
        V_j = bases["V_joint"]

        chart_export_data = {
            "chart_idx": chart_idx,
            "bases": {
                "U_mix": U_m.cpu(),
                "U_down": U_d.cpu(),
                "V_joint": V_j.cpu()
            },
            "layers": {}
        }

        for l in range(start_l, end_l):
            w = chart_weights[l]
            w_mix = w["mix"].to(device=device, dtype=torch.bfloat16)
            w_gate = w["gate"].to(device=device, dtype=torch.bfloat16)
            w_up = w["up"].to(device=device, dtype=torch.bfloat16)
            w_down = w["down"].to(device=device, dtype=torch.bfloat16)

            C_mix = torch.matmul(U_m.t(), w_mix)
            C_gate = torch.matmul(w_gate, V_j)
            C_up = torch.matmul(w_up, V_j)
            C_down = torch.matmul(U_d.t(), w_down)

            # Quantizar fatores para FP8 bloco-128
            C_mix_fp8, C_mix_s = quantize_fp8_block128(C_mix)
            C_gate_fp8, C_gate_s = quantize_fp8_block128(C_gate)
            C_up_fp8, C_up_s = quantize_fp8_block128(C_up)
            C_down_fp8, C_down_s = quantize_fp8_block128(C_down)

            chart_export_data["layers"][l] = {
                "factors_fp8": {
                    "C_mix": C_mix_fp8.cpu(), "C_mix_s": C_mix_s.cpu(),
                    "C_gate": C_gate_fp8.cpu(), "C_gate_s": C_gate_s.cpu(),
                    "C_up": C_up_fp8.cpu(), "C_up_s": C_up_s.cpu(),
                    "C_down": C_down_fp8.cpu(), "C_down_s": C_down_s.cpu(),
                },
                "submodules": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in native_submodules[l].items()},
                "is_attn": w["is_attn"]
            }

        torch.save(chart_export_data, chart_file)
        c_sz = os.path.getsize(chart_file) / (1024 * 1024)
        print(f"  -> Carta {chart_idx:02d} (camadas {start_l:02d}..{end_l-1:02d}) exportada em FP8: {c_sz:.1f} MB.")

        del chart_weights, native_submodules, chart_export_data
        torch.cuda.empty_cache()

    t_total = time.time() - t_start
    print(f"[OK] 16 cartas comprimidas e exportadas em {t_total:.2f}s!")

    # 4. Copiar estabilizadores e bases Atlas para o diretório autônomo
    print("\n[4/4] Empacotando estabilizadores e bases...")
    dst_stab = os.path.join(output_dir, "stabilizers.pt")
    if os.path.exists(stabilizers_path):
        shutil.copy2(stabilizers_path, dst_stab)
        print(f"  -> Estabilizadores empacotados ({os.path.getsize(dst_stab)/(1024*1024):.1f} MB).")
    
    dst_bases = os.path.join(output_dir, "atlas_bases.pt")
    if os.path.exists("checkpoints/atlas_bases.pt") and not os.path.exists(dst_bases):
        try:
            os.link("checkpoints/atlas_bases.pt", dst_bases)
        except Exception:
            shutil.copy2("checkpoints/atlas_bases.pt", dst_bases)
        print(f"  -> atlas_bases.pt vinculado ({os.path.getsize(dst_bases)/(1024*1024):.1f} MB).")

    total_bytes = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(output_dir) for f in files)
    total_gb = total_bytes / (1024 ** 3)
    print("=" * 90)
    print(f"MODELO ATLAS COMPRIMIDO AUTÔNOMO EXPORTADO COM SUCESSO!")
    print(f"Diretório: {output_dir}")
    print(f"Tamanho Total em Disco: {total_gb:.2f} GB (vs ~27 GB do FP8 original)")
    print("=" * 90)

    return output_dir
