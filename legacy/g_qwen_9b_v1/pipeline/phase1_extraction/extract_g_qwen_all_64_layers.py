"""Full 64-Layer Extractor for G-Qwen 9B (All 16 Macro-Charts).

Processes all 64 layers (0..63) of Qwen 3.8 27B into 16 compressed Stiefel+RVQ charts.
Skips charts that are already extracted and cached (e.g. charts 0..3).
"""

import os
import sys
import glob
import time
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-use extraction functions from extract_g_qwen_phase1
from tools.extract_g_qwen_phase1 import (
    find_snapshot_dir,
    quantize_residual_4bit,
    dequantize_residual_4bit,
    load_dequantized_weight,
    extract_stiefel_frames_rect,
)


def extract_all_64_layers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("   EXTRACAO COMPLETA DE 64 CAMADAS: G-QWEN 9B (16 MACRO-CARTAS STIEFEL + RVQ 4-BIT)      ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    snapshot_dir = find_snapshot_dir()
    output_dir = os.path.abspath("models/g_qwen_9b_phase1")
    os.makedirs(output_dir, exist_ok=True)

    num_charts = 16 # All 64 layers: 0..63
    r_mixer = 2560
    r_ffn = 1792

    total_t0 = time.time()
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "architecture": "G-Qwen-9B-Phase1-64Layers",
            "rank_mixer": r_mixer,
            "rank_ffn": r_ffn,
            "residual_quantization": "4-bit uniform block 128",
            "charts": {},
            "summary_metrics": {},
        }

    all_mixer_errors = []
    all_gate_errors = []
    all_up_errors = []
    all_down_errors = []

    for g in range(num_charts):
        chart_path = os.path.join(output_dir, f"chart_{g}.safetensors")
        layers = [4 * g + i for i in range(4)]

        # If already extracted and verified, skip!
        if os.path.exists(chart_path) and os.path.getsize(chart_path) > 500 * 1024 * 1024:
            print(f"\n[Carta {g+1}/16: Camadas {layers[0]}..{layers[-1]}] Ja extraida ({os.path.getsize(chart_path)/(1024*1024):.1f} MB) - Pulando.", flush=True)
            continue

        # Check if all 4 layer files exist (wait if currently downloading)
        while True:
            missing_layers = [l for l in layers if not os.path.exists(os.path.join(snapshot_dir, f"layers-{l}.safetensors"))]
            if not missing_layers:
                break
            print(f"\n[Carta {g+1}/16: Camadas {layers[0]}..{layers[-1]}] Aguardando download das camadas: {missing_layers}...", flush=True)
            time.sleep(5)

        t_chart = time.time()
        print(f"\n{'='*45} CARTA {g+1}/16: CAMADAS {layers[0]}..{layers[-1]} {'='*45}", flush=True)
        chart_tensors: Dict[str, torch.Tensor] = {}

        # 1. Mixer
        print("  [1/4] Mixer out_proj...", end="", flush=True)
        mixer_weights = []
        for l in layers:
            is_attn = (l % 4 == 3)
            key = f"model.language_model.layers.{l}.self_attn.o_proj" if is_attn else f"model.language_model.layers.{l}.linear_attn.out_proj"
            w = load_dequantized_weight(snapshot_dir, l, key).to(device)
            mixer_weights.append(w)

        U_mixer, V_mixer = extract_stiefel_frames_rect(mixer_weights, r_mixer, device)
        chart_tensors["basis_mixer_U"] = U_mixer.to(torch.float16).cpu()
        chart_tensors["basis_mixer_V"] = V_mixer.to(torch.float16).cpu()

        for idx, l in enumerate(layers):
            w = mixer_weights[idx]
            S = torch.matmul(U_mixer.t(), torch.matmul(w, V_mixer))
            w_hat = torch.matmul(U_mixer, torch.matmul(S, V_mixer.t()))
            R = w - w_hat
            packed_q, scale = quantize_residual_4bit(R, block_size=128)
            
            R_rec = dequantize_residual_4bit(packed_q, scale, R.shape, block_size=128)
            err = (torch.norm(w - (w_hat + R_rec)) / torch.norm(w)).item() * 100
            all_mixer_errors.append(err)

            chart_tensors[f"layer_{l}_mixer_S"] = S.to(torch.float16).cpu()
            chart_tensors[f"layer_{l}_mixer_res_q"] = packed_q.cpu()
            chart_tensors[f"layer_{l}_mixer_res_scale"] = scale.cpu()

        del mixer_weights, U_mixer, V_mixer
        torch.cuda.empty_cache()
        print(" Concluido.", flush=True)

        # 2. Gate
        print("  [2/4] FFN gate_proj...", end="", flush=True)
        gate_weights = [load_dequantized_weight(snapshot_dir, l, f"model.language_model.layers.{l}.mlp.gate_proj").to(device) for l in layers]
        U_gate, V_gate = extract_stiefel_frames_rect(gate_weights, r_ffn, device)
        chart_tensors["basis_ffn_gate_U"] = U_gate.to(torch.float16).cpu()
        chart_tensors["basis_ffn_gate_V"] = V_gate.to(torch.float16).cpu()

        for idx, l in enumerate(layers):
            w = gate_weights[idx]
            S = torch.matmul(U_gate.t(), torch.matmul(w, V_gate))
            w_hat = torch.matmul(U_gate, torch.matmul(S, V_gate.t()))
            R = w - w_hat
            packed_q, scale = quantize_residual_4bit(R, block_size=128)
            
            R_rec = dequantize_residual_4bit(packed_q, scale, R.shape, block_size=128)
            err = (torch.norm(w - (w_hat + R_rec)) / torch.norm(w)).item() * 100
            all_gate_errors.append(err)

            chart_tensors[f"layer_{l}_ffn_gate_S"] = S.to(torch.float16).cpu()
            chart_tensors[f"layer_{l}_ffn_gate_res_q"] = packed_q.cpu()
            chart_tensors[f"layer_{l}_ffn_gate_res_scale"] = scale.cpu()

        del gate_weights, U_gate, V_gate
        torch.cuda.empty_cache()
        print(" Concluido.", flush=True)

        # 3. Up
        print("  [3/4] FFN up_proj...", end="", flush=True)
        up_weights = [load_dequantized_weight(snapshot_dir, l, f"model.language_model.layers.{l}.mlp.up_proj").to(device) for l in layers]
        U_up, V_up = extract_stiefel_frames_rect(up_weights, r_ffn, device)
        chart_tensors["basis_ffn_up_U"] = U_up.to(torch.float16).cpu()
        chart_tensors["basis_ffn_up_V"] = V_up.to(torch.float16).cpu()

        for idx, l in enumerate(layers):
            w = up_weights[idx]
            S = torch.matmul(U_up.t(), torch.matmul(w, V_up))
            w_hat = torch.matmul(U_up, torch.matmul(S, V_up.t()))
            R = w - w_hat
            packed_q, scale = quantize_residual_4bit(R, block_size=128)
            
            R_rec = dequantize_residual_4bit(packed_q, scale, R.shape, block_size=128)
            err = (torch.norm(w - (w_hat + R_rec)) / torch.norm(w)).item() * 100
            all_up_errors.append(err)

            chart_tensors[f"layer_{l}_ffn_up_S"] = S.to(torch.float16).cpu()
            chart_tensors[f"layer_{l}_ffn_up_res_q"] = packed_q.cpu()
            chart_tensors[f"layer_{l}_ffn_up_res_scale"] = scale.cpu()

        del up_weights, U_up, V_up
        torch.cuda.empty_cache()
        print(" Concluido.", flush=True)

        # 4. Down
        print("  [4/4] FFN down_proj...", end="", flush=True)
        down_weights = [load_dequantized_weight(snapshot_dir, l, f"model.language_model.layers.{l}.mlp.down_proj").to(device) for l in layers]
        U_down, V_down = extract_stiefel_frames_rect(down_weights, r_ffn, device)
        chart_tensors["basis_ffn_down_U"] = U_down.to(torch.float16).cpu()
        chart_tensors["basis_ffn_down_V"] = V_down.to(torch.float16).cpu()

        for idx, l in enumerate(layers):
            w = down_weights[idx]
            S = torch.matmul(U_down.t(), torch.matmul(w, V_down))
            w_hat = torch.matmul(U_down, torch.matmul(S, V_down.t()))
            R = w - w_hat
            packed_q, scale = quantize_residual_4bit(R, block_size=128)
            
            R_rec = dequantize_residual_4bit(packed_q, scale, R.shape, block_size=128)
            err = (torch.norm(w - (w_hat + R_rec)) / torch.norm(w)).item() * 100
            all_down_errors.append(err)

            chart_tensors[f"layer_{l}_ffn_down_S"] = S.to(torch.float16).cpu()
            chart_tensors[f"layer_{l}_ffn_down_res_q"] = packed_q.cpu()
            chart_tensors[f"layer_{l}_ffn_down_res_scale"] = scale.cpu()

        del down_weights, U_down, V_down
        torch.cuda.empty_cache()
        print(" Concluido.", flush=True)

        # Save chart
        chart_tensors = {k: v.contiguous() for k, v in chart_tensors.items()}
        save_file(chart_tensors, chart_path)
        chart_size = os.path.getsize(chart_path) / (1024 * 1024)
        c_time = time.time() - t_chart
        print(f"  --> Carta {g} salva ({chart_size:.1f} MB em {c_time:.2f}s)!", flush=True)

        manifest["charts"][f"chart_{g}"] = {
            "layers": layers,
            "filepath": chart_path,
            "size_mb": chart_size,
            "extraction_time_s": c_time,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    total_time = time.time() - total_t0
    print("\n" + "=" * 105, flush=True)
    print(f"[CONCLUIDO] Extracao das 64 Camadas Finalizada em {total_time:.2f}s ({total_time/60:.2f} min)!", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    extract_all_64_layers()
