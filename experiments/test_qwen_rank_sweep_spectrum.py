"""Empirical Rank Sweep and Spectral Tail Analysis on Real Qwen 3.8 27B Weights.

Evaluates across 16 physical layers (Charts 0..3):
1. Exact singular spectrum decay of individual weight matrices (Unshared Lower Bound).
2. Shared Atlas projection error curve across ranks r in [256..5120].
3. Cumulative energy tail ratio sum_{i > r} sigma_i^2 / sum sigma_i^2.
4. Identifies critical ranks r_50%, r_20%, r_10%, r_5%, r_2%, r_1%.
5. Computes physical parameter count P(r) and tests whether 9.00B is mathematically sufficient.
"""

import os
import glob
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from typing import Dict, Any, List, Tuple


def find_snapshot_dir() -> str:
    base_pattern = os.path.expanduser(
        r"~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"
    )
    snapshots = glob.glob(base_pattern)
    if not snapshots:
        raise FileNotFoundError("Qwen 3.8 27B snapshot directory not found in cache.")
    return snapshots[0]


def load_dequantized_weight(snapshot_dir: str, layer_idx: int, is_attn: bool) -> torch.Tensor:
    filepath = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")
    with safe_open(filepath, framework="pt") as f:
        if is_attn:
            w_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"
            s_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight_scale_inv"
        else:
            w_key = f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight"
            s_key = f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight_scale_inv"
        w = f.get_tensor(w_key)
        s = f.get_tensor(s_key)

    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return w.to(torch.float32) * s_exp.to(torch.float32)


def project_stiefel(W: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Projects W onto Stiefel frames U in St(d_out, r) and V in St(d_in, r): W_hat = U (U^T W V) V^T."""
    S = torch.matmul(U.t(), torch.matmul(W, V))
    return torch.matmul(U, torch.matmul(S, V.t()))


def compute_physical_parameters(rank: int) -> int:
    """Computes total physical parameters of G-Qwen 9B as a function of chart rank r."""
    vocab_size = 152064
    d_model = 5120
    d_v = 6144
    d_ffn = 17408
    num_charts = 16
    num_layers = 64
    res_rank = 64

    embed = vocab_size * d_model # 778.5M
    
    # FFN bases: 16 charts x 3 projections x (d_model * 1792 + d_ffn * 1792)
    ffn_rank = 1792
    ffn_bases = num_charts * 3 * (d_model * ffn_rank + d_ffn * ffn_rank) # 1.94B
    
    # Mixer bases: 16 charts x (d_model * rank + d_v * rank)
    mixer_bases = num_charts * (d_model + d_v) * rank
    
    # FFN layer coords and residuals
    ffn_coords = 3 * (ffn_rank + 2 * ffn_rank * 128)
    ffn_res = 3 * (d_model * res_rank + d_ffn * res_rank)
    
    # Mixer layer coords and residuals
    mixer_coords = rank + 2 * rank * 128
    mixer_res = (d_v + d_model) * res_rank
    
    # Exclusive layer weights (norms, biases, gates, q/k projections)
    exclusive_per_layer = (d_model * 2048) * 2 + (d_model * d_v) + (d_model * 48) + 4 * (4096 + d_v) + 2 * d_model
    
    layer_total = num_layers * (ffn_coords + ffn_res + mixer_coords + mixer_res + exclusive_per_layer)
    
    return embed + ffn_bases + mixer_bases + layer_total


def run_rank_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("       SWEEP DE POSTO E ANALISE ESPECTRAL DE ECKART-YOUNG NOS PESOS REAIS DO QWEN 3.8 27B       ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    snapshot_dir = find_snapshot_dir()
    num_charts = 4
    ranks = [256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]

    print(f"Ranks sob varredura: {ranks}", flush=True)
    print(f"Camadas analisadas: 16 camadas físicas (4 macro-cartas de 4 camadas cada)", flush=True)
    print("-" * 105, flush=True)

    t0 = time.time()
    
    # Accumulators across all 16 layers
    # For each rank, list of errors across 16 layers
    shared_frobenius_errors = {r: [] for r in ranks}
    unshared_best_errors = {r: [] for r in ranks}
    all_singular_values = []

    for g in range(num_charts):
        t_chart = time.time()
        print(f"\n[Carta {g+1}/4: Camadas {4*g}..{4*g+3}] Carregando tensores...", end="", flush=True)

        chart_layers = [4 * g + i for i in range(4)]
        w_d0 = load_dequantized_weight(snapshot_dir, chart_layers[0], is_attn=False).to(device)
        w_d1 = load_dequantized_weight(snapshot_dir, chart_layers[1], is_attn=False).to(device)
        w_d2 = load_dequantized_weight(snapshot_dir, chart_layers[2], is_attn=False).to(device)
        w_attn = load_dequantized_weight(snapshot_dir, chart_layers[3], is_attn=True).to(device)
        weights_chart = [w_d0, w_d1, w_d2, w_attn]

        # 1. Compute Individual Layer SVD Spectrums (Unshared Theoretical Lower Bound)
        for w in weights_chart:
            # Thin SVD of 5120 x 6144
            s = torch.linalg.svdvals(w) # 5120 singular values in descending order
            all_singular_values.append(s.cpu())
            s_sq = s ** 2
            total_energy = torch.sum(s_sq).item()
            
            for r in ranks:
                tail_energy = torch.sum(s_sq[r:]).item() if r < 5120 else 0.0
                unshared_err = math.sqrt(tail_energy / total_energy)
                unshared_best_errors[r].append(unshared_err)

        # 2. Compute Joint Gramians for the Shared Atlas
        G_u = torch.matmul(w_d0, w_d0.t()) + torch.matmul(w_d1, w_d1.t()) + torch.matmul(w_d2, w_d2.t()) + torch.matmul(w_attn, w_attn.t())
        _, U_full = torch.linalg.eigh(G_u)
        U_full = torch.flip(U_full, dims=[1]) # [5120, 5120]

        G_v = torch.matmul(w_d0.t(), w_d0) + torch.matmul(w_d1.t(), w_d1) + torch.matmul(w_d2.t(), w_d2) + torch.matmul(w_attn.t(), w_attn)
        _, V_full = torch.linalg.eigh(G_v)
        V_full = torch.flip(V_full, dims=[1]) # [6144, 6144]

        # 3. Project each layer across all ranks
        for r in ranks:
            U_r = U_full[:, :r]
            V_r = V_full[:, :r]
            for w in weights_chart:
                w_hat = project_stiefel(w, U_r, V_r)
                frob_err = (torch.norm(w - w_hat) / torch.norm(w)).item()
                shared_frobenius_errors[r].append(frob_err)

        del w_d0, w_d1, w_d2, w_attn, weights_chart, G_u, G_v, U_full, V_full
        torch.cuda.empty_cache()
        print(f" Concluido em {time.time() - t_chart:.2f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nVarredura completa de 16 camadas finalizada em {total_time:.2f}s!", flush=True)

    # -----------------------------------------------------------------
    # AGGREGATE RESULTS TABLE
    # -----------------------------------------------------------------
    print("\n" + "=" * 105, flush=True)
    print("                     CURVA DE DECAIMENTO DE ERRO vs POSTO (SHARED ATLAS vs UNSHARED SVD)                  ", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Posto (r)':<10} | {'Parametros (P)':<16} | {'Orcamento <= 9B?':<16} | {'Unshared SVD (Opt)':<20} | {'Shared Atlas (Real)':<20} | {'Penalidade Compartilhamento':<22}", flush=True)
    print("-" * 105, flush=True)

    knee_detected = None
    for r in ranks:
        params = compute_physical_parameters(r)
        params_b = params / 1e9
        is_under_9b = "SIM (<= 9.00B)" if params <= 9_000_000_000 else "EXCEDE (> 9.00B)"
        
        opt_err = np.mean(unshared_best_errors[r]) * 100
        shared_err = np.mean(shared_frobenius_errors[r]) * 100
        overhead = shared_err - opt_err

        print(f"{r:<10} | {params_b:>6.2f}B ({params:>10,}) | {is_under_9b:<16} | {opt_err:>18.2f}% | {shared_err:>18.2f}% | {overhead:>+20.2f}%", flush=True)

    print("=" * 105, flush=True)

    # -----------------------------------------------------------------
    # CRITICAL RANKS FOR ERROR THRESHOLDS
    # -----------------------------------------------------------------
    print("\n" + "=" * 105, flush=True)
    print("               POSTOS CRITICOS NECESSARIOS PARA ATINGIR TARGETS DE ERRO (r_epsilon)               ", flush=True)
    print("=" * 105, flush=True)
    
    # Compute mean singular spectrum across all 16 layers
    all_s_matrix = torch.stack(all_singular_values, dim=0) # [16, 5120]
    mean_s = torch.mean(all_s_matrix, dim=0)
    mean_s_sq = mean_s ** 2
    total_e = torch.sum(mean_s_sq).item()

    targets = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01]
    print(f"{'Erro Alvo (epsilon)':<20} | {'Posto Minimo Otimo (Unshared)':<30} | {'Parametros no Modelo':<22} | {'Viavel em 9B?':<15}", flush=True)
    print("-" * 105, flush=True)

    for eps in targets:
        # Find minimum rank such that sqrt(tail / total) <= eps
        r_opt = 5120
        for i in range(1, 5120):
            tail = torch.sum(mean_s_sq[i:]).item()
            if math.sqrt(tail / total_e) <= eps:
                r_opt = i
                break
        
        cost_params = compute_physical_parameters(r_opt)
        cost_b = cost_params / 1e9
        feasible = "SIM" if cost_params <= 9_000_000_000 else "NAO (Excede)"

        print(f"{eps*100:>5.1f}% ({eps:.2f})       | r >= {r_opt:<24} | {cost_b:>6.2f}B ({cost_params:>10,}) | {feasible:<15}", flush=True)

    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_rank_sweep()
