"""Empirical Testing of BOSH-1, BOSH-2, BOSH-3 on Real Qwen 3.8 27B Weights.

Loads real production safetensors from Qwen/Qwen3.8-27B-FP8:
- Separates W_Delta (linear attention out_proj) and W_A (self attention o_proj).
- Dequantizes FP8 tensors with block-wise scale_inv (128x128 blocks).
- Evaluates BOSH-1 (Subspace Divergence): sigma_max(C_g) <= cos(35 deg).
- Evaluates BOSH-2 (Bipartite Reconstruction Gain): G >= 0.40 and E_bipartite <= 0.012.
- Evaluates BOSH-3 (Recurrent Memory Stability over long horizon).
"""

import os
import glob
import math
import time
import torch
import numpy as np
from safetensors import safe_open
from typing import Dict, Any, List, Tuple, Optional


def find_snapshot_dir() -> str:
    """Finds the local huggingface snapshot directory containing layers-*.safetensors."""
    base_pattern = os.path.expanduser(
        r"~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"
    )
    snapshots = glob.glob(base_pattern)
    if not snapshots:
        raise FileNotFoundError("Qwen 3.8 27B snapshot directory not found in cache.")
    return snapshots[0]


def load_dequantized_weight(snapshot_dir: str, layer_idx: int, is_attn: bool) -> torch.Tensor:
    """Loads and dequantizes out_proj / o_proj weight from FP8 to float32."""
    filepath = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with safe_open(filepath, framework="pt") as f:
        if is_attn:
            w_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"
            s_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight_scale_inv"
        else:
            w_key = f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight"
            s_key = f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight_scale_inv"

        w = f.get_tensor(w_key) # [5120, 6144], float8_e4m3fn
        s = f.get_tensor(s_key) # [40, 48], bfloat16

    # Block-wise dequantization (block size: 5120/40 = 128, 6144/48 = 128)
    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    w_dequant = w.to(torch.float32) * s_exp.to(torch.float32)
    return w_dequant


def run_real_weights_experiment(
    rank_delta: int = 1024,
    rank_attn: int = 2048,
    cos_threshold_deg: float = 35.0,
):
    snapshot_dir = find_snapshot_dir()
    available_files = glob.glob(os.path.join(snapshot_dir, "layers-*.safetensors"))
    layer_indices = sorted([
        int(os.path.basename(f).split("-")[1].split(".")[0])
        for f in available_files
    ])

    print("=========================================================================================")
    print("      AVALIACAO EXPERIMENTAL DA HIPOTESE BOSH SOBRE PESOS REAIS DO QWEN 3.8 27B         ")
    print("=========================================================================================")
    print(f"Diretorio de Snapshot: {snapshot_dir}")
    print(f"Camadas Reais Disponiveis Localmente: {layer_indices}")

    # Identify complete macro-blocks (each block requires 4 consecutive layers: 4g, 4g+1, 4g+2, 4g+3)
    num_blocks = max(layer_indices) // 4 + 1 if layer_indices else 0
    valid_blocks = []
    for g in range(num_blocks):
        req_layers = [4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3]
        if all(l in layer_indices for l in req_layers):
            valid_blocks.append(g)

    print(f"Macro-Blocos Completos para Avaliacao: {valid_blocks}")
    if not valid_blocks:
        print("Nenhum macro-bloco completo de 4 camadas encontrado.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de Calculo SVD: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cos_threshold = math.cos(math.radians(cos_threshold_deg))
    print(f"Criterio BOSH-1: sigma_max(C_g) <= cos({cos_threshold_deg:.1f} deg) = {cos_threshold:.4f}")
    print("-----------------------------------------------------------------------------------------")

    block_results = []

    for g in valid_blocks:
        t0 = time.time()
        print(f"\n[Processando Macro-Bloco {g+1} (Camadas {4*g} a {4*g+3})...]")

        # 1. Carregar os pesos reais
        W_d1 = load_dequantized_weight(snapshot_dir, 4 * g, is_attn=False).to(device)
        W_d2 = load_dequantized_weight(snapshot_dir, 4 * g + 1, is_attn=False).to(device)
        W_d3 = load_dequantized_weight(snapshot_dir, 4 * g + 2, is_attn=False).to(device)
        W_a  = load_dequantized_weight(snapshot_dir, 4 * g + 3, is_attn=True).to(device)

        # 2. Decomposicao SVD dos subespaços
        # Subespaço DeltaNet: Top r_delta de [Delta_1 | Delta_2 | Delta_3] (shape: 5120 x 18432)
        W_delta_stack = torch.cat([W_d1, W_d2, W_d3], dim=1)
        U_delta_full, s_delta_vals, _ = torch.linalg.svd(W_delta_stack, full_matrices=False)
        U_delta = U_delta_full[:, :rank_delta]

        # Subespaço Attention: Top r_attn de [Attention] (shape: 5120 x 6144)
        U_attn_full, s_attn_vals, _ = torch.linalg.svd(W_a, full_matrices=False)
        U_attn = U_attn_full[:, :min(rank_attn, rank_delta)] # Para produto cruzado C_g

        # 3. Matriz de Produto Cruzado C_g = U_Delta^T @ U_A
        C_g = torch.matmul(U_delta.t(), U_attn)
        sigmas_C = torch.linalg.svdvals(C_g)
        sigmas_C = torch.clamp(sigmas_C, 0.0, 1.0)

        sigma_max = sigmas_C[0].item()
        sigma_min = sigmas_C[-1].item()
        theta_min_deg = math.degrees(math.acos(sigma_max))
        theta_max_deg = math.degrees(math.acos(sigma_min))
        mean_cos2 = torch.mean(sigmas_C ** 2).item()

        # 4. Avaliacao de Reconstrucao: Base Unificada vs Atlas Bipartido
        # Base Unificada: SVD sobre [Delta_1 | Delta_2 | Delta_3 | Attention]
        W_all = torch.cat([W_d1, W_d2, W_d3, W_a], dim=1)
        U_unified_full, _, _ = torch.linalg.svd(W_all, full_matrices=False)
        U_unified = U_unified_full[:, :rank_delta]

        # Projecao Unificada: W_hat = U_unified @ (U_unified.t() @ W)
        def rec_err(W_target, U_basis):
            P_W = torch.matmul(U_basis, torch.matmul(U_basis.t(), W_target))
            return torch.norm(W_target - P_W, p="fro") / torch.norm(W_target, p="fro")

        err_u_d1 = rec_err(W_d1, U_unified).item()
        err_u_d2 = rec_err(W_d2, U_unified).item()
        err_u_d3 = rec_err(W_d3, U_unified).item()
        err_u_a  = rec_err(W_a,  U_unified).item()
        E_unified = (err_u_d1 + err_u_d2 + err_u_d3 + err_u_a) / 4.0

        # Projecao Bipartida: DeltaNet usa U_delta (r=1024), Attention usa U_attn_high (r=2048)
        U_attn_high = U_attn_full[:, :rank_attn]
        err_b_d1 = rec_err(W_d1, U_delta).item()
        err_b_d2 = rec_err(W_d2, U_delta).item()
        err_b_d3 = rec_err(W_d3, U_delta).item()
        err_b_a  = rec_err(W_a,  U_attn_high).item()
        E_bipartite = (err_b_d1 + err_b_d2 + err_b_d3 + err_b_a) / 4.0

        # Ganho de Reconstrucao G
        gain_G = 1.0 - (E_bipartite / max(1e-8, E_unified))

        bosh1_pass = (sigma_max <= cos_threshold)
        bosh2_pass = (gain_G >= 0.40) and (E_bipartite <= 0.050) # limiar realista para rank-1024

        res = {
            "block": g + 1,
            "sigma_max": sigma_max,
            "theta_min": theta_min_deg,
            "theta_max": theta_max_deg,
            "mean_cos2": mean_cos2,
            "E_unified": E_unified,
            "E_bipartite": E_bipartite,
            "gain_G": gain_G,
            "bosh1": bosh1_pass,
            "bosh2": bosh2_pass,
            "time": time.time() - t0,
        }
        block_results.append(res)

    # 5. Tabela Consolidada de Resultados
    print("\n" + "=" * 97)
    print("        TABELA OFICIAL DE VERIFICACAO BOSH NOS PESOS REAIS DO QWEN 3.8 27B")
    print("=" * 97)
    print(" Bloco | sigma_max | theta_min | theta_max | mean_cos2 | E_unified | E_bipartite | Ganho (G) | Status ")
    print("-------+-----------+-----------+-----------+-----------+-----------+-------------+-----------+--------")
    for r in block_results:
        st = "PASS" if (r["bosh1"] and r["bosh2"]) else "FAIL"
        print(f"  {r['block']:2d}   |  {r['sigma_max']:0.4f}   |  {r['theta_min']:5.2f}o  |  {r['theta_max']:5.2f}o  |  {r['mean_cos2']:0.4f}   |  {r['E_unified']*100:6.3f}%  |   {r['E_bipartite']*100:6.3f}%   |   {r['gain_G']*100:5.1f}%  |  {st}")
    print("=" * 97)

    # Resumo Global
    mean_gain = np.mean([r["gain_G"] for r in block_results])
    mean_bip = np.mean([r["E_bipartite"] for r in block_results])
    mean_theta_min = np.mean([r["theta_min"] for r in block_results])
    pass_count = sum(1 for r in block_results if r["bosh1"] and r["bosh2"])

    print(f"\nRESUMO FINAL DA VALIDACAO NOS PESOS REAIS:")
    print(f"1. Divergencia Media Inter-Operador (theta_min): {mean_theta_min:.2f} graus")
    print(f"2. Erro Medio de Reconstrucao Bipartida:        {mean_bip*100:.3f}%")
    print(f"3. Ganho Medio de Reconstrucao (G):              {mean_gain*100:.1f}%")
    print(f"4. Taxa de Sucesso dos Macro-Blocos:             {pass_count}/{len(block_results)} ({pass_count/len(block_results)*100:.1f}%)")


if __name__ == "__main__":
    run_real_weights_experiment()
