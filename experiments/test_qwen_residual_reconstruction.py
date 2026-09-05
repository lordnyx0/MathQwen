"""Empirical Evaluation of Residual Compensation Strategies for G-Qwen 9B.

Tests on real Qwen 3.8 27B production weights (Chart 0: Layers 0..3):
1. Base Shared Atlas at r = 2560: Delta W_F and Delta W_2.
2. Low-rank SVD residual adapters: A B^T with rank in [64, 128, 256].
3. Quantized dense residuals: 2-bit, 3-bit, 4-bit block quantization on R = W - W_hat.
4. Parameter footprint and verification against the 9.00B budget.
"""

import os
import glob
import math
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open


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


def quantize_block_uniform(tensor: torch.Tensor, n_bits: int, block_size: int = 128) -> torch.Tensor:
    """Simulates symmetric uniform block quantization (e.g. 2-bit, 3-bit, 4-bit)."""
    orig_shape = tensor.shape
    t_flat = tensor.view(-1, block_size)
    
    # Scale per block
    max_val = torch.amax(torch.abs(t_flat), dim=-1, keepdim=True).clamp(min=1e-8)
    q_max = (1 << (n_bits - 1)) - 1
    
    scale = max_val / q_max
    q_int = torch.round(t_flat / scale).clamp(-q_max, q_max)
    dequant = q_int * scale
    return dequant.view(orig_shape)


def run_residual_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("         AVALIACAO EXPERIMENTAL DE COMPENSACAO RESIDUAL NOS PESOS REAIS DO QWEN 3.8 27B          ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    snapshot_dir = find_snapshot_dir()
    
    # Load 4 layers of Chart 0 (3 DeltaNet + 1 Attention)
    print("\n[Carregando Pesos da Carta 0: Camadas 0..3]...", end="", flush=True)
    t0 = time.time()
    w_d0 = load_dequantized_weight(snapshot_dir, 0, is_attn=False).to(device)
    w_d1 = load_dequantized_weight(snapshot_dir, 1, is_attn=False).to(device)
    w_d2 = load_dequantized_weight(snapshot_dir, 2, is_attn=False).to(device)
    w_attn = load_dequantized_weight(snapshot_dir, 3, is_attn=True).to(device)
    weights = [w_d0, w_d1, w_d2, w_attn]
    print(f" Concluido em {time.time() - t0:.2f}s", flush=True)

    # 1. Compute Shared Atlas at r = 2560
    r_shared = 2560
    print(f"[Computando Base Stiefel Compartilhada r = {r_shared}]...", end="", flush=True)
    G_u = torch.matmul(w_d0, w_d0.t()) + torch.matmul(w_d1, w_d1.t()) + torch.matmul(w_d2, w_d2.t()) + torch.matmul(w_attn, w_attn.t())
    _, U_full = torch.linalg.eigh(G_u)
    U = torch.flip(U_full, dims=[1])[:, :r_shared]

    G_v = torch.matmul(w_d0.t(), w_d0) + torch.matmul(w_d1.t(), w_d1) + torch.matmul(w_d2.t(), w_d2) + torch.matmul(w_attn.t(), w_attn)
    _, V_full = torch.linalg.eigh(G_v)
    V = torch.flip(V_full, dims=[1])[:, :r_shared]
    print(" Concluido.", flush=True)

    # Base projection and residuals
    w_hat_base = []
    residuals = []
    for w in weights:
        S = torch.matmul(U.t(), torch.matmul(w, V))
        w_h = torch.matmul(U, torch.matmul(S, V.t()))
        w_hat_base.append(w_h)
        residuals.append(w - w_h)

    # Base Shared Atlas Error
    base_errors = [(torch.norm(residuals[i]) / torch.norm(weights[i])).item() * 100 for i in range(4)]
    mean_base_err = sum(base_errors) / 4.0
    print(f"\nErro Base do Shared Atlas (r=2560 sem residuo): {mean_base_err:.2f}%\n", flush=True)

    # -------------------------------------------------------------
    # 2. Evaluate Low-Rank SVD Residual Adapters
    # -------------------------------------------------------------
    print("=" * 105, flush=True)
    print("                  ESTRATEGIA A: RESIDUOS DE BAIXO POSTO SVD POR CAMADA (A_l B_l^T)              ", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Posto Residual (k)':<20} | {'Delta W_F':<15} | {'Reducao Erro':<15} | {'Parametros Adicionais (64L)':<30} | {'Total Modelo':<15}", flush=True)
    print("-" * 105, flush=True)

    for k_res in [32, 64, 128, 256, 512]:
        errs = []
        for i in range(4):
            R = residuals[i]
            # Thin SVD of residual
            u_r, s_r, v_r = torch.linalg.svd(R, full_matrices=False)
            R_k = torch.matmul(u_r[:, :k_res] * s_r[:k_res], v_r[:k_res, :])
            w_comp = w_hat_base[i] + R_k
            err = (torch.norm(weights[i] - w_comp) / torch.norm(weights[i])).item() * 100
            errs.append(err)
        
        m_err = sum(errs) / 4.0
        reduction = mean_base_err - m_err
        # Additional params across 64 layers: 64 x (5120 * k + 6144 * k) = 64 * 11264 * k
        add_params = 64 * (5120 + 6144) * k_res
        total_p = 7_005_650_944 + add_params
        total_b = total_p / 1e9
        print(f"k = {k_res:<16} | {m_err:>13.2f}% | {reduction:>+13.2f}% | {add_params:>12,} ({add_params/1e6:>6.1f}M)    | {total_b:>6.2f}B", flush=True)

    print("=" * 105, flush=True)

    # -------------------------------------------------------------
    # 3. Evaluate Dense Quantized Residuals (Block Uniform)
    # -------------------------------------------------------------
    print("\n" + "=" * 105, flush=True)
    print("                  ESTRATEGIA B: RESIDUOS DENSOS QUANTIZADOS POR BLOCO (RVQ)                    ", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Bits / Elemento':<20} | {'Delta W_F':<15} | {'Reducao Erro':<15} | {'Memoria Adicional (64L)':<30} | {'Total Modelo (Equiv)' :<20}", flush=True)
    print("-" * 105, flush=True)

    # Matrix size: 5120 x 6144 = 31,457,280 elements
    total_elements_64L = 64 * (5120 * 6144) # 2,013,265,920 elements

    for n_bits in [1, 2, 3, 4, 8]:
        errs = []
        for i in range(4):
            R = residuals[i]
            R_q = quantize_block_uniform(R, n_bits=n_bits, block_size=128)
            w_comp = w_hat_base[i] + R_q
            err = (torch.norm(weights[i] - w_comp) / torch.norm(weights[i])).item() * 100
            errs.append(err)

        m_err = sum(errs) / 4.0
        reduction = mean_base_err - m_err
        
        # Memory in MB: total_elements * n_bits / 8 / 1e6
        mem_mb = (total_elements_64L * n_bits) / (8 * 1024 * 1024)
        # FP32-equivalent parameter count: (total_elements * n_bits) / 32
        fp32_equiv = int((total_elements_64L * n_bits) / 32)
        total_p = 7_005_650_944 + fp32_equiv
        total_b = total_p / 1e9

        print(f"{n_bits}-bit uniform (blk 128) | {m_err:>13.2f}% | {reduction:>+13.2f}% | {mem_mb:>10.1f} MB ({fp32_equiv/1e6:>6.1f}M eq)  | {total_b:>6.2f}B", flush=True)

    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_residual_experiment()
