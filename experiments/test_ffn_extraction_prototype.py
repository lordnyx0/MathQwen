"""Fast Prototype for FFN Gramian and Stiefel Extraction."""

import os
import glob
import time
import math
import torch
from safetensors import safe_open


def find_snapshot_dir() -> str:
    base_pattern = os.path.expanduser(
        r"~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"
    )
    return glob.glob(base_pattern)[0]


def load_dequant_ffn(snapshot_dir: str, layer_idx: int, proj_name: str) -> torch.Tensor:
    filepath = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")
    with safe_open(filepath, framework="pt") as f:
        w = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.{proj_name}.weight")
        s = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.{proj_name}.weight_scale_inv")
    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return w.to(torch.float32) * s_exp.to(torch.float32)


def test_ffn_extraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    snapshot_dir = find_snapshot_dir()

    t0 = time.time()
    print("Carregando 4 camadas de gate_proj (Carta 0)...", end="", flush=True)
    # gate_proj shape: [17408, 5120]
    weights = [load_dequant_ffn(snapshot_dir, i, "gate_proj").to(device) for i in range(4)]
    print(f" Concluido em {time.time() - t0:.2f}s", flush=True)

    # Gramian on the smaller dimension (5120): G_v = sum W^T W in R^{5120 x 5120}
    t_decomp = time.time()
    G_v = sum(torch.matmul(w.t(), w) for w in weights)
    _, V_full = torch.linalg.eigh(G_v)
    r_ffn = 1792
    V = torch.flip(V_full, dims=[1])[:, :r_ffn] # [5120, 1792]
    del G_v, V_full

    # Project onto V to get left projections: M_i = W_i @ V in R^{17408 x 1792}
    # Horizontal concat: M = [M_0, M_1, M_2, M_3] in R^{17408 x (4 * 1792)} = R^{17408 x 7168}
    M_list = [torch.matmul(w, V) for w in weights]
    M_all = torch.cat(M_list, dim=1) # [17408, 7168]
    
    # Fast Gramian of M_all in the column dimension: G_m = M_all^T M_all in R^{7168 x 7168}
    # Or thin SVD of M_all
    # Since 7168 is small, thin SVD of M_all:
    # Actually, left singular vectors of M_all can be obtained from M_all @ (thin SVD of M_all)
    U, _, _ = torch.linalg.svd(M_all, full_matrices=False)
    U = U[:, :r_ffn] # [17408, 1792]
    del M_all, M_list
    print(f"Decomposicao Stiefel FFN concluida em {time.time() - t_decomp:.2f}s!", flush=True)

    # Evaluate reconstruction error
    errs_base = []
    errs_rvq = []
    for w in weights:
        S = torch.matmul(U.t(), torch.matmul(w, V))
        w_hat = torch.matmul(U, torch.matmul(S, V.t()))
        R = w - w_hat
        err_base = (torch.norm(R) / torch.norm(w)).item() * 100
        errs_base.append(err_base)

        # 4-bit block quantization (blk=128)
        R_flat = R.view(-1, 128)
        max_v = torch.amax(torch.abs(R_flat), dim=-1, keepdim=True).clamp(min=1e-8)
        q_scale = max_v / 7.0
        q_int = torch.round(R_flat / q_scale).clamp(-7, 7)
        R_q = (q_int * q_scale).view(R.shape)

        w_hybrid = w_hat + R_q
        err_rvq = (torch.norm(w - w_hybrid) / torch.norm(w)).item() * 100
        errs_rvq.append(err_rvq)

    print(f"Erro Base FFN (gate_proj, r={r_ffn}): {sum(errs_base)/4.0:.2f}%")
    print(f"Erro Hibrido FFN (Atlas + RVQ 4-bit): {sum(errs_rvq)/4.0:.2f}%")


if __name__ == "__main__":
    test_ffn_extraction()
