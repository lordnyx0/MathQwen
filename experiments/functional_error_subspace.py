# -*- coding: utf-8 -*-
"""
Sensibilidade Funcional via Jacobiana Ponderada em L63
Calcula J(x) = d z / d h no cabecote LMHead o RMSNorm,
G_J = E[J^T J], Sigma_functional = G_J^(1/2) Sigma_E G_J^(1/2),
e extrai o subespaco funcional U_c_func, comparando o alinhamento
com o subespaco puramente geometrico U_c.
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

repo_root = r"C:\Users\Nyx\Desktop\MathQwen"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, get_snapshot_dir

def compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor):
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    angles_deg = torch.acos(S) * (180.0 / math.pi)
    return S.cpu().numpy(), angles_deg.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print("SENSIBILIDADE FUNCIONAL VIA JACOBIANA PONDERADA (PHASE 2 - L63)")
    print("=" * 105)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Componentes da Cabeca (LMHead e RMSNorm)
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=16)
    lm_head_w = comp["lm_head_w"].float() # (151936, 5120)
    norm_weight = comp["norm_mod"].weight.data.float() # (5120,)
    d_model = 5120

    print("\n[Passo 1/4] Computando Gramiano do LMHead: G_head = W_head^T W_head...")
    t0 = time.time()
    # G_head in R^{5120 x 5120}
    G_head = torch.matmul(lm_head_w.t(), lm_head_w)
    print(f"  -> Gramiano computado em {time.time() - t0:.2f}s | Tr(G_head) = {torch.trace(G_head).item():.2e}")

    # 2. Computar Operador Medio da Jacobiana G_J = E[J^T J]
    print("\n[Passo 2/4] Avaliando Jacobiana media J = d z / d h sobre estados de ativacao...")
    h_wiki_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    X_wiki = torch.load(h_wiki_path, map_location=device).reshape(-1, d_model).float()
    N_sample = min(1024, X_wiki.shape[0])
    X_sample = X_wiki[:N_sample]

    # Para RMSNorm: s(h) = (1/d ||h||^2 + eps)^(-1/2)
    # dx_norm / dh = s(h) diag(gamma) (I - h h^T / (d mu_2))
    eps = 1e-6
    gamma = norm_weight.view(1, d_model)

    # Computar G_J acumulado
    G_J = torch.zeros(d_model, d_model, device=device)
    batch_sz = 128
    for b_start in range(0, N_sample, batch_sz):
        b_end = min(b_start + batch_sz, N_sample)
        H_b = X_sample[b_start:b_end] # (B, 5120)
        B = H_b.shape[0]

        mu2 = (H_b ** 2).mean(dim=-1, keepdim=True) + eps # (B, 1)
        s = torch.rsqrt(mu2) # (B, 1)

        # Matriz Jacobiana de RMSNorm para cada exemplo:
        # D_i = s_i * diag(gamma) - (s_i / (d * mu2_i)) * (diag(gamma) @ h_i @ h_i^T)
        # Aproximacao media de primeira ordem altamente estavel:
        s_mean = s.mean().item()
        mu2_mean = mu2.mean().item()
        D_mean = s_mean * torch.diag(norm_weight) - (s_mean / (d_model * mu2_mean)) * torch.matmul(torch.diag(norm_weight), torch.matmul(H_b.t(), H_b) / float(B))
        
        G_J_b = torch.matmul(D_mean.t(), torch.matmul(G_head, D_mean))
        G_J += G_J_b * (float(B) / float(N_sample))

    tr_GJ = torch.trace(G_J).item()
    print(f"  -> G_J concluido: Tr(G_J) = {tr_GJ:.2e}")

    # Raiz quadrada simetrica de G_J via eigh: G_J = V D V^T -> G_J^(1/2) = V D^(1/2) V^T
    print("  -> Decompondo G_J para obter G_J^(1/2)...")
    evals_GJ, evecs_GJ = torch.linalg.eigh(G_J)
    evals_GJ = torch.clamp(evals_GJ, min=1e-12)
    G_J_half = torch.matmul(evecs_GJ, torch.matmul(torch.diag(torch.sqrt(evals_GJ)), evecs_GJ.t()))

    # 3. Covariancias Funcionais Ponderadas
    print("\n[Passo 3/4] Construindo Covariancias Funcionais Sigma_functional = G_J^(1/2) Sigma_E G_J^(1/2)...")
    err_path = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    raw_errors = torch.load(err_path, map_location=device)
    domains = ["WikiText", "HF_Code", "GSM8K"]

    sigmas_func = {}
    sigmas_geom = {}
    for d in domains:
        E = raw_errors[d].to(device).float()
        Sigma = torch.matmul(E.t(), E) / float(E.shape[0])
        sigmas_geom[d] = Sigma
        Sigma_f = torch.matmul(G_J_half, torch.matmul(Sigma, G_J_half))
        sigmas_func[d] = Sigma_f

    Sigma_geom_avg = torch.stack(list(sigmas_geom.values()), dim=0).mean(dim=0)
    Sigma_func_avg = torch.stack(list(sigmas_func.values()), dim=0).mean(dim=0)

    # Autovalores do Nucleo Geometrico vs Funcional
    _, eVecs_geom = torch.linalg.eigh(Sigma_geom_avg)
    eVecs_geom = torch.flip(eVecs_geom, dims=[1])

    _, eVecs_func = torch.linalg.eigh(Sigma_func_avg)
    eVecs_func = torch.flip(eVecs_func, dims=[1])

    # 4. Comparacao de Alinhamento: Geometrico vs Funcional
    print("\n[Passo 4/4] Comparando Alinhamento: Subespaco Geometrico U_c vs Funcional U_c_func...")
    ranks = [4, 8, 16, 32, 64]
    comparison_records = {}

    print(f"{'Rank':<6} | {'Overlap Grassmanniano':<24} | {'Melhor Angulo (deg)':<22} | {'Angulo Medio (deg)':<20} | {'Pior Angulo (deg)':<18}")
    print("-" * 105)
    for r in ranks:
        U_geom = eVecs_geom[:, :r]
        U_func = eVecs_func[:, :r]
        s_vals, angles_deg = compute_principal_angles(U_geom, U_func)

        overlap = float(np.mean(s_vals))
        best_a = float(np.min(angles_deg))
        mean_a = float(np.mean(angles_deg))
        worst_a = float(np.max(angles_deg))

        comparison_records[str(r)] = {
            "overlap": overlap,
            "best_angle_deg": best_a,
            "mean_angle_deg": mean_a,
            "worst_angle_deg": worst_a,
            "singular_values": s_vals.tolist()
        }
        print(f"{r:<6} | {overlap * 100.0:6.2f}%                  | {best_a:6.2f}°                 | {mean_a:6.2f}°                | {worst_a:6.2f}°")

    # Salvar resultados
    out_json = os.path.join(repo_root, "experiments", "functional_error_subspace.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tr_G_head": torch.trace(G_head).item(),
            "tr_G_J": tr_GJ,
            "ranks": ranks,
            "comparison_records": comparison_records
        }, f, indent=2)
    print(f"\n[OK] Analise funcional concluida e salva em: {out_json}")

if __name__ == "__main__":
    main()
