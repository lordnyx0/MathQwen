# -*- coding: utf-8 -*-
"""
Common + Specific Error Manifold: Bootstrap Rigoroso da Estabilidade Amostral
Executa 50 resamplagens bootstrap para cada ambiente e para o nucleo comum Uc.
Calcula distancia geodesica no Grassmanniano d_Gr(U^A, U^B), angulo medio,
overlap e percentis (5%, 25%, 50%, 75%, 95%, pior caso) para r in [4, 8, 16, 32, 64, 128].
"""
import os
import sys
import time
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

repo_root = r"C:\Users\Nyx\Desktop\MathQwen"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def compute_subspace_metrics(U1: torch.Tensor, U2: torch.Tensor):
    """
    Calcula metricas entre dois subespacos ortonormais U1, U2 in R^{d x r}:
    - S: valores singulares de U1^T U2
    - angles_deg: angulos canonicos theta_i em graus
    - d_gr_deg: distancia geodesica no Grassmanniano sqrt(sum theta_i^2) em graus
    - d_chordal: distancia chordal sqrt(sum sin^2 theta_i)
    - overlap: media dos cos(theta_i)
    """
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    angles_rad = torch.acos(S)
    angles_deg = angles_rad * (180.0 / math.pi)
    
    d_gr_rad = torch.sqrt(torch.sum(angles_rad ** 2)).item()
    d_gr_deg = d_gr_rad * (180.0 / math.pi)
    d_chordal = torch.sqrt(torch.sum(torch.sin(angles_rad) ** 2)).item()
    overlap = torch.mean(S).item()
    mean_angle = torch.mean(angles_deg).item()
    worst_angle = torch.max(angles_deg).item()

    return {
        "d_gr_deg": d_gr_deg,
        "d_gr_rad": d_gr_rad,
        "d_chordal": d_chordal,
        "overlap": overlap,
        "mean_angle_deg": mean_angle,
        "worst_angle_deg": worst_angle
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("COMMON + SPECIFIC ERROR MANIFOLD - BOOTSTRAP DE ESTABILIDADE NO GRASSMANNIANO")
    print("=" * 100)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    err_path = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    raw_errors = torch.load(err_path, map_location=device)
    domains = ["WikiText", "HF_Code", "GSM8K"]
    E_dict = {d: raw_errors[d].to(device).float() for d in domains}
    N = 2048
    half = N // 2
    ranks = [4, 8, 16, 32, 64, 128]
    n_boot = 50

    print(f"\n[Passo 1/3] Executando {n_boot} iteracoes de bootstrap para r in {ranks}...")
    bootstrap_records = {d: {str(r): [] for r in ranks} for d in domains}
    bootstrap_records["Common_Core_Uc"] = {str(r): [] for r in ranks}

    torch.manual_seed(42)
    np.random.seed(42)

    for b_idx in range(n_boot):
        # Gerar splits A e B para cada dominio
        splits_A = {}
        splits_B = {}
        sigmas_A = {}
        sigmas_B = {}

        for d in domains:
            perm = torch.randperm(N, device=device)
            idx_A = perm[:half]
            idx_B = perm[half:]
            EA = E_dict[d][idx_A]
            EB = E_dict[d][idx_B]
            splits_A[d] = EA
            splits_B[d] = EB
            sigmas_A[d] = torch.matmul(EA.t(), EA) / float(half)
            sigmas_B[d] = torch.matmul(EB.t(), EB) / float(half)

        # Sigma Medio Comum para A e B
        Sigma_avg_A = torch.stack(list(sigmas_A.values()), dim=0).mean(dim=0)
        Sigma_avg_B = torch.stack(list(sigmas_B.values()), dim=0).mean(dim=0)

        # Autodecomposicoes
        for d in domains:
            _, eVecs_A = torch.linalg.eigh(sigmas_A[d])
            _, eVecs_B = torch.linalg.eigh(sigmas_B[d])
            # Ordem decrescente
            eVecs_A = torch.flip(eVecs_A, dims=[1])
            eVecs_B = torch.flip(eVecs_B, dims=[1])

            for r in ranks:
                U_A = eVecs_A[:, :r]
                U_B = eVecs_B[:, :r]
                m = compute_subspace_metrics(U_A, U_B)
                bootstrap_records[d][str(r)].append(m)

        # Decomposicao do Nucleo Comum Uc
        _, eVecs_c_A = torch.linalg.eigh(Sigma_avg_A)
        _, eVecs_c_B = torch.linalg.eigh(Sigma_avg_B)
        eVecs_c_A = torch.flip(eVecs_c_A, dims=[1])
        eVecs_c_B = torch.flip(eVecs_c_B, dims=[1])

        for r in ranks:
            Uc_A = eVecs_c_A[:, :r]
            Uc_B = eVecs_c_B[:, :r]
            m_c = compute_subspace_metrics(Uc_A, Uc_B)
            bootstrap_records["Common_Core_Uc"][str(r)].append(m_c)

        if (b_idx + 1) % 10 == 0:
            print(f"  -> Bootstrap progresso: {b_idx + 1}/{n_boot} iteracoes concluidas.")

    # [Passo 2/3] Sumarizacao Estatistica (Media, Std, Percentis, Pior Caso)
    print("\n[Passo 2/3] Calculando metricas consolidadas...")
    summary = {}
    target_keys = domains + ["Common_Core_Uc"]

    for k in target_keys:
        summary[k] = {}
        for r in ranks:
            recs = bootstrap_records[k][str(r)]
            overlaps = [x["overlap"] for x in recs]
            d_grs = [x["d_gr_deg"] for x in recs]
            mean_angles = [x["mean_angle_deg"] for x in recs]
            worst_angles = [x["worst_angle_deg"] for x in recs]

            summary[k][str(r)] = {
                "overlap": {
                    "mean": float(np.mean(overlaps)),
                    "std": float(np.std(overlaps)),
                    "p5": float(np.percentile(overlaps, 5)),
                    "p25": float(np.percentile(overlaps, 25)),
                    "p50_median": float(np.percentile(overlaps, 50)),
                    "p75": float(np.percentile(overlaps, 75)),
                    "p95": float(np.percentile(overlaps, 95)),
                    "worst_min": float(np.min(overlaps))
                },
                "d_gr_deg": {
                    "mean": float(np.mean(d_grs)),
                    "std": float(np.std(d_grs)),
                    "p5": float(np.percentile(d_grs, 5)),
                    "p50_median": float(np.percentile(d_grs, 50)),
                    "p95": float(np.percentile(d_grs, 95)),
                    "worst_max": float(np.max(d_grs))
                },
                "mean_angle_deg": {
                    "mean": float(np.mean(mean_angles)),
                    "std": float(np.std(mean_angles)),
                    "worst_max": float(np.max(mean_angles))
                },
                "worst_angle_deg": {
                    "mean": float(np.mean(worst_angles)),
                    "worst_max": float(np.max(worst_angles))
                }
            }

    # Imprimir Tabela de Estabilidade do Nucleo Comum e dos Ambientes
    print("\n" + "=" * 115)
    print("ESTABILIDADE DO NUCLEO COMUM (Uc) E DOS AMBIENTES SOB BOOTSTRAP (50 REAMOSTRAGENS)")
    print("=" * 115)
    print(f"{'Objeto':<16} | {'Rank':<5} | {'Overlap (Med +- Std)':<22} | {'Overlap P5-P95':<18} | {'Pior Overlap':<14} | {'Dist Gr (Graus)':<16} | {'Ang Med':<10}")
    print("-" * 115)
    for k in target_keys:
        for r in ranks:
            st = summary[k][str(r)]
            ov_m = st["overlap"]["mean"] * 100.0
            ov_s = st["overlap"]["std"] * 100.0
            p5 = st["overlap"]["p5"] * 100.0
            p95 = st["overlap"]["p95"] * 100.0
            wst_ov = st["overlap"]["worst_min"] * 100.0
            d_gr = st["d_gr_deg"]["mean"]
            a_med = st["mean_angle_deg"]["mean"]
            print(f"{k:<16} | {r:<5} | {ov_m:5.2f}% +- {ov_s:4.2f}%       | [{p5:5.1f}%, {p95:5.1f}%]   | {wst_ov:5.2f}%       | {d_gr:6.2f}°          | {a_med:5.2f}°")
        print("-" * 115)

    # Salvar JSON
    out_json = os.path.join(repo_root, "experiments", "common_specific_bootstrap.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n_boot": n_boot,
            "ranks": ranks,
            "summary": summary
        }, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {out_json}")

    # [Passo 3/3] Gerar Grafico common_specific_bootstrap.png
    print("\n[Passo 3/3] Gerando grafico common_specific_bootstrap.png...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Grafico 1: Overlap vs Rank com barras de erro (P5 a P95)
    ax1 = axes[0]
    colors = {"WikiText": "tab:blue", "HF_Code": "tab:orange", "GSM8K": "tab:green", "Common_Core_Uc": "tab:red"}
    markers = {"WikiText": "o", "HF_Code": "s", "GSM8K": "^", "Common_Core_Uc": "D"}

    for k in target_keys:
        means = [summary[k][str(r)]["overlap"]["mean"] * 100.0 for r in ranks]
        p5s = [summary[k][str(r)]["overlap"]["p5"] * 100.0 for r in ranks]
        p95s = [summary[k][str(r)]["overlap"]["p95"] * 100.0 for r in ranks]
        yerr_lower = [means[i] - p5s[i] for i in range(len(ranks))]
        yerr_upper = [p95s[i] - means[i] for i in range(len(ranks))]
        
        lbl = "Núcleo Comum Uc (Média)" if k == "Common_Core_Uc" else k
        lw = 2.5 if k == "Common_Core_Uc" else 1.8
        ax1.errorbar(ranks, means, yerr=[yerr_lower, yerr_upper], label=lbl, color=colors[k], marker=markers[k], linewidth=lw, capsize=4)

    ax1.set_title("Estabilidade por Bootstrap: Overlap no Grassmanniano (50 Splits)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rank do Subespaço (r)", fontsize=11)
    ax1.set_ylabel("Overlap Médio (%) [Intervalo P5-P95]", fontsize=11)
    ax1.set_ylim(45, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    # Grafico 2: Distancia Geodesica d_Gr vs Rank
    ax2 = axes[1]
    for k in target_keys:
        d_means = [summary[k][str(r)]["d_gr_deg"]["mean"] for r in ranks]
        d_p95s = [summary[k][str(r)]["d_gr_deg"]["p95"] for r in ranks]
        lbl = "Núcleo Comum Uc" if k == "Common_Core_Uc" else k
        lw = 2.5 if k == "Common_Core_Uc" else 1.8
        ax2.plot(ranks, d_means, label=lbl, color=colors[k], marker=markers[k], linewidth=lw)

    ax2.set_title("Distância Geodésica no Grassmanniano d_Gr (Graus)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Rank do Subespaço (r)", fontsize=11)
    ax2.set_ylabel("Distância Geodésica d_Gr (Graus)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_out = os.path.join(repo_root, "common_specific_bootstrap.png")
    plt.savefig(plot_out, dpi=300)
    plt.close()
    print(f"[OK] Grafico salvo em: {plot_out}")

if __name__ == "__main__":
    main()
