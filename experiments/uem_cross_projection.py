# -*- coding: utf-8 -*-
"""
Universal Error Manifold (UEM) - Projecao Cruzada, Angulos Principais e Estabilidade Bootstrap
Calcula a matriz Gamma(r) = [||E^(i) U_j U_j^T||_F^2 / ||E^(i)||_F^2],
angulos canonicos e sobreposicao no Grassmanniano entre dominios e subsets de bootstrap.
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

def compute_principal_angles(U_i: torch.Tensor, U_j: torch.Tensor):
    """
    Computa os angulos principais entre dois subespacos ortonormais U_i, U_j em R^{d x r}.
    Retorna: valores singulares sigma_a, angulos theta_a (graus), media, pior angulo e overlap.
    """
    M = torch.matmul(U_i.t(), U_j) # (r, r)
    # SVD de M
    U_m, S_m, V_m = torch.linalg.svd(M)
    sigmas = torch.clamp(S_m, 0.0, 1.0)
    thetas_rad = torch.acos(sigmas)
    thetas_deg = thetas_rad * (180.0 / math.pi)

    r = U_i.shape[1]
    overlap = (sigmas ** 2).sum().item() / float(r)
    mean_angle = thetas_deg.mean().item()
    worst_angle = thetas_deg.max().item()

    return {
        "singular_values": [float(x) for x in sigmas.cpu().numpy()],
        "angles_deg": [float(x) for x in thetas_deg.cpu().numpy()],
        "mean_angle_deg": mean_angle,
        "worst_angle_deg": worst_angle,
        "subspace_overlap": overlap
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("UNIVERSAL ERROR MANIFOLD (UEM) - PROJECAO CRUZADA E ESTABILIDADE DE SUBESPACOS")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar tensores de erro
    err_cache_file = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    if not os.path.exists(err_cache_file):
        print(f"ERRO: Cache de erros {err_cache_file} nao encontrado. Execute experiments/uem_analysis.py primeiro.")
        sys.exit(1)

    print(f"\n[Passo 1/4] Carregando tensores de erro de: {err_cache_file}...")
    errors_raw = torch.load(err_cache_file, map_location=device)
    errors = {k: v.float() for k, v in errors_raw.items()}
    domain_names = list(errors.keys())
    K = len(domain_names)
    print(f"  -> Dominios carregados ({K}): {domain_names}")

    # Calcular bases espectrais para cada dominio
    domain_bases = {}
    for d_name in domain_names:
        E_k = errors[d_name]
        N_k = E_k.shape[0]
        Sigma_k = torch.matmul(E_k.t(), E_k) / float(N_k)
        evals, evecs = torch.linalg.eigh(Sigma_k)
        domain_bases[d_name] = torch.flip(evecs, dims=[1]) # ordenado descrescente

    ranks = [8, 16, 32, 64, 128, 256]

    # 2. Matriz de Projecao Cruzada Gamma(r) = [||E^(i) U_j U_j^T||_F^2 / ||E^(i)||_F^2]
    print("\n[Passo 2/4] Calculando Matrizes de Projecao Cruzada Gamma(r)...")
    gamma_matrices = {}

    for r in ranks:
        gamma_r = np.zeros((K, K), dtype=np.float32)
        for i, d_i in enumerate(domain_names):
            E_i = errors[d_i]
            norm_E_i_sq = (E_i ** 2).sum().item()

            for j, d_j in enumerate(domain_names):
                U_j_r = domain_bases[d_j][:, :r]
                # E_i @ U_j @ U_j^T
                proj = torch.matmul(torch.matmul(E_i, U_j_r), U_j_r.t())
                norm_proj_sq = (proj ** 2).sum().item()
                gamma_r[i, j] = (norm_proj_sq / norm_E_i_sq) * 100.0

        gamma_matrices[str(r)] = gamma_r.tolist()
        print(f"\n--- Matriz Gamma(r={r}) (%) ---")
        print(f"{'Origem / Base':<16} | " + " | ".join([f"{dn:<12}" for dn in domain_names]))
        print("-" * (18 + 15 * K))
        for i, d_i in enumerate(domain_names):
            row_str = " | ".join([f"{gamma_r[i, j]:<11.2f}%" for j in range(K)])
            print(f"{d_i:<16} | {row_str}")

    # 3. Angulos Principais entre pares de dominios
    print("\n[Passo 3/4] Analisando Angulos Principais e Sobreposicao no Grassmanniano...")
    angles_analysis = {}

    for r in ranks:
        angles_analysis[str(r)] = {}
        for i in range(K):
            for j in range(i + 1, K):
                d_i, d_j = domain_names[i], domain_names[j]
                pair_name = f"{d_i}_vs_{d_j}"
                U_i_r = domain_bases[d_i][:, :r]
                U_j_r = domain_bases[d_j][:, :r]
                res = compute_principal_angles(U_i_r, U_j_r)
                angles_analysis[str(r)][pair_name] = res

        print(f"\nPosto r={r:03d}:")
        for pair_name, res in angles_analysis[str(r)].items():
            print(f"  {pair_name:<25} | Overlap: {res['subspace_overlap']*100:.2f}% | Angulo Medio: {res['mean_angle_deg']:.2f} deg | Pior Angulo: {res['worst_angle_deg']:.2f} deg")

    # 4. Bootstrap de Estabilidade Interna
    print("\n[Passo 4/4] Executando Bootstrap de Estabilidade Interna...")
    bootstrap_results = {}

    for d_name in domain_names:
        E_full = errors[d_name]
        N = E_full.shape[0]
        # Embaralhar e dividir em 3 fatias
        torch.manual_seed(42)
        perm = torch.randperm(N)
        s_sz = N // 3
        slice_A = E_full[perm[:s_sz]]
        slice_B = E_full[perm[s_sz:2*s_sz]]
        slice_C = E_full[perm[2*s_sz:3*s_sz]]

        # Bases de cada fatia
        def get_slice_basis(E_sub):
            Sig = torch.matmul(E_sub.t(), E_sub) / float(E_sub.shape[0])
            _, ev = torch.linalg.eigh(Sig)
            return torch.flip(ev, dims=[1])

        U_A = get_slice_basis(slice_A)
        U_B = get_slice_basis(slice_B)
        U_C = get_slice_basis(slice_C)

        b_domain = {}
        for r in ranks:
            ab = compute_principal_angles(U_A[:, :r], U_B[:, :r])
            ac = compute_principal_angles(U_A[:, :r], U_C[:, :r])
            bc = compute_principal_angles(U_B[:, :r], U_C[:, :r])
            overlaps = [ab["subspace_overlap"], ac["subspace_overlap"], bc["subspace_overlap"]]
            mean_deg = [ab["mean_angle_deg"], ac["mean_angle_deg"], bc["mean_angle_deg"]]
            b_domain[str(r)] = {
                "overlap_mean": float(np.mean(overlaps)),
                "overlap_std": float(np.std(overlaps)),
                "mean_angle_deg": float(np.mean(mean_deg)),
                "worst_angle_deg": float(max([ab["worst_angle_deg"], ac["worst_angle_deg"], bc["worst_angle_deg"]]))
            }
        bootstrap_results[d_name] = b_domain

    print("\n--- Estabilidade por Bootstrap (Media de Overlap interno entre fatias independentes) ---")
    print(f"{'Dominio':<16} | " + " | ".join([f"r={r:<4}" for r in ranks]))
    print("-" * 65)
    for d_name in domain_names:
        row_str = " | ".join([f"{bootstrap_results[d_name][str(r)]['overlap_mean']*100:<5.1f}%" for r in ranks])
        print(f"{d_name:<16} | {row_str}")

    # Salvar resultados em JSON
    out_json = os.path.join(repo_root, "experiments", "uem_cross_projection.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "domain_names": domain_names,
            "ranks": ranks,
            "gamma_matrices": gamma_matrices,
            "angles_analysis": angles_analysis,
            "bootstrap_results": bootstrap_results
        }, f, indent=2)
    print(f"\n[OK] Resultados de projecao cruzada salvos em: {out_json}")

    # 5. Geracao dos Graficos (PNG)
    print("\nGerando graficos de projecao cruzada e estabilidade...")

    # Plot 1: Heatmaps da Matriz Gamma para r=32, 64, 128, 256
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    plot_ranks = [32, 64, 128, 256]
    for ax, r in zip(axes.flatten(), plot_ranks):
        mat = np.array(gamma_matrices[str(r)])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(f"Matriz de Projeção Cruzada $\\Gamma(r={r})$", fontsize=11)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(domain_names, fontsize=9)
        ax.set_yticklabels(domain_names, fontsize=9)
        ax.set_xlabel("Base Projetora $U_j$", fontsize=10)
        ax.set_ylabel("Erro Alvo $E^{(i)}$", fontsize=10)
        for ii in range(K):
            for jj in range(K):
                text_col = "white" if mat[ii, jj] > 50 else "black"
                ax.text(jj, ii, f"{mat[ii, jj]:.1f}%", ha="center", va="center", color=text_col, fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Energia Capturada (%)")
    p_gamma = os.path.join(repo_root, "uem_cross_projection.png")
    plt.savefig(p_gamma)
    plt.close()
    print(f"  -> Salvo: {p_gamma}")

    # Plot 2: Distribuicao de Angulos Principais para r=64
    plt.figure(figsize=(10, 6), dpi=150)
    for pair_name, res in angles_analysis["64"].items():
        plt.plot(range(1, 65), res["angles_deg"], lw=2.2, label=pair_name)
    plt.axhline(45, color="red", ls=":", alpha=0.7, label="Limite de Ortogonalidade Efetiva (45°)")
    plt.title("Universal Error Manifold (UEM): Ângulos Principais no Posto $r=64$", fontsize=13, pad=12)
    plt.xlabel("Índice do Modo Canônico ($a=1 \\dots r$)", fontsize=11)
    plt.ylabel("Ângulo Principal $\\theta_a$ (graus)", fontsize=11)
    plt.ylim(0, 90)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p_angles = os.path.join(repo_root, "uem_principal_angles.png")
    plt.savefig(p_angles)
    plt.close()
    print(f"  -> Salvo: {p_angles}")

    # Plot 3: Bootstrap Stability Overlap vs Rank
    plt.figure(figsize=(10, 6), dpi=150)
    for d_name in domain_names:
        means = [bootstrap_results[d_name][str(r)]["overlap_mean"] * 100 for r in ranks]
        stds = [bootstrap_results[d_name][str(r)]["overlap_std"] * 100 for r in ranks]
        plt.errorbar(ranks, means, yerr=stds, marker="o", lw=2.2, capsize=4, label=f"{d_name} (Resampling)")
    plt.title("Universal Error Manifold (UEM): Estabilidade Interna por Bootstrap", fontsize=13, pad=12)
    plt.xlabel("Posto do Refinador ($r$)", fontsize=11)
    plt.ylabel("Sobreposição no Grassmanniano (%)", fontsize=11)
    plt.xticks(ranks)
    plt.ylim(50, 102)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p_boot = os.path.join(repo_root, "uem_bootstrap_stability.png")
    plt.savefig(p_boot)
    plt.close()
    print(f"  -> Salvo: {p_boot}")

    print("\n[OK] Analise UEM 2/3 (Projecao Cruzada e Estabilidade) concluida!")

if __name__ == "__main__":
    main()
