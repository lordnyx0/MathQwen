# -*- coding: utf-8 -*-
"""
Common + Specific Error Manifold (Ce = Ccommon + Cspecific) em L63
Decomposicao em dois niveis do erro funcional do Atlas:
E^(e) ~ E_common + E_specific^(e)
Compara tres metodos de extracao de U_c (Media, Alinhamento Grassmanniano e Minimax),
avalia o residuo R_e = (I - U_c U_c^T) E_e, e constroi o grid (r_c, r_s).
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

def solve_minimax_subspace(sigmas_norm: list, r: int, U_init: torch.Tensor, max_iters: int = 50, lr: float = 0.05):
    """
    Subgradiente projetado no Grassmanniano com retracao QR:
    max_{U^T U = I_r} min_e Tr(U^T Sigma_e_norm U)
    """
    U = U_init.clone()
    for step in range(max_iters):
        f_vals = [torch.trace(torch.matmul(U.t(), torch.matmul(S, U))).item() for S in sigmas_norm]
        min_idx = int(np.argmin(f_vals))
        S_worst = sigmas_norm[min_idx]
        grad = 2.0 * torch.matmul(S_worst, U)
        eta = lr * (0.95 ** (step // 10))
        U_new, _ = torch.linalg.qr(U + eta * grad)
        U = U_new
    return U

def compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor):
    """
    Calcula os angulos canonicos theta_i = arccos(sigma_i(U1^T U2)) em graus.
    """
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    angles_rad = torch.acos(S)
    angles_deg = angles_rad * (180.0 / math.pi)
    return S.cpu().numpy(), angles_deg.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("COMMON + SPECIFIC ERROR MANIFOLD - ANALISE ESPECTRAL E DECOMPOSICAO EM L63")
    print("=" * 100)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Erros Funcionais E^(e)
    err_path = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    if not os.path.exists(err_path):
        raise FileNotFoundError(f"Arquivo {err_path} nao encontrado!")

    raw_errors = torch.load(err_path, map_location=device)
    domains = ["WikiText", "HF_Code", "GSM8K"]
    E_dict = {d: raw_errors[d].to(device).float() for d in domains}
    N_dict = {d: E_dict[d].shape[0] for d in domains}
    d_model = 5120

    print("\n[Passo 1/6] Operadores de Erro Carregados:")
    traces = {}
    sigmas = {}
    sigmas_norm = []
    norm_orig_sq = {}

    for d in domains:
        E = E_dict[d]
        norm_sq = (E ** 2).sum().item()
        norm_orig_sq[d] = norm_sq
        Sigma = torch.matmul(E.t(), E) / float(N_dict[d])
        tr = torch.trace(Sigma).item()
        traces[d] = tr
        sigmas[d] = Sigma
        sigmas_norm.append(Sigma / tr)
        print(f"  -> {d:<10}: {N_dict[d]} estados | ||E||_F^2 = {norm_sq:.2f} | Tr(Sigma) = {tr:.2f}")

    # 2. Decomposicoes Espectrais Individuais U_e
    print("\n[Passo 2/6] Decompondo espectros individuais dos ambientes...")
    eigen_dict = {}
    for d in domains:
        eigvals, eigvecs = torch.linalg.eigh(sigmas[d])
        # Ordem decrescente
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigen_dict[d] = {"vals": eigvals, "vecs": eigvecs}
        print(f"  -> {d:<10}: Top 5 autovalores: {[round(x, 1) for x in eigvals[:5].cpu().tolist()]}")

    # 3. Metodos de Extracao do Nucleo Comum U_c
    print("\n[Passo 3/6] Comparando Metodos de Extracao do Nucleo Comum U_c (r_c in [4, 8, 16, 32, 64])...")
    rc_list = [4, 8, 16, 32, 64]
    
    # Metodo A: Covariancia Media
    Sigma_avg = torch.stack(list(sigmas.values()), dim=0).mean(dim=0)
    eigvals_avg, eigvecs_avg = torch.linalg.eigh(Sigma_avg)
    idx_avg = torch.argsort(eigvals_avg, descending=True)
    eigvals_avg = eigvals_avg[idx_avg]
    eigvecs_avg = eigvecs_avg[:, idx_avg]

    # Metodo B: Alinhamento no Grassmanniano (Grassmannian Average)
    # Metodo C: Minimax no Stiefel
    
    uc_methods = {"PCA_Avg": {}, "Grassmann_Align": {}, "Minimax": {}}
    method_comparison = []

    for r_c in rc_list:
        # A: PCA Avg
        U_a = eigvecs_avg[:, :r_c]
        uc_methods["PCA_Avg"][r_c] = U_a

        # B: Grassmannian Alignment
        P_sum = torch.zeros(d_model, d_model, device=device)
        for d in domains:
            U_e = eigen_dict[d]["vecs"][:, :r_c]
            P_sum += torch.matmul(U_e, U_e.t())
        eVals_m, eVecs_m = torch.linalg.eigh(P_sum)
        U_b = eVecs_m[:, torch.argsort(eVals_m, descending=True)[:r_c]]
        uc_methods["Grassmann_Align"][r_c] = U_b

        # C: Minimax
        U_c_opt = solve_minimax_subspace(sigmas_norm, r_c, U_a, max_iters=50, lr=0.05)
        uc_methods["Minimax"][r_c] = U_c_opt

        # Avaliar retencao de cada metodo
        for m_name, U_cand in [("PCA_Avg", U_a), ("Grassmann_Align", U_b), ("Minimax", U_c_opt)]:
            ret_e = {}
            for d in domains:
                proj_norm_sq = (torch.matmul(E_dict[d], U_cand) ** 2).sum().item()
                ret_e[d] = (proj_norm_sq / norm_orig_sq[d]) * 100.0
            worst_r = min(ret_e.values())
            mean_r = float(np.mean(list(ret_e.values())))
            method_comparison.append({
                "rank": r_c,
                "method": m_name,
                "domain_retentions": ret_e,
                "worst_retention": worst_r,
                "mean_retention": mean_r
            })
            print(f"  r_c={r_c:<2} | Metodo: {m_name:<16} | Media: {mean_r:5.2f}% | Pior: {worst_r:5.2f}% | Detalhes: { {k: round(v, 2) for k, v in ret_e.items()} }")

    # Selecionar Minimax como base do nucleo comum U_c
    U_c_selected = uc_methods["Minimax"]

    # 4. Decomposicao do Residuo Especifico R_e = (I - U_c U_c^T) E_e
    print("\n[Passo 4/6] Decompondo Residuos Especificos R_e e Construindo Grid (r_c, r_s)...")
    rs_list = [4, 8, 16, 32, 64, 128]
    grid_results = []
    bases_to_save = {"U_c": {}, "V_specific": {}}

    for r_c in rc_list:
        U_c = U_c_selected[r_c]
        bases_to_save["U_c"][r_c] = U_c.cpu()
        bases_to_save["V_specific"][r_c] = {}

        # Decompor residuo para cada dominio
        R_dict = {}
        V_dict = {}
        spec_eigvals_dict = {}

        for d in domains:
            E = E_dict[d]
            proj_c = torch.matmul(torch.matmul(E, U_c), U_c.t())
            R = E - proj_c
            R_dict[d] = R
            
            # SVD do residuo / eigendecomposicao de R^T R
            Sigma_R = torch.matmul(R.t(), R) / float(N_dict[d])
            eVals_R, eVecs_R = torch.linalg.eigh(Sigma_R)
            idx_R = torch.argsort(eVals_R, descending=True)
            eVals_R = eVals_R[idx_R]
            eVecs_R = eVecs_R[:, idx_R]
            V_dict[d] = eVecs_R
            spec_eigvals_dict[d] = eVals_R.cpu().numpy()
            bases_to_save["V_specific"][r_c][d] = eVecs_R[:, :128].cpu()

        # Grid (r_c, r_s)
        for r_s in rs_list:
            row_rec = {
                "r_c": r_c,
                "r_s": r_s,
                "total_rank": r_c + r_s,
                "common_retention": {},
                "specific_retention": {},
                "total_retention": {},
                "unexplained_error": {}
            }
            for d in domains:
                E = E_dict[d]
                R = R_dict[d]
                V_s = V_dict[d][:, :r_s]
                
                # Componente comum: ||E U_c U_c^T||^2 / ||E||^2
                c_norm_sq = (torch.matmul(E, U_c) ** 2).sum().item()
                # Componente especifica: ||R V_s V_s^T||^2 / ||E||^2
                s_norm_sq = (torch.matmul(R, V_s) ** 2).sum().item()
                
                tot_norm_sq = c_norm_sq + s_norm_sq
                tot_orig = norm_orig_sq[d]

                c_pct = (c_norm_sq / tot_orig) * 100.0
                s_pct = (s_norm_sq / tot_orig) * 100.0
                tot_pct = (tot_norm_sq / tot_orig) * 100.0
                unexp_pct = max(0.0, 100.0 - tot_pct)

                row_rec["common_retention"][d] = c_pct
                row_rec["specific_retention"][d] = s_pct
                row_rec["total_retention"][d] = tot_pct
                row_rec["unexplained_error"][d] = unexp_pct

            row_rec["worst_unexplained"] = max(row_rec["unexplained_error"].values())
            row_rec["mean_unexplained"] = float(np.mean(list(row_rec["unexplained_error"].values())))
            row_rec["worst_total_retention"] = min(row_rec["total_retention"].values())
            row_rec["mean_total_retention"] = float(np.mean(list(row_rec["total_retention"].values())))
            grid_results.append(row_rec)

    # Imprimir Tabela Canonica do Grid (r_c, r_s)
    print("\n" + "=" * 110)
    print(f"{'r_c':<5} | {'r_s':<5} | {'r_tot':<6} | {'WikiText Unexp':<15} | {'Code Unexp':<12} | {'GSM8K Unexp':<13} | {'Pior Caso Unexp':<16} | {'Media Retencao':<15}")
    print("-" * 110)
    for row in grid_results:
        if row["r_c"] in [8, 16, 32, 64] and row["r_s"] in [8, 16, 32, 64]:
            w_u = row["unexplained_error"]["WikiText"]
            c_u = row["unexplained_error"]["HF_Code"]
            g_u = row["unexplained_error"]["GSM8K"]
            wst = row["worst_unexplained"]
            m_r = row["mean_total_retention"]
            print(f"{row['r_c']:<5} | {row['r_s']:<5} | {row['total_rank']:<6} | {w_u:6.2f}%         | {c_u:6.2f}%      | {g_u:6.2f}%       | {wst:6.2f}%          | {m_r:6.2f}%")
    print("=" * 110)

    # 5. Projecao Cruzada de Duas Camadas e Alinhamento no Grassmanniano
    print("\n[Passo 5/6] Avaliando Projecao Cruzada de Duas Camadas e Angulos Principais...")
    cross_proj_records = {}
    alignment_records = {}

    for r_c in [16, 32, 64]:
        U_c = U_c_selected[r_c]
        alignment_records[str(r_c)] = {}
        for d in domains:
            U_e = eigen_dict[d]["vecs"][:, :r_c]
            s_vals, a_deg = compute_principal_angles(U_e, U_c)
            alignment_records[str(r_c)][d] = {
                "singular_values": s_vals.tolist(),
                "angles_deg": a_deg.tolist(),
                "mean_angle_deg": float(np.mean(a_deg)),
                "worst_angle_deg": float(np.max(a_deg)),
                "best_angle_deg": float(np.min(a_deg)),
                "overlap": float(np.mean(s_vals))
            }

        cross_proj_records[str(r_c)] = {"gamma_c": {}, "cross_transfer_matrix_rs16": []}
        for d in domains:
            c_ret = (torch.matmul(E_dict[d], U_c) ** 2).sum().item() / norm_orig_sq[d] * 100.0
            cross_proj_records[str(r_c)]["gamma_c"][d] = c_ret

        # Matriz de transferencia de V_e(16)
        r_s_eval = 16
        trans_mat = np.zeros((3, 3))
        for i, d_src in enumerate(domains):
            V_src = bases_to_save["V_specific"][r_c][d_src][:, :r_s_eval].to(device)
            for j, d_tgt in enumerate(domains):
                E_tgt = E_dict[d_tgt]
                R_tgt = E_tgt - torch.matmul(torch.matmul(E_tgt, U_c), U_c.t())
                norm_R_sq = (R_tgt ** 2).sum().item()
                if norm_R_sq > 0:
                    proj_R = (torch.matmul(R_tgt, V_src) ** 2).sum().item()
                    trans_mat[i, j] = (proj_R / norm_R_sq) * 100.0
                else:
                    trans_mat[i, j] = 0.0

        cross_proj_records[str(r_c)]["cross_transfer_matrix_rs16"] = trans_mat.tolist()
        print(f"\nr_c={r_c} | Matriz de Transferencia do Complemento Especifico V_e (r_s=16) sobre os Residuos R_tgt (%):")
        print(f"{'Src \\ Tgt':<12} | {'WikiText':<10} | {'HF_Code':<10} | {'GSM8K':<10}")
        for i, d_src in enumerate(domains):
            print(f"{d_src:<12} | {trans_mat[i,0]:6.2f}%    | {trans_mat[i,1]:6.2f}%    | {trans_mat[i,2]:6.2f}%")

    # 6. Salvar Artefatos e Gerar Graficos
    print("\n[Passo 6/6] Salvando checkpoints e gerando graficos publicaveis...")

    # Salvar checkpoints das bases
    ckpt_out = os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt")
    torch.save(bases_to_save, ckpt_out)
    print(f"  -> Checkpoint salvo em: {ckpt_out}")

    # Salvar JSONs
    res_json = os.path.join(repo_root, "experiments", "common_specific_results.json")
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump({
            "domains": domains,
            "traces": traces,
            "rc_list": rc_list,
            "rs_list": rs_list,
            "method_comparison": method_comparison,
            "grid_results": grid_results
        }, f, indent=2)
    print(f"  -> Resultados salvos em: {res_json}")

    cross_json = os.path.join(repo_root, "experiments", "common_specific_cross_projection.json")
    with open(cross_json, "w", encoding="utf-8") as f:
        json.dump({
            "alignment_records": alignment_records,
            "cross_proj_records": cross_proj_records
        }, f, indent=2)
    print(f"  -> Projecao cruzada salva em: {cross_json}")

    # ------------------ GERACAO DE GRAFICOS ------------------
    # 1. common_vs_specific_spectrum.png
    plt.figure(figsize=(10, 6))
    for d in domains:
        ev_orig = eigen_dict[d]["vals"][:64].cpu().numpy()
        plt.plot(range(1, 65), ev_orig, label=f"Original {d}", linewidth=2)
    plt.plot(range(1, 65), eigvals_avg[:64].cpu().numpy(), "k--", label="Espectro Medio (Sigma_avg)", linewidth=2.5)
    plt.yscale("log")
    plt.title("Espectro Singular: Erro Total vs Autovalores Principais", fontsize=13, fontweight="bold")
    plt.xlabel("Modo / Indice Singular", fontsize=11)
    plt.ylabel("Autovalor (Escala Log)", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, "common_vs_specific_spectrum.png"), dpi=300)
    plt.close()

    # 2. specific_residual_spectrum.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, r_c_choice in enumerate([16, 32]):
        ax = axes[ax_idx]
        U_c = U_c_selected[r_c_choice]
        for d in domains:
            E = E_dict[d]
            R = E - torch.matmul(torch.matmul(E, U_c), U_c.t())
            Sigma_R = torch.matmul(R.t(), R) / float(N_dict[d])
            ev_R, _ = torch.linalg.eigh(Sigma_R)
            ev_R = torch.sort(ev_R, descending=True)[0][:64].cpu().numpy()
            ax.plot(range(1, 65), ev_R, label=f"Residuo {d}", linewidth=2)
        ax.set_yscale("log")
        ax.set_title(f"Espectro Residual R_e apos remover U_c (r_c={r_c_choice})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Modo Residual", fontsize=10)
        ax.set_ylabel("Autovalor Residual (Log)", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, "specific_residual_spectrum.png"), dpi=300)
    plt.close()

    # 3. common_specific_retention.png
    rc_axis = [4, 8, 16, 32, 64]
    rs_axis = [4, 8, 16, 32, 64, 128]
    heat_matrix = np.zeros((len(rc_axis), len(rs_axis)))
    for r_idx, r_c in enumerate(rc_axis):
        for s_idx, r_s in enumerate(rs_axis):
            rec = next(r for r in grid_results if r["r_c"] == r_c and r["r_s"] == r_s)
            heat_matrix[r_idx, s_idx] = rec["worst_unexplained"]

    plt.figure(figsize=(9, 7))
    im = plt.imshow(heat_matrix, cmap="YlGnBu_r", aspect="auto")
    plt.colorbar(im, label="Erro Nao Explicado Pior Caso (%)")
    plt.xticks(range(len(rs_axis)), rs_axis)
    plt.yticks(range(len(rc_axis)), rc_axis)
    plt.xlabel("Rank Especifico (r_s)", fontsize=12, fontweight="bold")
    plt.ylabel("Rank Comum (r_c)", fontsize=12, fontweight="bold")
    plt.title("Grid de Erro Nao Explicado: (r_c, r_s) -> Pior Caso (%)", fontsize=13, fontweight="bold")
    for i in range(len(rc_axis)):
        for j in range(len(rs_axis)):
            val = heat_matrix[i, j]
            color = "white" if val < 10.0 else "black"
            plt.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, "common_specific_retention.png"), dpi=300)
    plt.close()

    # 4. common_specific_pareto.png
    plt.figure(figsize=(10, 6))
    ranks_tot = [r["total_rank"] for r in grid_results]
    ret_tot = [r["worst_total_retention"] for r in grid_results]
    rcs = [r["r_c"] for r in grid_results]
    
    scatter = plt.scatter(ranks_tot, ret_tot, c=rcs, cmap="viridis", s=70, edgecolors="k", zorder=3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Rank Comum (r_c)", fontsize=11)

    sorted_pts = sorted(zip(ranks_tot, ret_tot, grid_results), key=lambda x: (x[0], -x[1]))
    pareto_pts = []
    curr_max_ret = -1.0
    for r_tot, ret, rec in sorted_pts:
        if ret > curr_max_ret:
            pareto_pts.append((r_tot, ret, rec))
            curr_max_ret = ret

    px = [p[0] for p in pareto_pts]
    py = [p[1] for p in pareto_pts]
    plt.plot(px, py, "r--", linewidth=2.5, label="Fronteira de Pareto (r_c + r_s)", zorder=2)
    for p in pareto_pts:
        rec = p[2]
        if p[0] in [8, 16, 24, 32, 48, 64, 80, 96, 128]:
            plt.annotate(f"({rec['r_c']}+{rec['r_s']})", (p[0], p[1]), textcoords="offset points", xytext=(0,8), ha="center", fontsize=8, fontweight="bold")

    plt.title("Fronteira de Pareto: Custo Total de Parametros (r_c + r_s) vs Retencao Pior Caso", fontsize=13, fontweight="bold")
    plt.xlabel("Rank Total (r_c + r_s)", fontsize=11)
    plt.ylabel("Retencao no Pior Dominio (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, "common_specific_pareto.png"), dpi=300)
    plt.close()

    # 5. common_subspace_alignment.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, r_c_choice in enumerate([16, 32]):
        ax = axes[ax_idx]
        for d in domains:
            rec = alignment_records[str(r_c_choice)][d]
            ax.plot(range(1, r_c_choice + 1), rec["angles_deg"], marker="o", label=f"{d} (Med={rec['mean_angle_deg']:.1f}°)", linewidth=2)
        ax.set_title(f"Angulos Canonicos U_e vs U_c (r_c={r_c_choice})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Indice do Modo Comum", fontsize=10)
        ax.set_ylabel("Angulo Canonico (Graus)", fontsize=10)
        ax.set_ylim(0, 90)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, "common_subspace_alignment.png"), dpi=300)
    plt.close()

    # 6. common_specific_cross_projection.png
    t_mat = np.array(cross_proj_records["32"]["cross_transfer_matrix_rs16"])
    plt.figure(figsize=(8, 6))
    im = plt.imshow(t_mat, cmap="magma", vmin=0, vmax=100)
    plt.colorbar(im, label="Retencao Residual Explicada (%)")
    plt.xticks(range(3), domains)
    plt.yticks(range(3), domains)
    plt.xlabel("Dominio de Destino (Alvo do Residuo R_tgt)", fontsize=11, fontweight="bold")
    plt.ylabel("Dominio de Origem (Base Especifica V_src)", fontsize=11, fontweight="bold")
    plt.title("Transferencia Cruzada do Complemento Especifico V_e (r_s=16, r_c=32)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            val = t_mat[i, j]
            color = "white" if val < 50.0 else "black"
            plt.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, "common_specific_cross_projection.png"), dpi=300)
    plt.close()

    print("\n[OK] Fase 1 concluida com sucesso! Todos os dados e graficos foram gerados.")

if __name__ == "__main__":
    main()
