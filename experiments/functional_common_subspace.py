# -*- coding: utf-8 -*-
"""
EXPERIMENTO A, B e F: DESCOBERTA DO NUCLEO COMPARTILHADO FUNCIONAL
- Calcula G_c^e = U_c^T G_J^e U_c em Wikitext, HF Code e GSM8K.
- Decompõe G_c^avg = Q Lambda Q^T e extrai U_f = U_c Q.
- Resolve Minimax funcional no Stiefel e compara com média.
- Mede o espectro, concentração cumulativa e estabilidade entre domínios.
- Experimento F: Avalia a razão sensibilidade/energia s_i / (e_i + eps) para cada vetor.
- Salva checkpoints Uc_functional_r{4,8,16,24,32}.pt e gráfico functional_common_spectrum.png.
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

from reference.loader import load_qwen_reference_components

def compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor):
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    angles_deg = torch.acos(S) * (180.0 / math.pi)
    return S.cpu().numpy(), angles_deg.cpu().numpy()

def solve_minimax_rotation(G_list: list, r: int, Q_init: torch.Tensor, max_iters: int = 60, lr: float = 0.05):
    """
    Subgradiente projetado no grupo ortogonal:
    max_{Q^T Q = I_r} min_e Tr(Q^T G_e Q) / Tr(G_e)
    """
    Q = Q_init[:, :r].clone()
    traces = [torch.trace(G).item() for G in G_list]
    G_norm = [G / tr for G, tr in zip(G_list, traces)]

    for step in range(max_iters):
        f_vals = [torch.trace(torch.matmul(Q.t(), torch.matmul(G, Q))).item() for G in G_norm]
        min_idx = int(np.argmin(f_vals))
        G_worst = G_norm[min_idx]
        grad = 2.0 * torch.matmul(G_worst, Q)
        eta = lr * (0.95 ** (step // 10))
        Q_new, _ = torch.linalg.qr(Q + eta * grad)
        Q = Q_new
    return Q

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print("CICLO DE ENGENHARIA: DESCOBERTA DO NUCLEO COMPARTILHADO FUNCIONAL (EXPERIMENTOS A, B, F)")
    print("=" * 110)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Componentes da Cabeça e U_c(32)
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=16)
    lm_head_w = comp["lm_head_w"].float() # (151936, 5120)
    norm_weight = comp["norm_mod"].weight.data.float() # (5120,)
    d_model = 5120

    bases_path = os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt")
    if not os.path.exists(bases_path):
        raise FileNotFoundError(f"Arquivo {bases_path} nao encontrado!")
    saved_bases = torch.load(bases_path, map_location=device)
    U_c_32 = saved_bases["U_c"][32].float().to(device) # (5120, 32)
    print(f"  -> U_c(32) carregado: shape = {U_c_32.shape} | Orthonorm check = {torch.dist(torch.matmul(U_c_32.t(), U_c_32), torch.eye(32, device=device)).item():.2e}")

    # 2. Computar G_head = W_head^T W_head
    print("\n[Passo 1/5] Computando Gramiano do LMHead: G_head = W_head^T W_head...")
    t0 = time.time()
    G_head = torch.matmul(lm_head_w.t(), lm_head_w)
    print(f"  -> G_head concluido em {time.time() - t0:.2f}s | Tr(G_head) = {torch.trace(G_head).item():.2e}")

    # 3. Carregar Ativações dos Três Ambientes
    print("\n[Passo 2/5] Carregando estados de ativacao em L63 para os 3 dominios...")
    domains = ["WikiText", "HF_Code", "GSM8K"]
    X_dict = {}

    # WikiText
    h_wiki = torch.load(os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt"), map_location=device).reshape(-1, d_model).float()
    X_dict["WikiText"] = h_wiki[:1024]

    # HF Code
    h_code_raw = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    X_dict["HF_Code"] = h_code_raw["h_in_63"].reshape(-1, d_model).float()[:1024]

    # GSM8K
    h_gsm_raw = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    X_dict["GSM8K"] = h_gsm_raw["h_in_63"].reshape(-1, d_model).float()[:1024]

    for d in domains:
        print(f"  -> {d:<10}: {X_dict[d].shape[0]} tokens amostrados")

    # 4. Computar Operador Jacobiano Médio G_J^e e G_c^e = U_c^T G_J^e U_c
    print("\n[Passo 3/5] Avaliando Jacobiana media e Gramianos restritos G_c^e = U_c^T G_J^e U_c...")
    eps = 1e-6
    G_J_dict = {}
    G_c_dict = {}

    for d in domains:
        X_e = X_dict[d]
        N_e = X_e.shape[0]
        G_J_e = torch.zeros(d_model, d_model, device=device)
        batch_sz = 128
        for b_start in range(0, N_e, batch_sz):
            b_end = min(b_start + batch_sz, N_e)
            H_b = X_e[b_start:b_end]
            B = H_b.shape[0]
            mu2 = (H_b ** 2).mean(dim=-1, keepdim=True) + eps
            s = torch.rsqrt(mu2)
            s_mean = s.mean().item()
            mu2_mean = mu2.mean().item()
            D_mean = s_mean * torch.diag(norm_weight) - (s_mean / (d_model * mu2_mean)) * torch.matmul(torch.diag(norm_weight), torch.matmul(H_b.t(), H_b) / float(B))
            G_J_b = torch.matmul(D_mean.t(), torch.matmul(G_head, D_mean))
            G_J_e += G_J_b * (float(B) / float(N_e))

        G_J_dict[d] = G_J_e
        G_c_e = torch.matmul(U_c_32.t(), torch.matmul(G_J_e, U_c_32)) # (32, 32)
        G_c_dict[d] = G_c_e
        tr_c = torch.trace(G_c_e).item()
        tr_tot = torch.trace(G_J_e).item()
        print(f"  -> {d:<10}: Tr(G_J) = {tr_tot:.2e} | Tr(G_c) = {tr_c:.2e} | Retencao funcional em U_c(32) = {tr_c / tr_tot * 100:.2f}%")

    # Média dos Gramianos Restritos
    G_c_avg = (G_c_dict["WikiText"] + G_c_dict["HF_Code"] + G_c_dict["GSM8K"]) / 3.0
    G_J_avg = (G_J_dict["WikiText"] + G_J_dict["HF_Code"] + G_J_dict["GSM8K"]) / 3.0

    # 5. Diagonalização Espectral de G_c_avg: G_c_avg = Q Lambda Q^T
    print("\n[Passo 4/5] Diagonalizando G_c_avg (32x32) para extrair a rotacao funcional Q...")
    L_c, Q_c = torch.linalg.eigh(G_c_avg)
    idx_desc = torch.argsort(L_c, descending=True)
    lambda_sorted = L_c[idx_desc]
    Q_sorted = Q_c[:, idx_desc] # (32, 32)

    # Base Funcional U_f = U_c Q
    U_f_32 = torch.matmul(U_c_32, Q_sorted) # (5120, 32)
    orth_err = torch.dist(torch.matmul(U_f_32.t(), U_f_32), torch.eye(32, device=device)).item()
    print(f"  -> U_f(32) gerado com sucesso! Erro de ortonormalidade: {orth_err:.2e}")

    # Concentração Espectral e Energia Cumulativa
    cum_energy = torch.cumsum(lambda_sorted, dim=0) / torch.sum(lambda_sorted)
    ranks_eval = [4, 8, 16, 24, 32]
    print("\n" + "-" * 85)
    print(f"{'Rank rf':<10} | {'Autovalor lambda_rf':<22} | {'Energia Funcional Cumulativa':<32}")
    print("-" * 85)
    cum_energy_dict = {}
    for r in ranks_eval:
        val = lambda_sorted[r-1].item()
        cum = cum_energy[r-1].item() * 100.0
        cum_energy_dict[r] = cum
        print(f"{r:<10} | {val:<22.4e} | {cum:<32.2f}%")
    print("-" * 85)

    # Minimax Subspace Solution
    print("\n  -> Computando solucao Minimax em SO(32)...")
    Q_minimax_dict = {}
    for r in ranks_eval:
        Q_init = Q_sorted[:, :r]
        Q_mm = solve_minimax_rotation([G_c_dict[d] for d in domains], r, Q_init)
        Q_minimax_dict[r] = Q_mm

    # Salvar Checkpoints de U_c_functional
    os.makedirs(os.path.join(repo_root, "checkpoints"), exist_ok=True)
    for r in ranks_eval:
        ckpt_path = os.path.join(repo_root, "checkpoints", f"Uc_functional_r{r}.pt")
        sub_basis = U_f_32[:, :r].contiguous()
        torch.save({
            "r": r,
            "U_f": sub_basis,
            "Q": Q_sorted[:, :r].contiguous(),
            "Q_minimax": Q_minimax_dict[r].contiguous(),
            "lambda": lambda_sorted[:r],
            "cum_energy": cum_energy[r-1].item()
        }, ckpt_path)
        print(f"  [OK] Salvo: checkpoints/Uc_functional_r{r}.pt (shape = {sub_basis.shape})")

    # 6. Experimento F — Localizar Vetores Críticos: Razão Sensibilidade / Energia
    print("\n[Passo 5/5] Experimento F: Avaliando razao sensibilidade funcional / energia euclidiana...")
    err_path = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    raw_errors = torch.load(err_path, map_location=device)
    Sigma_E_avg = torch.zeros(d_model, d_model, device=device)
    for d in domains:
        E = raw_errors[d].to(device).float()
        Sigma_E_avg += torch.matmul(E.t(), E) / (float(E.shape[0]) * 3.0)

    tr_Sigma_E = torch.trace(Sigma_E_avg).item()
    tr_GJ_avg = torch.trace(G_J_avg).item()

    vector_metrics = []
    eps_e = 1e-6
    for i in range(32):
        u_i = U_f_32[:, i:i+1] # (5120, 1)
        s_i = torch.matmul(u_i.t(), torch.matmul(G_J_avg, u_i)).item()
        e_i = torch.matmul(u_i.t(), torch.matmul(Sigma_E_avg, u_i)).item()
        s_norm = s_i / tr_GJ_avg
        e_norm = e_i / tr_Sigma_E
        ratio = s_norm / (e_norm + eps_e)

        vector_metrics.append({
            "vector": f"u_{i+1}",
            "index": i,
            "s_i": s_i,
            "e_i": e_i,
            "s_norm": s_norm,
            "e_norm": e_norm,
            "ratio": ratio
        })

    # Tabela do Experimento F
    print("\n" + "=" * 110)
    print("EXPERIMENTO F: IDENTIFICACAO DAS DIRECOES DE ALTA INFLUENCIA FUNCIONAL VS ENERGIA DO ERRO")
    print("=" * 110)
    print(f"{'Vetor':<8} | {'Energia Erro (e_norm)':<24} | {'Sensib Func (s_norm)':<24} | {'Razao Func/Energia':<20} | {'Status'}")
    print("-" * 110)
    sorted_by_ratio = sorted(vector_metrics, key=lambda x: x["ratio"], reverse=True)
    for vm in vector_metrics[:16]:
        stat = "ALTA ALAVANCAGEM" if vm["ratio"] > 1.2 else ("EQUILIBRADO" if vm["ratio"] >= 0.8 else "SUB-SENSIBILIDADE")
        print(f"{vm['vector']:<8} | {vm['e_norm']*100:<23.4f}% | {vm['s_norm']*100:<23.4f}% | {vm['ratio']:<20.3f} | {stat}")
    print("-" * 110)

    top_leverage = [vm["vector"] for vm in sorted_by_ratio[:5]]
    print(f"\nTop-5 Vetores de Maior Alavancagem Funcional Relativa: {', '.join(top_leverage)}")

    # 7. Salvar JSON de Resultados
    results_json_path = os.path.join(repo_root, "experiments", "functional_common_results.json")
    json_data = {
        "lambda_spectrum": [x.item() for x in lambda_sorted],
        "cum_energy": cum_energy_dict,
        "traces": {
            "Tr_GJ_avg": tr_GJ_avg,
            "Tr_Sigma_E": tr_Sigma_E,
            "domain_traces": {d: torch.trace(G_c_dict[d]).item() for d in domains}
        },
        "vector_analysis": vector_metrics,
        "sorted_by_leverage": sorted_by_ratio
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {results_json_path}")

    # 8. Plotar Gráfico Publicável: functional_common_spectrum.png
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("Descoberta do Núcleo Compartilhado Funcional (L63) - Experimentos A, B e F", fontsize=16, fontweight="bold")

    # Subplot 1: Espectro e Energia Cumulativa
    ax1 = axes[0, 0]
    x_ax = np.arange(1, 33)
    l_np = lambda_sorted.cpu().numpy()
    c_np = cum_energy.cpu().numpy() * 100.0
    color = "tab:blue"
    ax1.set_xlabel("Modo Funcional (k)", fontweight="bold")
    ax1.set_ylabel("Autovalor lambda_k (Sensibilidade)", color=color, fontweight="bold")
    ax1.semilogy(x_ax, l_np, marker="o", color=color, lw=2, label="lambda_k")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color_twin = "tab:red"
    ax1_twin.set_ylabel("Energia Funcional Cumulativa (%)", color=color_twin, fontweight="bold")
    ax1_twin.plot(x_ax, c_np, marker="s", color=color_twin, lw=2, linestyle="--", label="Cum. Energy")
    ax1_twin.tick_params(axis="y", labelcolor=color_twin)
    ax1_twin.axhline(90.0, color="green", linestyle=":", label="90% Threshold")
    ax1_twin.axhline(95.0, color="purple", linestyle=":", label="95% Threshold")
    ax1.set_title("Espectro Singular de G_c e Concentração Cumulativa", fontweight="bold")

    # Subplot 2: Comparação Multidomínio dos Autovalores de G_c
    ax2 = axes[0, 1]
    for d, c in zip(domains, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        L_d, _ = torch.linalg.eigh(G_c_dict[d])
        L_d_desc = torch.sort(L_d, descending=True).values.cpu().numpy()
        ax2.semilogy(x_ax, L_d_desc, marker=".", lw=1.8, label=d, color=c)
    ax2.set_xlabel("Modo Funcional (k)", fontweight="bold")
    ax2.set_ylabel("Autovalores de G_c^e", fontweight="bold")
    ax2.set_title("Estabilidade Multidomínio do Espectro Funcional", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Subplot 3: Dispersão Energia do Erro vs Sensibilidade Funcional
    ax3 = axes[1, 0]
    e_vals = [vm["e_norm"] * 100.0 for vm in vector_metrics]
    s_vals = [vm["s_norm"] * 100.0 for vm in vector_metrics]
    sc = ax3.scatter(e_vals, s_vals, c=[vm["ratio"] for vm in vector_metrics], cmap="viridis", s=100, edgecolors="k")
    for i in range(32):
        if vector_metrics[i]["ratio"] > 1.15 or vector_metrics[i]["s_norm"] > 0.05:
            ax3.annotate(f"u_{i+1}", (e_vals[i], s_vals[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax3.plot([0, max(e_vals)], [0, max(e_vals)], "r--", alpha=0.5, label="Equilibrio (Ratio = 1.0)")
    ax3.set_xlabel("Energia Euclidiana do Erro (%)", fontweight="bold")
    ax3.set_ylabel("Sensibilidade Funcional (%)", fontweight="bold")
    ax3.set_title("Experimento F: Energia Euclidiana vs Sensibilidade Funcional", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    cbar = fig.colorbar(sc, ax=ax3)
    cbar.set_label("Razao Funcional / Energia", fontweight="bold")

    # Subplot 4: Razão de Alavancagem por Vetor
    ax4 = axes[1, 1]
    ratios = [vm["ratio"] for vm in vector_metrics]
    colors_bar = ["tab:green" if r > 1.2 else ("tab:blue" if r >= 0.8 else "tab:gray") for r in ratios]
    ax4.bar(x_ax, ratios, color=colors_bar, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax4.axhline(1.0, color="red", linestyle="--", label="Referencia Neutra (1.0)")
    ax4.set_xlabel("Vetor Funcional (u_i)", fontweight="bold")
    ax4.set_ylabel("Razao s_i / (e_i + eps)", fontweight="bold")
    ax4.set_title("Alavancagem Funcional Relativa por Vetor", fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    chart_path = os.path.join(repo_root, "experiments", "functional_common_spectrum.png")
    fig.savefig(chart_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "functional_common_spectrum.png"), dpi=300)
    plt.close(fig)
    print(f"[OK] Grafico salvo em: {chart_path} e raiz do repositorio.")
    print("=" * 110)

if __name__ == "__main__":
    main()
