# -*- coding: utf-8 -*-
"""
Universal Error Manifold (UEM) - Construcao do Refinador Universal e Validacao Cruzada
Gera os estabilizadores analiticos de forma fechada W_down = U, W_up = (Z^T Z + lambda I)^{-1} Z^T E
e avalia a generalizacao funcional cruzada entre dominios out-of-distribution (A -> B, C).
"""
import os
import sys
import time
import json
import math
import copy
import torch
import torch.nn.functional as F
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

from reference.loader import load_qwen_reference_components, get_snapshot_dir
from atlas.residual import LinearResidualStabilizer

def fit_closed_form_refiner(X_fit: torch.Tensor, E_fit: torch.Tensor, U_base: torch.Tensor, lambda_reg: float = 1e-3):
    """
    Ajuste em forma fechada do refinador linear:
    W_down = U_base (5120, r)
    Z = X @ W_down (N, r)
    W_up = (Z^T Z + lambda I)^{-1} Z^T E (r, 5120)
    """
    W_down = U_base.to(dtype=torch.bfloat16)
    Z = torch.matmul(X_fit.float(), W_down.float()) # (N, r)
    r = W_down.shape[1]

    ZTZ = torch.matmul(Z.t(), Z)
    reg_val = lambda_reg * (torch.trace(ZTZ) / float(r))
    reg_I = reg_val * torch.eye(r, device=X_fit.device, dtype=torch.float32)

    L = torch.linalg.cholesky(ZTZ + reg_I)
    W_up_opt = torch.cholesky_solve(torch.matmul(Z.t(), E_fit.float()), L)
    W_up = W_up_opt.to(dtype=torch.bfloat16)

    stab = LinearResidualStabilizer(W_down, W_up)
    stab.freeze()
    return stab

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("UNIVERSAL ERROR MANIFOLD (UEM) - CONSTRUCAO DO REFINADOR E VALIDACAO CRUZADA")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar bases UEM salvas
    bases_path = os.path.join(repo_root, "checkpoints", "uem_bases_l63.pt")
    if not os.path.exists(bases_path):
        print(f"ERRO: Bases UEM {bases_path} nao encontradas. Execute experiments/uem_analysis.py primeiro.")
        sys.exit(1)

    print(f"\n[Passo 1/4] Carregando bases UEM de: {bases_path}...")
    uem_bases = torch.load(bases_path, map_location=device)
    U_avg_all = uem_bases["U_avg_all"].to(device)
    U_minimax_dict = {k: v.to(device) for k, v in uem_bases["U_minimax"].items()}

    # 2. Carregar estados X e erros E de cada dominio
    err_cache_file = os.path.join(repo_root, "checkpoints", "uem_errors_l63.pt")
    errors_raw = torch.load(err_cache_file, map_location=device)
    errors = {k: v.to(device) for k, v in errors_raw.items()}

    h_wiki_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    h_hf_path = os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt")
    h_gsm8k_path = os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt")

    X_dict = {
        "WikiText": torch.load(h_wiki_path, map_location=device).reshape(-1, 5120).to(device, dtype=torch.bfloat16),
        "HF_Code": torch.load(h_hf_path, map_location=device)["h_in_63"].to(device).reshape(-1, 5120).to(torch.bfloat16)
    }
    if os.path.exists(h_gsm8k_path):
        X_dict["GSM8K"] = torch.load(h_gsm8k_path, map_location=device)["h_in_63"].to(device).reshape(-1, 5120).to(torch.bfloat16)
    else:
        # split se GSM8k nao disponivel
        half = X_dict["WikiText"].shape[0] // 2
        X_dict = {
            "WikiText_Train": X_dict["WikiText"][:half],
            "WikiText_Test": X_dict["WikiText"][half:],
            "HF_Code": X_dict["HF_Code"]
        }

    domain_names = list(errors.keys())
    print(f"  -> Dominios ativos para validacao: {domain_names}")

    ranks = [16, 32, 64, 128, 256]
    candidate_types = ["avg", "minimax"]

    # 3. Validacao Cruzada de Reducao de Erro Funcional
    print("\n[Passo 2/4] Executando Ajuste Fechado em cada Dominio e Avaliacao Cruzada...")
    # Para cada par de (Dominio de Treino -> Dominios de Teste)
    # Medir a variancia residual retida: ||E_test - C(X_test)||_F^2 / ||E_test||_F^2
    validation_records = []

    for c_type in candidate_types:
        print(f"\n==================== AVALIANDO BASE UNIVERSAL: U_{c_type.upper()} ====================")
        for r in ranks:
            if c_type == "avg":
                U_cand = U_avg_all[:, :r]
            else:
                U_cand = U_minimax_dict[r][:, :r]

            for d_fit in domain_names:
                X_fit = X_dict[d_fit]
                E_fit = errors[d_fit]

                # Ajuste analitico fechado
                stab = fit_closed_form_refiner(X_fit, E_fit, U_cand)

                # Avaliar em todos os dominios (in-domain e out-of-domain)
                rec = {
                    "base_type": c_type,
                    "rank": r,
                    "fit_domain": d_fit,
                    "test_metrics": {},
                    "worst_retention_pct": 100.0
                }

                retentions = []
                for d_test in domain_names:
                    X_test = X_dict[d_test]
                    E_test = errors[d_test]
                    norm_orig_sq = (E_test.float() ** 2).sum().item()

                    with torch.no_grad():
                        # C(X) = X @ W_down @ W_up
                        corr = torch.matmul(torch.matmul(X_test, stab.W_down), stab.W_up).float()
                        residual_err = E_test.float() - corr
                        norm_res_sq = (residual_err ** 2).sum().item()

                    unexplained_ratio = norm_res_sq / norm_orig_sq
                    captured_ratio = max(0.0, 1.0 - unexplained_ratio) * 100.0
                    retentions.append(captured_ratio)
                    rec["test_metrics"][d_test] = {
                        "unexplained_error_pct": unexplained_ratio * 100.0,
                        "captured_error_pct": captured_ratio
                    }

                rec["worst_retention_pct"] = float(np.min(retentions))
                rec["mean_retention_pct"] = float(np.mean(retentions))
                validation_records.append(rec)

                out_str = " | ".join([f"{d}: {rec['test_metrics'][d]['captured_error_pct']:<5.1f}%" for d in domain_names])
                print(f"U_{c_type:<7} | r={r:<3} | Treino: {d_fit:<16} | Testes: [{out_str}] | Pior Caso: {rec['worst_retention_pct']:<5.1f}%")

    # 4. Tabela Comparativa Consolidada (Treino em WikiText -> Teste nos demais)
    print("\n" + "=" * 110)
    print("                    TABELA DE GENERALIZACAO CRUZADA OUT-OF-DOMAIN (TREINO EM DOMINIO A)")
    print("=" * 110)
    header = f"{'Base':<8} | {'Rank':<6} | {'Treino':<16} | " + " | ".join([f"Teste {dn:<10}" for dn in domain_names]) + " | {'Pior Caso':<10} | {'Media':<8}"
    print(header)
    print("-" * 110)
    for rec in validation_records:
        if rec["fit_domain"] == domain_names[0]: # Treino no Dominio 1 (ex: WikiText)
            t_vals = " | ".join([f"{rec['test_metrics'][dn]['captured_error_pct']:<15.2f}%" for dn in domain_names])
            print(f"U_{rec['base_type']:<6} | r={rec['rank']:<4} | {rec['fit_domain']:<16} | {t_vals} | {rec['worst_retention_pct']:<9.2f}% | {rec['mean_retention_pct']:<7.2f}%")
    print("=" * 110)

    # Salvar resultados em JSON
    val_json_path = os.path.join(repo_root, "experiments", "uem_refiner_validation.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "domain_names": domain_names,
            "ranks": ranks,
            "validation_records": validation_records
        }, f, indent=2)
    print(f"\n[OK] Resultados de validacao cruzada salvos em: {val_json_path}")

    # 5. Geracao do Grafico (PNG)
    print("\nGerando grafico de generalizacao cruzada...")
    plt.figure(figsize=(10, 6), dpi=150)
    # Filtrar treinos em Dominio 1
    d1 = domain_names[0]
    avg_worsts = [r["worst_retention_pct"] for r in validation_records if r["base_type"] == "avg" and r["fit_domain"] == d1]
    mm_worsts = [r["worst_retention_pct"] for r in validation_records if r["base_type"] == "minimax" and r["fit_domain"] == d1]

    plt.plot(ranks, avg_worsts, marker="s", lw=2.2, color="#d95f02", label=f"$U_{{\\rm avg}}$ (Ajustado em {d1} $\\to$ Pior Caso Out-of-Domain)")
    plt.plot(ranks, mm_worsts, marker="^", lw=2.5, color="#1b9e77", label=f"$U_{{\\rm minimax}}$ (Ajustado em {d1} $\\to$ Pior Caso Out-of-Domain)")
    plt.axhline(80, color="gray", ls=":", alpha=0.7, label="80% Erro Capturado Out-of-Domain")
    plt.axhline(90, color="gray", ls="-.", alpha=0.7, label="90% Erro Capturado Out-of-Domain")
    plt.title("Universal Error Manifold (UEM): Generalização Funcional Cruzada Out-of-Domain", fontsize=13, pad=12)
    plt.xlabel("Posto do Refinador ($r$)", fontsize=11)
    plt.ylabel("Erro Funcional Capturado no Pior Domínio (%)", fontsize=11)
    plt.xticks(ranks)
    plt.ylim(0, 102)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p_cross = os.path.join(repo_root, "uem_refiner_crossdomain.png")
    plt.savefig(p_cross)
    plt.close()
    print(f"  -> Salvo: {p_cross}")

    print("\n[OK] Analise UEM 3/3 (Refinador Universal Fechado) concluida com sucesso!")

if __name__ == "__main__":
    main()
