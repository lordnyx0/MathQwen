# -*- coding: utf-8 -*-
"""
experiments/functional_residual_spectrum.py

CICLO 26: Espectro Funcional do Complemento Ortogonal U_c^\perp
--------------------------------------------------------------
Objetivo:
Determinar a dimensao funcional efetiva r_perp^* dos 5.088 graus de liberdade
restantes em U_c^\perp atraves do objeto funcional exato:
    G = E_n [ J_n^T J_n ] in R^(5120x5120)
    G_perp = P_perp G P_perp

Metricas Extraidas:
    - Autovalores ordenados lambda_1 >= lambda_2 >= ... >= lambda_5088 >= 0
    - Dimensoes efetivas r_50, r_75, r_90, r_95, r_99 para Wiki, Code, GSM e Global
    - Comparacao entre Energia Geometrica C_h e Sensibilidade Funcional G_perp
    - Sondas Causais Delta PPL(v_spectral) vs Delta PPL(v_random)
    - Analise de sensibilidade funcional camada a camada l in {0, 8, 16, 24, 32, 40, 48, 56, 63}
    - Overlap de Grassmann e angulos canonicos interdominios em U_c^\perp
    - Classificacao formal nos Cenarios A, B, C ou D
"""

import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

repo_root = r"C:\Users\Nyx\Desktop\MathQwen"
sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, get_snapshot_dir

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("CICLO 26: ESPECTRO FUNCIONAL DO COMPLEMENTO ORTOGONAL U_c^\\perp")
    print("=" * 115)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0)}) | Inicio: {time.strftime('%H:%M:%S')}")

    # -------------------------------------------------------------------------
    # 1. Estado Congelado & Definição do Complemento Ortogonal
    # -------------------------------------------------------------------------
    print("\n[Passo 1/7] Carregando estado congelado e validando ortogonalidade de U_c(32)...")
    
    # Carregar U_c(32)
    uc_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    uc_data = torch.load(uc_path, map_location=device)
    Uc = uc_data["U_f"].to(device=device, dtype=torch.float32)
    Uc, _ = torch.linalg.qr(Uc) # Garantir ortonormalidade exata de maquina
    
    ortho_err = torch.norm(torch.matmul(Uc.t(), Uc) - torch.eye(32, device=device)).item()
    print(f"  [OK] U_c(32) carregado. Erro de ortonormalidade ||U_c^T U_c - I|| = {ortho_err:.2e}")
    assert ortho_err < 1e-4, f"U_c nao e ortonormal: {ortho_err}"

    # Carregar componentes de referencia congelados
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=2)
    lm_w = comp["lm_head_w"] # [248320, 5120], bfloat16
    norm_mod = comp["norm_mod"]
    for p in norm_mod.parameters():
        p.requires_grad = False
    norm_mod.eval()
    gamma = norm_mod.weight.data.to(device=device, dtype=torch.float32) # [5120]
    vocab_sz, d_model = lm_w.shape
    print(f"  [OK] Referencia carregada: Vocab={vocab_sz}, d_model={d_model}, RMSNorm gamma shape={gamma.shape}")

    # Funcao de projecao em U_c^perp sem materializar matriz 5120x5120
    def project_perp_vec(x):
        # x: [..., 5120]
        # x_perp = x - (x @ Uc) @ Uc^T
        return x - torch.matmul(torch.matmul(x, Uc), Uc.t())

    def project_perp_mat(M):
        # M: [5120, 5120]
        # P_perp M P_perp = (M - (M @ Uc) @ Uc^T) - Uc @ (Uc^T @ (M - (M @ Uc) @ Uc^T))
        MU = torch.matmul(M, Uc)
        M_Pperp = M - torch.matmul(MU, Uc.t())
        M_perp = M_Pperp - torch.matmul(Uc, torch.matmul(Uc.t(), M_Pperp))
        return 0.5 * (M_perp + M_perp.t())

    # Validar ||U_c^T P_perp||_F approx 0
    test_rand = torch.randn(100, d_model, device=device)
    test_perp = project_perp_vec(test_rand)
    leakage = torch.norm(torch.matmul(test_perp, Uc)).item() / torch.norm(test_rand).item()
    print(f"  [OK] Verificacao de P_perp: vazamento residual ||P_perp(X) U_c|| / ||X|| = {leakage:.2e}")

    # -------------------------------------------------------------------------
    # 2. Construcao da Matriz Gram da Cabeca de Saida M_head
    # -------------------------------------------------------------------------
    print("\n[Passo 2/7] Construindo matriz Gram da cabeca M_head = W_head^T W_head em blocos...")
    t0_head = time.time()
    M_head = torch.zeros(d_model, d_model, device=device, dtype=torch.float32)
    chunk_sz = 31040 # 8 chunks de 31.040 linhas
    for c_idx in range(0, vocab_sz, chunk_sz):
        w_c = lm_w[c_idx:c_idx + chunk_sz].float()
        M_head.addmm_(w_c.t(), w_c)
    torch.cuda.synchronize()
    print(f"  [OK] M_head [5120x5120] construida em {time.time()-t0_head:.2f}s (Norma: {torch.norm(M_head):.2e})")

    # -------------------------------------------------------------------------
    # 3. Carregando Ativacoes Multi-Dominio (WikiText, Code, GSM8K, Global)
    # -------------------------------------------------------------------------
    print("\n[Passo 3/7] Carregando e alinhando dados multi-dominio em L63...")
    # WikiText
    p_data = torch.load(os.path.join(repo_root, "checkpoints", "prof_targets_cache.pt"), map_location=device)
    h_wiki_63 = p_data["prof_intermediates_tr"][63].to(device=device, dtype=torch.float32) # [16, 64, 5120]
    h_wiki_flat = h_wiki_63.view(-1, d_model) # [1024, 5120]

    # HF Code
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_63 = hf_data["h_in_63"].to(device=device, dtype=torch.float32) # [16, 128, 5120]
    h_code_flat = h_code_63.view(-1, d_model)[:1024] # Alinhar a 1024 tokens

    # GSM8K
    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_63 = gsm_data["h_in_63"].to(device=device, dtype=torch.float32) # [16, 128, 5120]
    h_gsm_flat = h_gsm_63.view(-1, d_model)[:1024] # Alinhar a 1024 tokens

    # Global Multitask Union
    h_global_flat = torch.cat([h_wiki_flat, h_code_flat, h_gsm_flat], dim=0) # [3072, 5120]
    print(f"  WikiText tokens: {h_wiki_flat.shape[0]} | Code: {h_code_flat.shape[0]} | GSM8K: {h_gsm_flat.shape[0]} | Global: {h_global_flat.shape[0]}")

    # -------------------------------------------------------------------------
    # 4. Computando Matriz Funcional G = E_n [ J_n^T J_n ] Exata
    # -------------------------------------------------------------------------
    print("\n[Passo 4/7] Computando matriz funcional analitica exata G = E_n[J_n^T J_n] por dominio...")
    
    def compute_exact_G(h_tokens):
        # h_tokens: [N, 5120]
        N = h_tokens.shape[0]
        # s_n = rsqrt(mean(h_n^2) + eps)
        s = (torch.mean(h_tokens**2, dim=-1, keepdim=True) + 1e-6).rsqrt() # [N, 1]
        u = (s**3 / d_model) * (gamma * h_tokens) # [N, 5120]
        w = h_tokens # [N, 5120]

        # Term 1: E[s_n^2] * (diag(gamma) @ M_head @ diag(gamma))
        mean_s2 = torch.mean(s**2)
        term1 = mean_s2 * (gamma.unsqueeze(1) * M_head * gamma.unsqueeze(0))

        # Term 2: (1/N) * [ w^T @ (s * (Q * gamma)) + transposto ]
        Q = torch.matmul(u, M_head) # [N, 5120]
        Q_gamma = s * (Q * gamma) # [N, 5120]
        term2 = torch.matmul(w.t(), Q_gamma) / N
        term2 = term2 + term2.t()

        # Term 3: (1/N) * w^T @ diag(u^T M_head u) @ w
        alpha = torch.sum(u * Q, dim=-1, keepdim=True) # [N, 1]
        term3 = torch.matmul(w.t(), alpha * w) / N

        G = term1 - term2 + term3
        return 0.5 * (G + G.t())

    domains = {
        "WikiText": h_wiki_flat,
        "HF Code": h_code_flat,
        "GSM8K": h_gsm_flat,
        "Global": h_global_flat
    }

    spectra = {}
    top_eigenvecs = {}
    G_perp_dict = {}

    for name, h_data in domains.items():
        t0_d = time.time()
        G_mat = compute_exact_G(h_data)
        G_perp = project_perp_mat(G_mat)
        G_perp_dict[name] = G_perp

        # Eigendecomposicao exata
        vals, vecs = torch.linalg.eigh(G_perp)
        vals = torch.flip(vals, dims=[0])
        vecs = torch.flip(vecs, dims=[1]) # [5120, 5120]

        # 5088 autovalores ativos em U_c^perp
        pos_vals = vals[:5088]
        pos_vals = torch.clamp(pos_vals, min=0.0)
        total_energy = torch.sum(pos_vals).item()
        cum_energy = torch.cumsum(pos_vals, dim=0) / total_energy

        # Dimensoes efetivas
        r50 = (torch.searchsorted(cum_energy, 0.50) + 1).item()
        r75 = (torch.searchsorted(cum_energy, 0.75) + 1).item()
        r90 = (torch.searchsorted(cum_energy, 0.90) + 1).item()
        r95 = (torch.searchsorted(cum_energy, 0.95) + 1).item()
        r99 = (torch.searchsorted(cum_energy, 0.99) + 1).item()

        # Armazenar
        spectra[name] = {
            "eigenvalues": pos_vals.cpu().numpy().tolist(),
            "cumulative_energy": cum_energy.cpu().numpy().tolist(),
            "total_energy": total_energy,
            "r50": r50,
            "r75": r75,
            "r90": r90,
            "r95": r95,
            "r99": r99,
            "lambda_1": pos_vals[0].item(),
            "lambda_10": pos_vals[9].item(),
            "lambda_32": pos_vals[31].item(),
            "lambda_64": pos_vals[63].item(),
            "lambda_256": pos_vals[255].item(),
            "lambda_1024": pos_vals[1023].item(),
        }
        top_eigenvecs[name] = vecs[:, :256].detach() # Top-256 modos

        print(f"  [{name}] G_perp calculado em {time.time()-t0_d:.2f}s | lambda_1={pos_vals[0].item():.2f}, lambda_256={pos_vals[255].item():.2f}")
        print(f"         r_50={r50} | r_75={r75} | r_90={r90} | r_95={r95} | r_99={r99}")

    # Validacao com Eigensolver Aleatorizado (Randomized Range Finder + SVD)
    print("\n  Validando com Randomized Range Finder + SVD (Halko et al., 2011, k=256)...")
    k_rand = 256
    Omega = torch.randn(d_model, k_rand, device=device)
    G_glob = G_perp_dict["Global"]
    Y = torch.matmul(G_glob, Omega)
    Y = torch.matmul(G_glob, Y)
    Q_rand, _ = torch.linalg.qr(Y)
    B = torch.matmul(Q_rand.t(), torch.matmul(G_glob, Q_rand))
    eig_b, _ = torch.linalg.eigh(B)
    rand_vals = torch.flip(eig_b, dims=[0])
    exact_vals = torch.tensor(spectra["Global"]["eigenvalues"][:k_rand], device=device)
    rand_rel_err = torch.mean(torch.abs(rand_vals - exact_vals) / (exact_vals + 1e-6)).item()
    print(f"  [OK] Concordancia Randomized SVD vs Eigh exato (k=256): Erro relativo medio = {rand_rel_err:.2e}")

    # -------------------------------------------------------------------------
    # 5. Comparacao: Energia Geometrica C_h vs Sensibilidade Funcional G_perp
    # -------------------------------------------------------------------------
    print("\n[Passo 5/7] Comparando Energia Geometrica de Ativacao C_h vs Sensibilidade Funcional G_perp...")
    h_glob_perp = project_perp_vec(h_global_flat)
    N_glob = h_glob_perp.shape[0]
    C_h = torch.matmul(h_glob_perp.t(), h_glob_perp) / N_glob
    C_h_perp = project_perp_mat(C_h)

    vals_ch, vecs_ch = torch.linalg.eigh(C_h_perp)
    vals_ch = torch.flip(vals_ch, dims=[0])[:5088]
    vals_ch = torch.clamp(vals_ch, min=0.0)
    vecs_ch = torch.flip(vecs_ch, dims=[1])
    cum_ch = torch.cumsum(vals_ch, dim=0) / torch.sum(vals_ch)

    r50_ch = (torch.searchsorted(cum_ch, 0.50) + 1).item()
    r75_ch = (torch.searchsorted(cum_ch, 0.75) + 1).item()
    r90_ch = (torch.searchsorted(cum_ch, 0.90) + 1).item()
    r95_ch = (torch.searchsorted(cum_ch, 0.95) + 1).item()
    r99_ch = (torch.searchsorted(cum_ch, 0.99) + 1).item()

    print(f"  Energia Geometrica (C_h): r_50={r50_ch} | r_75={r75_ch} | r_90={r90_ch} | r_95={r95_ch} | r_99={r99_ch}")
    print(f"  Sensibilidade Funcional (G_perp Global): r_50={spectra['Global']['r50']} | r_90={spectra['Global']['r90']} | r_95={spectra['Global']['r95']} | r_99={spectra['Global']['r99']}")

    # Identificando direcoes disjuntas:
    v_high_geom = vecs_ch[:, 0]
    sens_of_high_geom = torch.matmul(v_high_geom.t(), torch.matmul(G_perp_dict["Global"], v_high_geom)).item()
    top_sens = spectra["Global"]["lambda_1"]

    v_high_func = top_eigenvecs["Global"][:, 0]
    geom_of_high_func = torch.matmul(v_high_func.t(), torch.matmul(C_h_perp, v_high_func)).item()
    top_geom = vals_ch[0].item()

    print(f"  Direcao com Top Energia Geometrica: Energia={top_geom:.2f}, Sensibilidade Funcional={sens_of_high_geom:.2f} ({sens_of_high_geom/top_sens*100:.2f}% do max)")
    print(f"  Direcao com Top Sensibilidade Funcional: Sensibilidade={top_sens:.2f}, Energia Geometrica={geom_of_high_func:.2f} ({geom_of_high_func/top_geom*100:.2f}% do max)")

    # -------------------------------------------------------------------------
    # 6. Sondagem Causal dos Top-16 Modos Espectrais vs Controle Aleatorio
    # -------------------------------------------------------------------------
    print("\n[Passo 6/7] Executando sondagem causal dos top-16 modos espectrais vs controle aleatorio em U_c^perp...")
    h_test_eval = p_data["h_prof_test_final"][:8, :64].to(device=device, dtype=torch.bfloat16) # [8, 64, 5120]
    targets_eval = hf_data["input_ids"][:8, :64].to(device=device) # Targets de teste

    def evaluate_loss_ppl(h_mod):
        out = norm_mod(h_mod)
        logits = F.linear(out, lm_w)
        l_flat = logits[:, :-1, :].reshape(-1, vocab_sz)
        tgt_flat = targets_eval[:, 1:].reshape(-1)
        ce = F.cross_entropy(l_flat.float(), tgt_flat, reduction="mean").item()
        ppl = math.exp(min(ce, 20.0))
        return ce, ppl

    ce_base, ppl_base = evaluate_loss_ppl(h_test_eval)
    print(f"  Baseline Held-out: CE={ce_base:.4f} | PPL={ppl_base:.2f}")

    alphas = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    causal_results = {"alphas": alphas, "spectral_modes": [], "random_controls": []}

    torch.manual_seed(42)
    rand_raw = torch.randn(d_model, 16, device=device)
    rand_perp = project_perp_vec(rand_raw.t()).t()
    rand_ortho, _ = torch.linalg.qr(rand_perp) # [5120, 16]

    spectral_v16 = top_eigenvecs["Global"][:, :16] # [5120, 16]

    spectral_mean_dppl = []
    random_mean_dppl = []

    for idx in range(16):
        v_spec = spectral_v16[:, idx].to(dtype=torch.bfloat16)
        v_rand = rand_ortho[:, idx].to(dtype=torch.bfloat16)

        dppl_spec_list = []
        dppl_rand_list = []

        for a in alphas:
            if a == 0.0:
                dppl_spec_list.append(0.0)
                dppl_rand_list.append(0.0)
                continue
            # Espectral
            _, ppl_s = evaluate_loss_ppl(h_test_eval + a * v_spec)
            dppl_spec_list.append(ppl_s - ppl_base)
            # Aleatorio
            _, ppl_r = evaluate_loss_ppl(h_test_eval + a * v_rand)
            dppl_rand_list.append(ppl_r - ppl_base)

        causal_results["spectral_modes"].append({
            "mode": idx + 1,
            "lambda": spectra["Global"]["eigenvalues"][idx],
            "dppl": dppl_spec_list
        })
        causal_results["random_controls"].append({
            "mode": idx + 1,
            "dppl": dppl_rand_list
        })
        spectral_mean_dppl.append(float(np.mean([abs(x) for x in dppl_spec_list])))
        random_mean_dppl.append(float(np.mean([abs(x) for x in dppl_rand_list])))

    mean_causal_ratio = float(np.mean(spectral_mean_dppl) / (np.mean(random_mean_dppl) + 1e-6))
    print(f"  [OK] Validacao Causal: |Delta PPL(v_spectral)| = {np.mean(spectral_mean_dppl):.2f}")
    print(f"                         |Delta PPL(v_random)|   = {np.mean(random_mean_dppl):.2f}")
    print(f"                         Razao de Efeito Causal  = {mean_causal_ratio:.2f}x")

    # -------------------------------------------------------------------------
    # 7. Analise de Sensibilidade Funcional por Camada
    # -------------------------------------------------------------------------
    print("\n[Passo 7/7] Analisando espectro funcional residual por camada l in {0,8,16,24,32,40,48,56,63}...")
    checkpoints_l = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    layerwise_results = {}

    N_wiki = h_wiki_flat.shape[0]

    for l in checkpoints_l:
        h_l = p_data["prof_intermediates_tr"][l].to(device=device, dtype=torch.float32).view(N_wiki, d_model)
        if l == 63:
            G_l_perp = G_perp_dict["WikiText"]
        else:
            # Propagador linear de resposta: T = (H_l^T H_l + reg I)^(-1) H_l^T H_63
            reg = 1e-2 * torch.norm(h_l) / N_wiki
            HlTHl = torch.matmul(h_l.t(), h_l) + reg * torch.eye(d_model, device=device)
            HlT_H63 = torch.matmul(h_l.t(), h_wiki_flat)
            T_prop = torch.linalg.solve(HlTHl, HlT_H63)
            G_l = torch.matmul(T_prop, torch.matmul(G_perp_dict["WikiText"], T_prop.t()))
            G_l_perp = project_perp_mat(G_l)

        vals_l, _ = torch.linalg.eigh(G_l_perp)
        vals_l = torch.flip(vals_l, dims=[0])[:5088]
        vals_l = torch.clamp(vals_l, min=0.0)
        cum_l = torch.cumsum(vals_l, dim=0) / torch.sum(vals_l)

        r50_l = (torch.searchsorted(cum_l, 0.50) + 1).item()
        r90_l = (torch.searchsorted(cum_l, 0.90) + 1).item()
        r95_l = (torch.searchsorted(cum_l, 0.95) + 1).item()

        layerwise_results[f"L{l:02d}"] = {
            "layer": l,
            "r50": r50_l,
            "r90": r90_l,
            "r95": r95_l,
            "lambda_1": float(vals_l[0].item())
        }
        print(f"  L{l:02d}: r_50={r50_l} | r_90={r90_l} | r_95={r95_l} | lambda_1={vals_l[0].item():.2f}")

    # -------------------------------------------------------------------------
    # 8. Overlap de Grassmann e Angulos Canonicos Interdominios em U_c^perp
    # -------------------------------------------------------------------------
    print("\nCalculando matriz de overlap de Grassmann dos top modos em U_c^perp...")
    task_names = ["WikiText", "HF Code", "GSM8K"]
    overlap_matrix = {}
    for t1 in task_names:
        overlap_matrix[t1] = {}
        for t2 in task_names:
            V1 = top_eigenvecs[t1][:, :64] # [5120, 64]
            V2 = top_eigenvecs[t2][:, :64] # [5120, 64]
            V1T_V2 = torch.matmul(V1.t(), V2)
            overlap = (torch.norm(V1T_V2)**2 / 64.0).item()
            overlap_matrix[t1][t2] = float(overlap)

    print("  Matriz de Overlap (k=64):")
    for t1 in task_names:
        row = " | ".join([f"{t2}: {overlap_matrix[t1][t2]:.3f}" for t2 in task_names])
        print(f"    {t1:8s} -> {row}")

    # -------------------------------------------------------------------------
    # 9. Estimativa Estrutural, Classificacao nos Cenarios e Salvar Resultados
    # -------------------------------------------------------------------------
    r_perp_star = spectra["Global"]["r95"]
    r_total_min = 32 + r_perp_star
    pct_original = (r_total_min / d_model) * 100.0

    # Classificacao de Cenario
    if r_perp_star < 64:
        scenario = "Cenario A (Complemento Fortemente Concentrado)"
    elif r_perp_star < 256:
        scenario = "Cenario B (Complemento Moderadamente Concentrado)"
    elif r_perp_star < 2048:
        scenario = "Cenario C- (Complemento de Media-Alta Dimensao)"
    else:
        scenario = "Cenario C/D (Complemento de Altissima Dimensao / Espectro Espalhado)"

    print("\n" + "=" * 115)
    print("DIAGNOSTICO FINAL DO CICLO 26:")
    print(f"  Menor dimensao residual funcional estimada (r_perp,95): {r_perp_star}")
    print(f"  Candidato estrutural minimo: r_total = 32 + {r_perp_star} = {r_total_min} ({pct_original:.2f}% dos 5.120 eixos)")
    print(f"  Classificacao Formal: {scenario}")
    print("=" * 115)

    final_results = {
        "spectra": {
            k: {
                "r50": v["r50"], "r75": v["r75"], "r90": v["r90"], "r95": v["r95"], "r99": v["r99"],
                "lambda_1": v["lambda_1"], "lambda_10": v["lambda_10"], "lambda_32": v["lambda_32"],
                "lambda_64": v["lambda_64"], "lambda_256": v["lambda_256"], "lambda_1024": v["lambda_1024"],
                "top_50_eigenvalues": v["eigenvalues"][:50],
                "top_50_cumulative": v["cumulative_energy"][:50]
            } for k, v in spectra.items()
        },
        "geometric_activation_energy": {
            "r50": r50_ch, "r75": r75_ch, "r90": r90_ch, "r95": r95_ch, "r99": r99_ch,
            "top_10_eigenvalues": vals_ch[:10].cpu().numpy().tolist()
        },
        "causal_validation": {
            "mean_causal_ratio": mean_causal_ratio,
            "mean_spectral_dppl": float(np.mean(spectral_mean_dppl)),
            "mean_random_dppl": float(np.mean(random_mean_dppl)),
            "modes": causal_results["spectral_modes"]
        },
        "layerwise_sensitivity": layerwise_results,
        "domain_overlap_k64": overlap_matrix,
        "structural_estimate": {
            "r_perp_star": r_perp_star,
            "r_total_min": r_total_min,
            "pct_original": float(pct_original),
            "scenario": scenario
        },
        "elapsed_seconds": float(time.time() - t_start)
    }

    # Salvar JSON no diretorio experiments/
    json_path = os.path.join(repo_root, "experiments", "functional_residual_spectrum_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResultados gravados em: {json_path}")

    # -------------------------------------------------------------------------
    # 10. Geracao dos Graficos Diagnosticos
    # -------------------------------------------------------------------------
    print("\nGerando graficos diagnosticos...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Grafico 1: Espectro Funcional vs Geometrico
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ranks = np.arange(1, 5089)
    for name in ["Global", "WikiText", "HF Code", "GSM8K"]:
        axes[0].plot(ranks, spectra[name]["cumulative_energy"], label=f"G_perp ({name}) - r95={spectra[name]['r95']}", lw=2)
    axes[0].plot(ranks, cum_ch.cpu().numpy(), label=f"C_h (Ativacao Geom.) - r95={r95_ch}", color="black", linestyle="--", lw=2)
    axes[0].axhline(0.95, color="red", linestyle=":", alpha=0.7, label="Limiar 95%")
    axes[0].axhline(0.90, color="orange", linestyle=":", alpha=0.7, label="Limiar 90%")
    axes[0].set_xlim(1, 5088)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Numero de Modos em U_c^perp (Posto Residual r)")
    axes[0].set_ylabel("Energia Funcional Cumulativa E(r)")
    axes[0].set_title("Espectro Funcional Cumulativo E(r) vs Energia Geometrica C_h", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right")

    for name in ["Global", "WikiText", "HF Code", "GSM8K"]:
        axes[1].semilogy(ranks[:1000], spectra[name]["eigenvalues"][:1000], label=f"{name}", lw=1.8)
    axes[1].semilogy(ranks[:1000], vals_ch[:1000].cpu().numpy(), label="C_h Geometrico", color="black", linestyle="--", lw=1.8)
    axes[1].set_xlabel("Indice do Modo i (Top 1000)")
    axes[1].set_ylabel("Autovalor lambda_i (Escala Log)")
    axes[1].set_title("Decaimento dos Autovalores de G_perp e C_h", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    p1 = os.path.join(repo_root, "experiments", "residual_functional_spectrum.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  [OK] Grafico 1 salvo: {p1}")

    # Grafico 2: Sensibilidade por Camada
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_indices = [v["layer"] for v in layerwise_results.values()]
    r95_layers = [v["r95"] for v in layerwise_results.values()]
    r90_layers = [v["r90"] for v in layerwise_results.values()]
    r50_layers = [v["r50"] for v in layerwise_results.values()]

    ax.plot(layer_indices, r95_layers, marker="o", lw=2.5, color="#1f77b4", label="r95 (95% sensibilidade)")
    ax.plot(layer_indices, r90_layers, marker="s", lw=2.0, color="#ff7f0e", label="r90 (90% sensibilidade)")
    ax.plot(layer_indices, r50_layers, marker="^", lw=1.5, color="#2ca02c", label="r50 (50% sensibilidade)")
    ax.set_xticks(checkpoints_l)
    ax.set_xlabel("Profundidade da Camada (l)")
    ax.set_ylabel("Dimensao Residual Efetiva r_alpha")
    ax.set_title("Evolucao da Dimensao Funcional Efetiva ao Longo das Camadas", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(repo_root, "experiments", "residual_functional_layerwise.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  [OK] Grafico 2 salvo: {p2}")

    # Grafico 3: Overlap entre Dominios
    fig, ax = plt.subplots(figsize=(7, 6))
    overlap_data = np.zeros((3, 3))
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            overlap_data[i, j] = overlap_matrix[t1][t2]
    
    im = ax.imshow(overlap_data, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(task_names)
    ax.set_yticklabels(task_names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{overlap_data[i, j]:.3f}", ha="center", va="center", color="black" if overlap_data[i, j] < 0.6 else "white", fontweight="bold")
    ax.set_title("Matriz de Overlap de Grassmann dos Top-64 Modos em U_c^perp", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Overlap (1/k) ||V_1^T V_2||_F^2")
    plt.tight_layout()
    p3 = os.path.join(repo_root, "experiments", "residual_functional_domain_overlap.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  [OK] Grafico 3 salvo: {p3}")

    print(f"\nCiclo 26 concluido com sucesso em {time.time()-t_start:.1f}s.")

if __name__ == "__main__":
    main()
