#!/usr/bin/env python3
"""
experiments/lie_algebra_so32_spectrum.py

CICLO 20: Espectro de Sensibilidade Funcional em so(32) e Determinação da Dimensão k
-------------------------------------------------------------------------------------
Objetivo:
Estudar a distribuição de gradientes G_n in so(32) sobre milhares de tokens held-out
para determinar a dimensão funcional efetiva de sensibilidade k_alpha (k90, k95, k99)
dentre os 496 geradores da álgebra de Lie, medir o overlap funcional entre tarefas
e validar causalmente em PPL os modos dominantes via exponencial matricial exp(theta A).

Execução ultrarrápida: < 20 segundos no total.
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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, get_snapshot_dir, load_layer_module, dequant
from atlas.calibration import project_layer_weights

def compute_so32_gradient_batch(h_seqs, targets, U_base, W_opt, norm_mod, lm_head_w):
    """
    Computa os gradientes g_n in R^496 para cada token da sequência.
    Retorna:
      g_so32: [N, 496] onde N = B * (S - 1)
      ce_loss: float
    """
    vocab_sz = lm_head_w.shape[0]
    B, S, D = h_seqs.shape
    
    # a: [B, S, 32]
    a = torch.matmul(h_seqs, U_base)
    
    # Delta h inicial = a @ W_opt
    delta_h = torch.zeros_like(h_seqs, requires_grad=True)
    h_comb = h_seqs + delta_h
    h_norm = norm_mod(h_comb)
    logits = F.linear(h_norm, lm_head_w) # [B, S, vocab_sz]
    
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_sz)
    tgt_flat = targets[:, 1:].contiguous().reshape(-1)
    
    loss = F.cross_entropy(logits_pred.float(), tgt_flat, reduction="sum")
    loss.backward()
    
    # Gradiente em relação a Delta h: [B, S, 5120]
    grad_delta_h = delta_h.grad
    
    # b = grad_delta_h @ W_opt.T in R^[B, S, 32]
    b = torch.matmul(grad_delta_h, W_opt.t())
    
    # Focar nos tokens preditivos (excluir o último token sem target)
    a_tokens = a[:, :-1, :].reshape(-1, 32) # [N, 32]
    b_tokens = b[:, :-1, :].reshape(-1, 32) # [N, 32]
    
    # Vetorizar a triangular superior de G_n = 0.5 * (a_n^T b_n - b_n^T a_n)
    idx_i, idx_j = torch.triu_indices(32, 32, offset=1, device=h_seqs.device)
    # g_n(i, j) = 0.5 * (a_i * b_j - a_j * b_i)
    g_so32 = 0.5 * (a_tokens[:, idx_i] * b_tokens[:, idx_j] - a_tokens[:, idx_j] * b_tokens[:, idx_i]) # [N, 496]
    
    return g_so32.detach(), (loss.item() / tgt_flat.numel())

def analyze_covariance_spectrum(g_mat):
    """
    g_mat: [N, 496]
    Calcula a covariância empírica C in R^(496x496) e seu espectro de autovalores em float32.
    """
    N, D = g_mat.shape
    g_f = g_mat.float()
    mean_g = g_f.mean(dim=0, keepdim=True)
    g_centered = g_f - mean_g
    
    # Covariância C = (1 / (N - 1)) * g_centered.T @ g_centered in float32
    C = torch.matmul(g_centered.t(), g_centered) / (N - 1) # [496, 496]
    
    # Eigendecomposição simétrica
    eigenvals, eigenvecs = torch.linalg.eigh(C) # autovalores em ordem crescente
    eigenvals = torch.flip(eigenvals, dims=[0]) # ordem decrescente
    eigenvecs = torch.flip(eigenvecs, dims=[1]) # [496, 496]
    
    # Truncar ruídos numéricos negativos
    eigenvals = torch.clamp(eigenvals, min=0.0)
    total_var = eigenvals.sum().item()
    var_ratio = (eigenvals / (total_var + 1e-12)).cpu().numpy()
    cum_var = np.cumsum(var_ratio)
    
    # Calcular k90, k95, k99
    k90 = int(np.searchsorted(cum_var, 0.90)) + 1
    k95 = int(np.searchsorted(cum_var, 0.95)) + 1
    k99 = int(np.searchsorted(cum_var, 0.99)) + 1
    
    return {
        "eigenvals": eigenvals.cpu().numpy(),
        "eigenvecs": eigenvecs,
        "var_ratio": var_ratio,
        "cum_var": cum_var,
        "k90": k90,
        "k95": k95,
        "k99": k99,
        "total_var": total_var
    }

def compute_subspace_overlap(V1, V2, k=10):
    """
    Computa o overlap de Grassmann (ângulos canônicos) entre os top-k autovetores V1 e V2 in R^(496 x k).
    """
    V1_k = V1[:, :k].float()
    V2_k = V2[:, :k].float()
    
    # Matriz de projeção cruzada
    M = torch.matmul(V1_k.t(), V2_k) # [k, k]
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, 0.0, 1.0)
    cosines = S.cpu().numpy()
    mean_cos = float(np.mean(cosines))
    principal_angles_deg = np.arccos(cosines) * (180.0 / math.pi)
    
    return {
        "mean_cosine": mean_cos,
        "cosines": cosines.tolist(),
        "principal_angles_deg": principal_angles_deg.tolist()
    }

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("CICLO 20: ESPECTRO DE SENSIBILIDADE FUNCIONAL EM so(32) E DETERMINAÇÃO DA DIMENSÃO k")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Horário: {time.strftime('%H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()

    # 1. Carregar Componentes Base (lm_head_w, norm_mod, rotary)
    print("\n[Passo 1/5] Carregando componentes do Qwen e dados held-out...")
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=16)
    cfg = comp["cfg"]
    lm_head_w = comp["lm_head_w"]
    norm_mod = comp["norm_mod"]
    vocab_sz = lm_head_w.shape[0]

    # Carregar U_c(32)
    func_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    saved_func = torch.load(func_path, map_location=device)
    U_base = saved_func["U_f"].to(device=device, dtype=torch.bfloat16)

    # WikiText: carregar ativações de L63 e tokens
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds_wiki["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
    wiki_tokens = tokens_all[:32 * 64].view(32, 64).to(device)
    wiki_te_targets = wiki_tokens[16:24] # 8 seqs mantidas em teste

    # Obter h_63 para WikiText via Atlas forward
    h63_cache_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    h_in_63 = torch.load(h63_cache_path, map_location=device).to(device)

    atlas_bases_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    raw_bases = torch.load(atlas_bases_path, map_location=device)
    chart15_bases = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in raw_bases[15].items()}

    l63_mod = load_layer_module(63, cfg, device, snapshot_dir=snapshot_dir)
    import safetensors.torch as st
    lp = os.path.join(snapshot_dir, "layers-63.safetensors")
    with st.safe_open(lp, framework="pt") as f:
        mix_w = dequant(f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight"), f.get_tensor("model.language_model.layers.63.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        gate_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        up_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        down_w = dequant(f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight"), f.get_tensor("model.language_model.layers.63.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
        cw = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": True}
    p_w = project_layer_weights(cw, chart15_bases)
    l63_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
    l63_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
    l63_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
    l63_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

    pos_ids = torch.arange(64, device=device).view(1, 1, -1).expand(3, 32, -1)
    pos_emb = comp["rotary"](h_in_63, pos_ids)
    with torch.no_grad():
        out_s = l63_mod(h_in_63, position_embeddings=pos_emb)
        h_wiki_all = (out_s[0] if isinstance(out_s, tuple) else out_s).to(torch.bfloat16)
    del l63_mod
    torch.cuda.empty_cache()

    h_wiki_te = h_wiki_all[16:24] # 8 seqs (512 tokens)

    # HF Code (8 seqs de 64 tokens)
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_te = hf_data["h_in_63"][8:16, :64].to(device=device, dtype=torch.bfloat16)
    code_te_targets = hf_data["input_ids"][8:16, :64].to(device)

    # GSM8K (8 seqs de 64 tokens)
    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_te = gsm_data["h_in_63"][8:16, :64].to(device=device, dtype=torch.bfloat16)
    gsm_te_targets = gsm_data["input_ids"][8:16, :64].to(device)

    # 2. Carregar e Congelar W_opt
    p_data = torch.load("checkpoints/prof_targets_cache.pt", map_location=device)
    logits_prof_tr = p_data["logits_prof_tr"].to(device)
    prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
    y_tr_63 = prof_intermediates_tr[63].to(device).reshape(-1, 5120)
    h_wiki_tr = h_wiki_all[:16]
    wiki_tr_targets = wiki_tokens[:16]

    W_opt = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    opt_w = torch.optim.AdamW([W_opt], lr=1e-4, weight_decay=1e-3)
    corr_tr0 = torch.matmul(torch.matmul(h_wiki_tr, U_base), W_opt)
    l_tr0 = F.linear(norm_mod(h_wiki_tr + corr_tr0), lm_head_w)
    s_flat0 = l_tr0[:, :-1, :].reshape(-1, vocab_sz)
    t_flat0 = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
    y_flat0 = wiki_tr_targets[:, 1:].contiguous().reshape(-1)
    loss0 = F.kl_div(F.log_softmax(s_flat0.float(), dim=-1), F.softmax(t_flat0.float(), dim=-1), reduction="batchmean") + \
            0.1 * F.cross_entropy(s_flat0.float(), y_flat0, reduction="mean") + \
            0.05 * F.mse_loss((h_wiki_tr + corr_tr0).reshape(-1, 5120).float(), y_tr_63.float())
    loss0.backward()
    opt_w.step()

    W_opt.requires_grad = False
    U_base.requires_grad = False
    print("  [OK] U_c(32) e W_opt RIGOROSAMENTE CONGELADOS.")

    # 3. Etapa 20A: Extração da Distribuição de Gradientes em so(32)
    print("\n[Passo 2/5] Extraindo gradientes g_n in R^496 sobre 1.512 tokens held-out...")
    t_grad0 = time.time()
    g_wiki, ce_wiki = compute_so32_gradient_batch(h_wiki_te, wiki_te_targets, U_base, W_opt, norm_mod, lm_head_w)
    g_code, ce_code = compute_so32_gradient_batch(h_code_te, code_te_targets, U_base, W_opt, norm_mod, lm_head_w)
    g_gsm, ce_gsm = compute_so32_gradient_batch(h_gsm_te, gsm_te_targets, U_base, W_opt, norm_mod, lm_head_w)
    
    g_global = torch.cat([g_wiki, g_code, g_gsm], dim=0) # [N_total, 496]
    t_grad = time.time() - t_grad0
    print(f"  [OK] Gradientes extraídos em {t_grad*1000:.1f} ms! Total de tokens: N = {g_global.shape[0]}")
    print(f"       Wiki: {g_wiki.shape[0]} tokens | Code: {g_code.shape[0]} tokens | GSM: {g_gsm.shape[0]} tokens")

    # 4. Decomposição Espectral de Covariância em so(32)
    print("\n[Passo 3/5] Decompondo matrizes de covariância C in R^(496x496) e extraindo k_alpha...")
    spec_global = analyze_covariance_spectrum(g_global)
    spec_wiki = analyze_covariance_spectrum(g_wiki)
    spec_code = analyze_covariance_spectrum(g_code)
    spec_gsm = analyze_covariance_spectrum(g_gsm)

    print("=" * 80)
    print(f"{'Domínio':<15} | {'k_90 (90% var)':<18} | {'k_95 (95% var)':<18} | {'k_99 (99% var)':<18}")
    print("-" * 80)
    print(f"{'Global':<15} | {spec_global['k90']:<18} | {spec_global['k95']:<18} | {spec_global['k99']:<18}")
    print(f"{'WikiText':<15} | {spec_wiki['k90']:<18} | {spec_wiki['k95']:<18} | {spec_wiki['k99']:<18}")
    print(f"{'HF Code':<15} | {spec_code['k90']:<18} | {spec_code['k95']:<18} | {spec_code['k99']:<18}")
    print(f"{'GSM8K':<15} | {spec_gsm['k90']:<18} | {spec_gsm['k95']:<18} | {spec_gsm['k99']:<18}")
    print("=" * 80)

    # Análise de Overlap de Grassmann entre os subespaços de cada tarefa (top-10 modos)
    k_eval = 10
    overlap_wiki_code = compute_subspace_overlap(spec_wiki["eigenvecs"], spec_code["eigenvecs"], k=k_eval)
    overlap_wiki_gsm = compute_subspace_overlap(spec_wiki["eigenvecs"], spec_gsm["eigenvecs"], k=k_eval)
    overlap_code_gsm = compute_subspace_overlap(spec_code["eigenvecs"], spec_gsm["eigenvecs"], k=k_eval)

    print(f"\n  Overlap de Subespaços em so(32) (Top-{k_eval} Modos):")
    print(f"    Wiki vs Code: Cosseno Médio = {overlap_wiki_code['mean_cosine']:.4f}")
    print(f"    Wiki vs GSM:  Cosseno Médio = {overlap_wiki_gsm['mean_cosine']:.4f}")
    print(f"    Code vs GSM:  Cosseno Médio = {overlap_code_gsm['mean_cosine']:.4f}")

    # 5. Etapa 20B: Validação Causal dos Modos Dominantes vs Nulos em PPL
    print("\n[Passo 4/5] Validação Causal em PPL: Rotações exp(theta A) nos modos dominantes vs nulos...")
    idx_i, idx_j = torch.triu_indices(32, 32, offset=1, device=device)
    
    # Reconstruir os geradores A_m in so(32) para os top modos e 1 modo aleatório
    def get_generator_matrix(v):
        A = torch.zeros(32, 32, device=device, dtype=torch.bfloat16)
        A[idx_i, idx_j] = v.to(device=device, dtype=torch.bfloat16)
        A = A - A.t()
        # Normalizar por norma de Frobenius
        norm_f = torch.norm(A.float())
        if norm_f > 1e-6:
            A = A / norm_f
        return A

    A1 = get_generator_matrix(spec_global["eigenvecs"][:, 0]) # Modo 1
    A2 = get_generator_matrix(spec_global["eigenvecs"][:, 1]) # Modo 2
    A3 = get_generator_matrix(spec_global["eigenvecs"][:, 2]) # Modo 3
    
    # Gerador aleatório de controle nulo
    torch.manual_seed(42)
    rand_v = torch.randn(496, device=device)
    A_rand = get_generator_matrix(rand_v)

    # Avaliar curva de resposta angular theta in [-0.5, +0.5] rad em Code e GSM8K
    angles_rad = np.linspace(-0.5, 0.5, 21) # 21 pontos
    causal_results = {"angles_deg": (angles_rad * 180 / math.pi).tolist(), "modes": {}}

    test_modes = [("Modo 1 (Top 1)", A1), ("Modo 2 (Top 2)", A2), ("Modo 3 (Top 3)", A3), ("Modo Aleatório (Nulo)", A_rand)]
    
    for m_label, A_mat in test_modes:
        ppl_curve_code = []
        ppl_curve_gsm = []
        for th in angles_rad:
            # Q = exp(th * A) in SO(32)
            Q_mat = torch.linalg.matrix_exp(th * A_mat.float()).to(torch.bfloat16)
            
            with torch.no_grad():
                # Avaliar em Code
                a_c = torch.matmul(h_code_te, U_base)
                a_c_rot = torch.matmul(a_c, Q_mat)
                corr_c = torch.matmul(a_c_rot, W_opt)
                l_c = F.linear(norm_mod(h_code_te + corr_c), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
                ce_c = F.cross_entropy(l_c.float(), code_te_targets[:, 1:].contiguous().reshape(-1)).item()
                ppl_curve_code.append(math.exp(ce_c))
                
                # Avaliar em GSM
                a_g = torch.matmul(h_gsm_te, U_base)
                a_g_rot = torch.matmul(a_g, Q_mat)
                corr_g = torch.matmul(a_g_rot, W_opt)
                l_g = F.linear(norm_mod(h_gsm_te + corr_g), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
                ce_g = F.cross_entropy(l_g.float(), gsm_te_targets[:, 1:].contiguous().reshape(-1)).item()
                ppl_curve_gsm.append(math.exp(ce_g))
                
        causal_results["modes"][m_label] = {
            "code_ppl": ppl_curve_code,
            "gsm_ppl": ppl_curve_gsm,
            "best_code_ppl": min(ppl_curve_code),
            "best_gsm_ppl": min(ppl_curve_gsm),
            "code_gain": min(ppl_curve_code) - ppl_curve_code[10], # delta vs theta=0 (índice 10)
            "gsm_gain": min(ppl_curve_gsm) - ppl_curve_gsm[10]
        }
        print(f"    {m_label:<25} | Delta PPL Code: {causal_results['modes'][m_label]['code_gain']:+.2f} pts | Delta PPL GSM: {causal_results['modes'][m_label]['gsm_gain']:+.2f} pts")

    # 6. Salvar Resultados Numéricos em JSON
    results = {
        "k_dimensions": {
            "global": {"k90": spec_global["k90"], "k95": spec_global["k95"], "k99": spec_global["k99"]},
            "wiki": {"k90": spec_wiki["k90"], "k95": spec_wiki["k95"], "k99": spec_wiki["k99"]},
            "code": {"k90": spec_code["k90"], "k95": spec_code["k95"], "k99": spec_code["k99"]},
            "gsm": {"k90": spec_gsm["k90"], "k95": spec_gsm["k95"], "k99": spec_gsm["k99"]}
        },
        "variance_ratio_top20": spec_global["var_ratio"][:20].tolist(),
        "cumulative_variance_top20": spec_global["cum_var"][:20].tolist(),
        "overlap_top10": {
            "wiki_vs_code": overlap_wiki_code,
            "wiki_vs_gsm": overlap_wiki_gsm,
            "code_vs_gsm": overlap_code_gsm
        },
        "causal_modes": causal_results
    }

    res_path = os.path.join(repo_root, "experiments", "lie_algebra_spectrum_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [OK] Resultados salvos em: {res_path}")

    # 7. Geração das 3 Figuras Científicas em 300 DPI
    print("\n[Passo 5/5] Gerando figuras científicas...")

    # Figura 1: Espectro de Covariância em so(32) e Decaimento de Autovalores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Decaimento de Autovalores (Escala Logarítmica)
    ax = axes[0]
    ranks = np.arange(1, 101)
    ax.plot(ranks, spec_global["eigenvals"][:100], "b-", lw=2, label=f"Global (k90={spec_global['k90']})")
    ax.plot(ranks, spec_wiki["eigenvals"][:100], "g--", alpha=0.7, label=f"WikiText (k90={spec_wiki['k90']})")
    ax.plot(ranks, spec_code["eigenvals"][:100], "r--", alpha=0.7, label=f"HF Code (k90={spec_code['k90']})")
    ax.plot(ranks, spec_gsm["eigenvals"][:100], "m--", alpha=0.7, label=f"GSM8K (k90={spec_gsm['k90']})")
    ax.set_yscale("log")
    ax.set_xlabel(r"Índice do Modo Autovetor em $\mathfrak{so}(32)$ (1 a 100 de 496)", fontsize=11)
    ax.set_ylabel(r"Autovalor de Covariância $\lambda_i$ (log)", fontsize=11)
    ax.set_title(r"Espectro de Sensibilidade Funcional em $\mathfrak{so}(32)$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Subplot 2: Variância Acumulada e Cutoffs k_alpha
    ax = axes[1]
    ax.plot(ranks, spec_global["cum_var"][:100] * 100, "b-", lw=2.5, label="Global Acumulado")
    ax.plot(ranks, spec_code["cum_var"][:100] * 100, "r--", alpha=0.7, label="HF Code Acumulado")
    ax.plot(ranks, spec_gsm["cum_var"][:100] * 100, "m--", alpha=0.7, label="GSM8K Acumulado")
    ax.axhline(90, color="orange", linestyle=":", label=f"k90 = {spec_global['k90']} modos")
    ax.axhline(95, color="green", linestyle=":", label=f"k95 = {spec_global['k95']} modos")
    ax.axhline(99, color="purple", linestyle=":", label=f"k99 = {spec_global['k99']} modos")
    ax.set_xlabel(r"Número de Modos $k$", fontsize=11)
    ax.set_ylabel(r"Variância Explicada Acumulada (%)", fontsize=11)
    ax.set_title(rf"Determinação da Dimensão Funcional Efetiva ($k_{{90}}={spec_global['k90']}, k_{{95}}={spec_global['k95']}$)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    plt.suptitle(r"Ciclo 20: Espectro da Álgebra de Lie $\mathfrak{so}(32)$ sobre 1.512 Tokens Held-Out", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(repo_root, "experiments", "so32_covariance_spectrum.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 1 salva: {fig1_path}")

    # Figura 2: Matriz de Overlap e Ângulos de Grassmann entre Tarefas em so(32)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Heatmap de Cossenos Médios
    ax = axes[0]
    tasks = ["WikiText", "HF Code", "GSM8K"]
    overlap_matrix = np.array([
        [1.0, overlap_wiki_code["mean_cosine"], overlap_wiki_gsm["mean_cosine"]],
        [overlap_wiki_code["mean_cosine"], 1.0, overlap_code_gsm["mean_cosine"]],
        [overlap_wiki_gsm["mean_cosine"], overlap_code_gsm["mean_cosine"], 1.0]
    ])
    im = ax.imshow(overlap_matrix, cmap="YlGnBu", vmin=0, vmax=1.0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{overlap_matrix[i, j]:.3f}", ha="center", va="center", color="black" if overlap_matrix[i, j] < 0.7 else "white", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(tasks, fontsize=11, fontweight="bold")
    ax.set_yticklabels(tasks, fontsize=11, fontweight="bold")
    ax.set_title(rf"Overlap de Grassmann (Top-{k_eval} Modos em $\mathfrak{{so}}(32)$)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Gráfico dos 10 Ângulos Principais
    ax = axes[1]
    angles_idx = np.arange(1, k_eval + 1)
    ax.plot(angles_idx, overlap_wiki_code["principal_angles_deg"], "ro-", lw=1.5, label="Wiki vs Code")
    ax.plot(angles_idx, overlap_wiki_gsm["principal_angles_deg"], "go-", lw=1.5, label="Wiki vs GSM")
    ax.plot(angles_idx, overlap_code_gsm["principal_angles_deg"], "mo-", lw=1.5, label="Code vs GSM")
    ax.set_xlabel(r"Índice do Ângulo Principal de Grassmann", fontsize=11)
    ax.set_ylabel(r"Ângulo Canônico (graus)", fontsize=11)
    ax.set_title(rf"Ângulos Principais entre Tarefas em $\mathbb{{R}}^{{496}}$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle(r"Ciclo 20: Alinhamento e Overlap dos Subespaços Funcionais em $\mathfrak{so}(32)$", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = os.path.join(repo_root, "experiments", "so32_domain_overlap_matrix.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 2 salva: {fig2_path}")

    # Figura 3: Validação Causal em PPL (Modos Dominantes vs Aleatório)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    angles_deg = np.array(causal_results["angles_deg"])

    # Code PPL
    ax = axes[0]
    for m_label in causal_results["modes"]:
        curve = causal_results["modes"][m_label]["code_ppl"]
        style = "k--" if "Aleatório" in m_label else "-"
        width = 1.5 if "Aleatório" in m_label else 2.0
        ax.plot(angles_deg, curve, style, lw=width, label=f"{m_label} (gain: {causal_results['modes'][m_label]['code_gain']:+.2f})")
    ax.set_xlabel(r"Ângulo de Rotação $\theta$ no Modo (graus)", fontsize=11)
    ax.set_ylabel(r"PPL HF Code (micro-batch)", fontsize=11)
    ax.set_title(r"Resposta Causal em HF Code", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    # GSM8K PPL
    ax = axes[1]
    for m_label in causal_results["modes"]:
        curve = causal_results["modes"][m_label]["gsm_ppl"]
        style = "k--" if "Aleatório" in m_label else "-"
        width = 1.5 if "Aleatório" in m_label else 2.0
        ax.plot(angles_deg, curve, style, lw=width, label=f"{m_label} (gain: {causal_results['modes'][m_label]['gsm_gain']:+.2f})")
    ax.set_xlabel(r"Ângulo de Rotação $\theta$ no Modo (graus)", fontsize=11)
    ax.set_ylabel(r"PPL GSM8K (micro-batch)", fontsize=11)
    ax.set_title(r"Resposta Causal em GSM8K", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    plt.suptitle(r"Ciclo 20: Validação Causal de PPL com $Q(\theta) = \exp(\theta A)$ em $\text{SO}(32)$", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(repo_root, "experiments", "so32_causal_modes_ppl.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 3 salva: {fig3_path}")

    total_time = time.time() - t_start
    print(f"\n[SUCESSO] Ciclo 20 finalizado em {total_time:.2f} segundos!")
    print("=" * 115)

if __name__ == "__main__":
    main()
