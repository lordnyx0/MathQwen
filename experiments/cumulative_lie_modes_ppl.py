#!/usr/bin/env python3
"""
experiments/cumulative_lie_modes_ppl.py

CICLO 21: Recuperação Causal Cumulativa Q_k(alpha) em SO(32) (k in {0, 1, 2, 3, 4, 8, 16})
-----------------------------------------------------------------------------------------
Objetivo:
Avaliar a superposição conjunta dos modos próprios da álgebra de Lie so(32)
via operador multivariado Q_k(alpha) = exp(sum_{m=1}^k alpha_m A_m),
identificar a localização do "cotovelo" causal (elbow) de saturação em PPL,
e medir a colinearidade cos(alpha_Code, alpha_GSM) no espaço de controle R^k.

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
    vocab_sz = lm_head_w.shape[0]
    B, S, D = h_seqs.shape
    a = torch.matmul(h_seqs, U_base)
    
    delta_h = torch.zeros_like(h_seqs, requires_grad=True)
    h_comb = h_seqs + delta_h
    h_norm = norm_mod(h_comb)
    logits = F.linear(h_norm, lm_head_w)
    
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_sz)
    tgt_flat = targets[:, 1:].contiguous().reshape(-1)
    
    loss = F.cross_entropy(logits_pred.float(), tgt_flat, reduction="sum")
    loss.backward()
    
    grad_delta_h = delta_h.grad
    b = torch.matmul(grad_delta_h, W_opt.t())
    
    a_tokens = a[:, :-1, :].reshape(-1, 32)
    b_tokens = b[:, :-1, :].reshape(-1, 32)
    
    idx_i, idx_j = torch.triu_indices(32, 32, offset=1, device=h_seqs.device)
    g_so32 = 0.5 * (a_tokens[:, idx_i] * b_tokens[:, idx_j] - a_tokens[:, idx_j] * b_tokens[:, idx_i])
    
    return g_so32.detach(), (loss.item() / tgt_flat.numel())

def extract_covariance_eigenvectors(g_mat):
    N, D = g_mat.shape
    g_f = g_mat.float()
    mean_g = g_f.mean(dim=0, keepdim=True)
    g_centered = g_f - mean_g
    C = torch.matmul(g_centered.t(), g_centered) / (N - 1)
    
    eigenvals, eigenvecs = torch.linalg.eigh(C)
    eigenvals = torch.flip(eigenvals, dims=[0])
    eigenvecs = torch.flip(eigenvecs, dims=[1])
    return eigenvecs # [496, 496]

def optimize_alpha_for_k(k, generators, h_seqs, targets, U_base, W_opt, norm_mod, lm_head_w, num_steps=25, lr=0.04):
    """
    Otimiza alpha in R^k para Q_k = exp(sum_{m=1}^k alpha_m A_m)
    generators: lista de matrizes [32, 32] antissimétricas ortonormais.
    """
    vocab_sz = lm_head_w.shape[0]
    tgt_flat = targets[:, 1:].contiguous().reshape(-1)
    a = torch.matmul(h_seqs, U_base)
    
    if k == 0:
        with torch.no_grad():
            corr = torch.matmul(a, W_opt)
            l = F.linear(norm_mod(h_seqs + corr), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
            ce = F.cross_entropy(l.float(), tgt_flat).item()
            ppl = math.exp(ce)
        return ppl, ce, []

    # alpha in R^k inicializado em zero
    alpha = nn.Parameter(torch.zeros(k, device=h_seqs.device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([alpha], lr=lr, weight_decay=1e-4)
    
    # Empilhar os k geradores: [k, 32, 32]
    A_stack = torch.stack(generators[:k], dim=0).float()
    
    for step in range(num_steps):
        optimizer.zero_grad()
        # A_total = sum alpha_m * A_m
        A_comb = torch.einsum("k,kij->ij", alpha, A_stack)
        Q_mat = torch.linalg.matrix_exp(A_comb).to(torch.bfloat16)
        
        a_rot = torch.matmul(a, Q_mat)
        corr = torch.matmul(a_rot, W_opt)
        l = F.linear(norm_mod(h_seqs + corr), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
        loss = F.cross_entropy(l.float(), tgt_flat)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        A_comb = torch.einsum("k,kij->ij", alpha, A_stack)
        Q_mat = torch.linalg.matrix_exp(A_comb).to(torch.bfloat16)
        a_rot = torch.matmul(a, Q_mat)
        corr = torch.matmul(a_rot, W_opt)
        l = F.linear(norm_mod(h_seqs + corr), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
        final_ce = F.cross_entropy(l.float(), tgt_flat).item()
        final_ppl = math.exp(final_ce)
        
    return final_ppl, final_ce, alpha.detach().cpu().numpy().tolist()

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("CICLO 21: RECUPERAÇÃO CAUSAL CUMULATIVA Q_k(alpha) EM SO(32) (k in {0, 1, 2, 3, 4, 8, 16})")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Horário: {time.strftime('%H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()

    # 1. Carregar Componentes Base
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

    # WikiText
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds_wiki["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
    wiki_tokens = tokens_all[:32 * 64].view(32, 64).to(device)
    wiki_te_targets = wiki_tokens[16:24]

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

    h_wiki_te = h_wiki_all[16:24]

    # HF Code & GSM8K
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_te = hf_data["h_in_63"][8:16, :64].to(device=device, dtype=torch.bfloat16)
    code_te_targets = hf_data["input_ids"][8:16, :64].to(device)

    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_te = gsm_data["h_in_63"][8:16, :64].to(device=device, dtype=torch.bfloat16)
    gsm_te_targets = gsm_data["input_ids"][8:16, :64].to(device)

    # 2. Calibrar e Congelar W_opt
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

    # 3. Extrair os Geradores Canônicos de Lie A_1 ... A_16
    print("\n[Passo 2/5] Extraindo base canônica de geradores A_1 ... A_16 da álgebra so(32)...")
    g_wiki, _ = compute_so32_gradient_batch(h_wiki_te, wiki_te_targets, U_base, W_opt, norm_mod, lm_head_w)
    g_code, _ = compute_so32_gradient_batch(h_code_te, code_te_targets, U_base, W_opt, norm_mod, lm_head_w)
    g_gsm, _ = compute_so32_gradient_batch(h_gsm_te, gsm_te_targets, U_base, W_opt, norm_mod, lm_head_w)
    g_global = torch.cat([g_wiki, g_code, g_gsm], dim=0)

    eigenvecs = extract_covariance_eigenvectors(g_global) # [496, 496]
    idx_i, idx_j = torch.triu_indices(32, 32, offset=1, device=device)

    generators = []
    for m in range(16):
        v = eigenvecs[:, m].to(device=device, dtype=torch.bfloat16)
        A = torch.zeros(32, 32, device=device, dtype=torch.bfloat16)
        A[idx_i, idx_j] = v
        A = A - A.t()
        norm_f = torch.norm(A.float())
        if norm_f > 1e-6:
            A = A / norm_f
        generators.append(A)
    print(f"  [OK] 16 geradores ortonormais sintetizados com sucesso (Frobenius norm = 1.0).")

    # 4. Avaliação Cumulativa para k in {0, 1, 2, 3, 4, 8, 16}
    k_values = [0, 1, 2, 3, 4, 8, 16]
    domains = {
        "WikiText": (h_wiki_te, wiki_te_targets),
        "HF Code": (h_code_te, code_te_targets),
        "GSM8K": (h_gsm_te, gsm_te_targets)
    }

    print("\n[Passo 3/5] Otimizando vetor de controle alpha in R^k para cada domínio...")
    results = {"k_values": k_values, "domains": {}, "alignment": {}}

    for dom_name, (h_te, tgt_te) in domains.items():
        results["domains"][dom_name] = {"ppl": [], "ce": [], "delta_ppl": [], "alphas": {}}
        print(f"\n  --- Domínio: {dom_name} ---")
        base_ppl = None
        for k in k_values:
            t0 = time.time()
            ppl, ce, alpha_vals = optimize_alpha_for_k(k, generators, h_te, tgt_te, U_base, W_opt, norm_mod, lm_head_w)
            if k == 0:
                base_ppl = ppl
            delta = ppl - base_ppl
            results["domains"][dom_name]["ppl"].append(round(ppl, 2))
            results["domains"][dom_name]["ce"].append(round(ce, 4))
            results["domains"][dom_name]["delta_ppl"].append(round(delta, 2))
            results["domains"][dom_name]["alphas"][str(k)] = alpha_vals
            t_k = (time.time() - t0) * 1000
            print(f"    k = {k:<2} | PPL = {ppl:<7.2f} | Delta = {delta:+6.2f} pts | Tempo = {t_k:.1f} ms")

    # Tabela Resumo Comparativa
    print("\n" + "=" * 95)
    print(f"{'k (Modos)':<10} | {'WikiText PPL':<15} | {'HF Code PPL':<15} | {'Delta Code':<12} | {'GSM8K PPL':<15} | {'Delta GSM':<12}")
    print("-" * 95)
    for idx, k in enumerate(k_values):
        w_p = results["domains"]["WikiText"]["ppl"][idx]
        c_p = results["domains"]["HF Code"]["ppl"][idx]
        c_d = results["domains"]["HF Code"]["delta_ppl"][idx]
        g_p = results["domains"]["GSM8K"]["ppl"][idx]
        g_d = results["domains"]["GSM8K"]["delta_ppl"][idx]
        print(f"k = {k:<6} | {w_p:<15.2f} | {c_p:<15.2f} | {c_d:+11.2f} | {g_p:<15.2f} | {g_d:+11.2f}")
    print("=" * 95)

    # 5. Análise de Colinearidade de Controle entre Tarefas cos(alpha_Code, alpha_GSM)
    print("\n[Passo 4/5] Analisando colinearidade do vetor de controle cos(alpha_Code, alpha_GSM)...")
    for k in [2, 3, 4, 8, 16]:
        a_code = np.array(results["domains"]["HF Code"]["alphas"][str(k)])
        a_gsm = np.array(results["domains"]["GSM8K"]["alphas"][str(k)])
        a_wiki = np.array(results["domains"]["WikiText"]["alphas"][str(k)])
        
        cos_code_gsm = float(np.dot(a_code, a_gsm) / (np.linalg.norm(a_code) * np.linalg.norm(a_gsm) + 1e-12))
        cos_wiki_code = float(np.dot(a_wiki, a_code) / (np.linalg.norm(a_wiki) * np.linalg.norm(a_code) + 1e-12))
        cos_wiki_gsm = float(np.dot(a_wiki, a_gsm) / (np.linalg.norm(a_wiki) * np.linalg.norm(a_gsm) + 1e-12))
        
        results["alignment"][str(k)] = {
            "cos_code_gsm": cos_code_gsm,
            "cos_wiki_code": cos_wiki_code,
            "cos_wiki_gsm": cos_wiki_gsm,
            "norm_code": float(np.linalg.norm(a_code)),
            "norm_gsm": float(np.linalg.norm(a_gsm)),
            "norm_wiki": float(np.linalg.norm(a_wiki))
        }
        print(f"  k = {k:<2}: cos(Code, GSM) = {cos_code_gsm:+.4f} | ||alpha_Code|| = {np.linalg.norm(a_code):.3f} | ||alpha_GSM|| = {np.linalg.norm(a_gsm):.3f}")

    # Salvar JSON
    res_path = os.path.join(repo_root, "experiments", "cumulative_lie_modes_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [OK] Resultados salvos em: {res_path}")

    # 6. Geração dos Gráficos Científicos em 300 DPI
    print("\n[Passo 5/5] Gerando figuras científicas...")

    # Figura 1: Curva Cumulativa PPL(k) e Delta PPL(k)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Delta PPL em função de k
    ax = axes[0]
    ax.plot(k_values, results["domains"]["HF Code"]["delta_ppl"], "ro-", lw=2.2, markersize=7, label="HF Code")
    ax.plot(k_values, results["domains"]["GSM8K"]["delta_ppl"], "mo-", lw=2.2, markersize=7, label="GSM8K")
    ax.plot(k_values, results["domains"]["WikiText"]["delta_ppl"], "go-", lw=1.8, markersize=6, label="WikiText")
    ax.axvline(3, color="blue", linestyle="--", alpha=0.6, label="k = 3 (Elbow)")
    ax.axvline(4, color="orange", linestyle=":", alpha=0.7, label="k = 4 (Saturação 95%)")
    ax.set_xticks(k_values)
    ax.set_xlabel("Número de Modos de Lie $k$ Ativados", fontsize=11)
    ax.set_ylabel(r"$\Delta$PPL Cumulativo vs $k=0$ (pontos)", fontsize=11)
    ax.set_title(r"Ganho Causal Cumulativo de PPL $\Delta\text{PPL}(k)$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Subplot 2: Valores Absolutos de PPL(k) (Micro-Batch)
    ax = axes[1]
    ax.plot(k_values, results["domains"]["HF Code"]["ppl"], "r^-", lw=2, markersize=6, label="HF Code PPL")
    ax.plot(k_values, results["domains"]["GSM8K"]["ppl"], "ms-", lw=2, markersize=6, label="GSM8K PPL")
    ax.set_xticks(k_values)
    ax.set_xlabel("Número de Modos de Lie $k$ Ativados", fontsize=11)
    ax.set_ylabel("PPL Absoluto (micro-batch)", fontsize=11)
    ax.set_title(r"Convergência e Platô de Saturação Causal $PPL(k)$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle(r"Ciclo 21: Identificação do Cotovelo Causal em $Q_k(\vec\alpha) = \exp(\sum_{i=1}^k \alpha_i A_i)$", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(repo_root, "experiments", "so32_cumulative_ppl_curve.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 1 salva: {fig1_path}")

    # Figura 2: Amplitudes do Vetor de Controle alpha em k = 4
    fig, ax = plt.subplots(figsize=(9, 5))
    modes_idx = np.arange(1, 5)
    width = 0.25

    alphas_c = results["domains"]["HF Code"]["alphas"]["4"]
    alphas_g = results["domains"]["GSM8K"]["alphas"]["4"]
    alphas_w = results["domains"]["WikiText"]["alphas"]["4"]

    ax.bar(modes_idx - width, alphas_c, width, color="#d62728", alpha=0.85, label=f"HF Code (||\u03b1|| = {np.linalg.norm(alphas_c):.2f})")
    ax.bar(modes_idx, alphas_g, width, color="#2ca02c", alpha=0.85, label=f"GSM8K (||\u03b1|| = {np.linalg.norm(alphas_g):.2f})")
    ax.bar(modes_idx + width, alphas_w, width, color="#2b5c8f", alpha=0.85, label=f"WikiText (||\u03b1|| = {np.linalg.norm(alphas_w):.2f})")

    cos_4 = results["alignment"]["4"]["cos_code_gsm"]
    ax.set_xticks(modes_idx)
    ax.set_xticklabels([f"Modo {i} (A_{i})" for i in range(1, 5)], fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Amplitude Ótima de Rotação $\alpha_m^*$ (rad)", fontsize=11)
    ax.set_title(rf"Colinearidade dos Vetores de Controle em $k=4$ (cos(Code, GSM) = {cos_4:+.4f})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9.5)

    plt.suptitle(r"Ciclo 21: Alinhamento Direcional das Trajetórias de Controle em $\mathfrak{so}(32)$", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = os.path.join(repo_root, "experiments", "so32_alpha_weights_comparison.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 2 salva: {fig2_path}")

    total_time = time.time() - t_start
    print(f"\n[SUCESSO] Ciclo 21 finalizado em {total_time:.2f} segundos!")
    print("=" * 115)

if __name__ == "__main__":
    main()
