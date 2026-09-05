# -*- coding: utf-8 -*-
"""
TESTE DE CAUSALIDADE POR ABLACAO DIRECIONAL E GENERALIZACAO MULTIDOMINIO EM L63
- Adição progressiva top-k modos funcionais: k in {0, 1, 2, 4, 8, 16, 24, 32} -> PPL(k), Top1(k), KL(k).
- Remoção progressiva das direções de maior alavancagem: sem u4, sem {u4, u10}, etc.
- Avaliação de generalização em WikiText, HF Code e GSM8K.
- Gera gráfico functional_directional_ablation.png e salva experiments/functional_directional_ablation.json.
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
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

from reference.loader import load_qwen_reference_components, load_layer_module, dequant, get_snapshot_dir
from atlas.projection import project_layer_weights

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor = None):
    vocab_size = logits.shape[-1]
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    targets_flat = targets.reshape(-1)
    nll = F.cross_entropy(logits_pred, targets_flat, reduction="mean").item()
    ppl = math.exp(nll)
    top1 = (logits_pred.argmax(dim=-1) == targets_flat).float().mean().item() * 100.0

    cos_sim = 0.0
    kl_div = 0.0
    if teacher_logits is not None:
        t_pred = teacher_logits[:, :-1, :].reshape(-1, vocab_size).float()
        cos_sim = F.cosine_similarity(logits_pred, t_pred, dim=-1).mean().item()
        p_t = F.softmax(t_pred, dim=-1)
        log_p_s = F.log_softmax(logits_pred, dim=-1)
        kl_div = F.kl_div(log_p_s, p_t, reduction="batchmean").item()

    return {
        "nll": nll,
        "ppl": ppl,
        "top1": top1,
        "cos_sim": cos_sim,
        "kl_div": kl_div
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("TESTE DE CAUSALIDADE POR ABLACAO DIRECIONAL E GENERALIZACAO MULTIDOMINIO EM L63")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Componentes e Dados de Referência
    snapshot_dir = get_snapshot_dir()
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_train = 16
    num_test = 16
    total_seqs = num_train + num_test
    seq_len = 64

    input_ids = tokens_all[:total_seqs * seq_len].view(total_seqs, seq_len).to(device)
    train_ids = input_ids[:num_train]
    test_ids = input_ids[num_train:]
    train_targets = train_ids[:, 1:].contiguous()
    test_targets = test_ids[:, 1:].contiguous()

    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=total_seqs)
    cfg = comp["cfg"]
    lm_head_w = comp["lm_head_w"]
    norm_mod = comp["norm_mod"]
    vocab_sz = lm_head_w.shape[0]

    p_data = torch.load("checkpoints/prof_targets_cache.pt", map_location=device)
    logits_prof_tr = p_data["logits_prof_tr"].to(device)
    logits_prof_test = p_data["logits_prof_test"].to(device)
    prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
    y_tr_63 = prof_intermediates_tr[63].to(device).reshape(-1, 5120)

    # Entrada de L63 e Forward do Atlas
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

    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, total_seqs, -1)
    pos_emb = comp["rotary"](h_in_63, pos_ids)
    with torch.no_grad():
        out_s = l63_mod(h_in_63, position_embeddings=pos_emb)
        h_63_raw = (out_s[0] if isinstance(out_s, tuple) else out_s).to(torch.bfloat16)

    del l63_mod
    torch.cuda.empty_cache()

    # 2. Carregar a Base Funcional U_f(32)
    func_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    if not os.path.exists(func_path):
        raise FileNotFoundError(f"Arquivo {func_path} nao encontrado!")
    saved_func = torch.load(func_path, map_location=device)
    U_f_32 = saved_func["U_f"].to(device=device, dtype=torch.bfloat16) # (5120, 32)
    print(f"  -> U_f(32) carregado com sucesso: shape = {U_f_32.shape}")

    # 3. Adição Cumulativa: k in {0, 1, 2, 4, 8, 16, 24, 32}
    k_eval_list = [0, 1, 2, 4, 8, 16, 24, 32]
    cumulative_results = []

    print("\n" + "=" * 110)
    print("TESTE DE ABLACAO 1: ADICAO CUMULATIVA DOS TOP-k MODOS FUNCIONAIS")
    print("=" * 110)
    print(f"{'k':<6} | {'Rank rf':<10} | {'Parâmetros':<12} | {'Test PPL':<10} | {'Delta vs SVD-256':<18} | {'Top-1 Match':<12} | {'KL Div':<8}")
    print("-" * 110)

    for k in k_eval_list:
        if k == 0:
            with torch.no_grad():
                l_te = F.linear(norm_mod(h_63_raw[num_train:]), lm_head_w)
                m_k = compute_metrics(l_te, test_targets, teacher_logits=logits_prof_test)
            p_cnt = 0
        else:
            U_k = U_f_32[:, :k].contiguous()
            W_k = nn.Parameter(torch.zeros(k, 5120, device=device, dtype=torch.bfloat16))
            p_cnt = W_k.numel()
            opt = torch.optim.AdamW([W_k], lr=1e-4, weight_decay=1e-3)

            # 1 passo de otimização terminal (ponto ótimo de generalização observado)
            h_tr = h_63_raw[:num_train]
            corr_tr = torch.matmul(torch.matmul(h_tr, U_k), W_k)
            l_tr = F.linear(norm_mod(h_tr + corr_tr), lm_head_w)
            s_flat = l_tr[:, :-1, :].reshape(-1, vocab_sz)
            t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
            y_flat = train_targets.reshape(-1)

            loss_kl = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean")
            loss_ce = F.cross_entropy(s_flat.float(), y_flat, reduction="mean")
            loss_mse = F.mse_loss((h_tr + corr_tr).reshape(-1, 5120).float(), y_tr_63.float())
            loss = loss_kl + 0.1 * loss_ce + 0.05 * loss_mse
            loss.backward()
            opt.step()

            with torch.no_grad():
                corr_te = torch.matmul(torch.matmul(h_63_raw[num_train:], U_k), W_k)
                l_te = F.linear(norm_mod(h_63_raw[num_train:] + corr_te), lm_head_w)
                m_k = compute_metrics(l_te, test_targets, teacher_logits=logits_prof_test)

        delta_ppl = m_k["ppl"] - 192.68
        cumulative_results.append({
            "k": k,
            "params": p_cnt,
            "ppl": m_k["ppl"],
            "top1": m_k["top1"],
            "kl": m_k["kl_div"],
            "delta_ppl": delta_ppl
        })
        print(f"{k:<6} | {k:<10} | {p_cnt:<12,} | {m_k['ppl']:<10.2f} | {delta_ppl:+6.2f} pts{'':<10} | {m_k['top1']:<11.2f}% | {m_k['kl_div']:<8.4f}")
    print("-" * 110)

    # 4. Teste de Remoção Progressiva dos Vetores de Maior Alavancagem
    # Do Experimento F: vetores com maior ratio são u_4, u_10, u_5, u_3, u_17
    print("\n" + "=" * 110)
    print("TESTE DE ABLACAO 2: REMOCAO PROGRESSIVA DAS DIRECOES DE MAIOR ALAVANCAGEM")
    print("=" * 110)
    removal_configs = [
        {"name": "Base Completa U_f(32)", "removed_indices": []},
        {"name": "Sem Top-1 Alavancagem (sem u_4)", "removed_indices": [3]},
        {"name": "Sem Top-2 Alavancagem (sem u_4, u_10)", "removed_indices": [3, 9]},
        {"name": "Sem Top-4 Alavancagem (sem u_4, u_10, u_5, u_3)", "removed_indices": [3, 9, 4, 2]}
    ]

    removal_results = []
    print(f"{'Configuracao':<35} | {'Canais':<8} | {'Test PPL':<10} | {'Delta vs Base':<16} | {'Top-1 Match':<12}")
    print("-" * 110)

    base_ppl = cumulative_results[-1]["ppl"]
    for rem in removal_configs:
        keep_indices = [i for i in range(32) if i not in rem["removed_indices"]]
        U_sub = U_f_32[:, keep_indices].contiguous()
        k_sub = len(keep_indices)

        W_sub = nn.Parameter(torch.zeros(k_sub, 5120, device=device, dtype=torch.bfloat16))
        opt_sub = torch.optim.AdamW([W_sub], lr=1e-4, weight_decay=1e-3)

        h_tr = h_63_raw[:num_train]
        corr_tr = torch.matmul(torch.matmul(h_tr, U_sub), W_sub)
        l_tr = F.linear(norm_mod(h_tr + corr_tr), lm_head_w)
        s_flat = l_tr[:, :-1, :].reshape(-1, vocab_sz)
        t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
        y_flat = train_targets.reshape(-1)

        loss = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean") + \
               0.1 * F.cross_entropy(s_flat.float(), y_flat, reduction="mean") + \
               0.05 * F.mse_loss((h_tr + corr_tr).reshape(-1, 5120).float(), y_tr_63.float())
        loss.backward()
        opt_sub.step()

        with torch.no_grad():
            corr_te = torch.matmul(torch.matmul(h_63_raw[num_train:], U_sub), W_sub)
            l_te = F.linear(norm_mod(h_63_raw[num_train:] + corr_te), lm_head_w)
            m_rem = compute_metrics(l_te, test_targets, teacher_logits=logits_prof_test)

        delta_base = m_rem["ppl"] - base_ppl
        removal_results.append({
            "name": rem["name"],
            "k": k_sub,
            "ppl": m_rem["ppl"],
            "top1": m_rem["top1"],
            "kl": m_rem["kl_div"],
            "delta_base": delta_base
        })
        print(f"{rem['name']:<35} | {k_sub:<8} | {m_rem['ppl']:<10.2f} | {delta_base:+6.2f} pts{'':<8} | {m_rem['top1']:<11.2f}%")
    print("-" * 110)

    # 5. Avaliação de Generalização Multidomínio (WikiText, HF Code, GSM8K)
    print("\n" + "=" * 110)
    print("AVALIACAO DE GENERALIZACAO MULTIDOMINIO (WIKITEXT / HF CODE / GSM8K)")
    print("=" * 110)

    # Carregar dados de teste de Code e GSM8K
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_in = hf_data["h_in_63"].to(device=device, dtype=torch.bfloat16)
    code_ids = hf_data["input_ids"].to(device)
    code_targets = code_ids[:, 1:].contiguous()

    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_in = gsm_data["h_in_63"].to(device=device, dtype=torch.bfloat16)
    gsm_ids = gsm_data["input_ids"].to(device)
    gsm_targets = gsm_ids[:, 1:].contiguous()

    # Usar o modelo treinado com U_f(32)
    U_32 = U_f_32
    W_32 = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    opt_32 = torch.optim.AdamW([W_32], lr=1e-4, weight_decay=1e-3)
    corr_tr = torch.matmul(torch.matmul(h_tr, U_32), W_32)
    l_tr = F.linear(norm_mod(h_tr + corr_tr), lm_head_w)
    s_flat = l_tr[:, :-1, :].reshape(-1, vocab_sz)
    t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
    y_flat = train_targets.reshape(-1)
    loss = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean") + \
           0.1 * F.cross_entropy(s_flat.float(), y_flat, reduction="mean") + \
           0.05 * F.mse_loss((h_tr + corr_tr).reshape(-1, 5120).float(), y_tr_63.float())
    loss.backward()
    opt_32.step()

    gen_results = {}
    domains_eval = [
        ("WikiText (Test)", h_63_raw[num_train:], test_targets),
        ("HF Code", h_code_in, code_targets),
        ("GSM8K", h_gsm_in, gsm_targets)
    ]

    print(f"{'Domínio':<20} | {'Atlas Raw PPL':<15} | {'U_f(32) PPL':<15} | {'Ganho Delta':<15} | {'Top-1 U_f(32)':<15}")
    print("-" * 110)

    for d_name, h_eval_d, targets_d in domains_eval:
        with torch.no_grad():
            # Raw
            l_raw_d = F.linear(norm_mod(h_eval_d), lm_head_w)
            m_raw_d = compute_metrics(l_raw_d, targets_d)
            # U_f(32)
            corr_d = torch.matmul(torch.matmul(h_eval_d, U_32), W_32)
            l_uf_d = F.linear(norm_mod(h_eval_d + corr_d), lm_head_w)
            m_uf_d = compute_metrics(l_uf_d, targets_d)

        delta_d = m_uf_d["ppl"] - m_raw_d["ppl"]
        gen_results[d_name] = {
            "raw_ppl": m_raw_d["ppl"],
            "uf32_ppl": m_uf_d["ppl"],
            "delta_ppl": delta_d,
            "top1": m_uf_d["top1"]
        }
        print(f"{d_name:<20} | {m_raw_d['ppl']:<15.2f} | {m_uf_d['ppl']:<15.2f} | {delta_d:+6.2f} pts{'':<7} | {m_uf_d['top1']:<14.2f}%")
    print("-" * 110)

    # 6. Salvar Gráfico: functional_directional_ablation.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle("Teste de Causalidade por Ablação Direcional em L63", fontsize=14, fontweight="bold")

    # Plot 1: Curva PPL vs k modos funcionais
    ax1 = axes[0]
    k_plot = [r["k"] for r in cumulative_results]
    ppl_plot = [r["ppl"] for r in cumulative_results]
    top1_plot = [r["top1"] for r in cumulative_results]

    color1 = "tab:blue"
    ax1.plot(k_plot, ppl_plot, marker="o", color=color1, lw=2.5, label="Test PPL")
    ax1.axhline(192.68, color="gray", linestyle="--", label="Baseline SVD-256 (192.68)")
    ax1.axhline(179.94, color="gold", linestyle=":", label="Recorde L63 Mono (179.94)")
    ax1.set_xlabel("Número de Modos Funcionais Adicionados (k)", fontweight="bold")
    ax1.set_ylabel("Test Perplexity (PPL)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Curva Causal: PPL vs Dimensão Funcional k", fontweight="bold")

    ax1_twin = ax1.twinx()
    color2 = "tab:green"
    ax1_twin.plot(k_plot, top1_plot, marker="s", color=color2, lw=2, linestyle="--", label="Top-1 Match (%)")
    ax1_twin.set_ylabel("Top-1 Match (%)", color=color2, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.legend(loc="upper right")

    # Plot 2: Degradação de PPL por Remoção de Vetores Críticos
    ax2 = axes[1]
    rem_names = [r["name"].replace(" (sem ", "\n(sem ") for r in removal_results]
    rem_ppls = [r["ppl"] for r in removal_results]
    colors_rem = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]
    bars = ax2.bar(range(len(rem_names)), rem_ppls, color=colors_rem, alpha=0.85, edgecolor="k")
    ax2.set_xticks(range(len(rem_names)))
    ax2.set_xticklabels(rem_names, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Test Perplexity (PPL)", fontweight="bold")
    ax2.set_title("Degradação Causal: Remoção de Direções de Alta Alavancagem", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    for bar, r in zip(bars, removal_results):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{r['ppl']:.2f}\n({r['delta_base']:+4.2f})", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(repo_root, "experiments", "functional_directional_ablation.png")
    fig.savefig(chart_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "functional_directional_ablation.png"), dpi=300)
    plt.close(fig)
    print(f"\n[OK] Gráfico salvo em: {chart_path} e raiz do repositório.")

    # 7. Salvar Resultados JSON
    results_json = {
        "cumulative_addition": cumulative_results,
        "removal_ablation": removal_results,
        "generalization_multidomain": gen_results
    }
    json_path = os.path.join(repo_root, "experiments", "functional_directional_ablation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"[OK] Resultados salvos em: {json_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
