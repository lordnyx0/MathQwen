#!/usr/bin/env python3
"""
experiments/multivariate_lie_gating.py

CICLO 22: Micro-Gate Multivariado Dinâmico alpha(x) em SO(32) e Generalização Causal
-------------------------------------------------------------------------------------
Objetivo:
Avaliar se uma política dinâmica multivariada de reorientação em SO(32):
    alpha(x) = alpha_0 + Delta_alpha * tanh(W_alpha @ a(x) + b_alpha) in R^k
    Q(x) = exp(sum_{m=1}^k alpha_m(x) A_m) in SO(32)
treinada unicamente sobre o estado comprimido a = h @ U_c in R^32 (66, 132 ou 264 params)
consegue rotear autonomamente diferentes tarefas (WikiText, HF Code, GSM8K) em sequências
estritamente mantidas em teste (held-out), comparando com:
  - Modelo A: Q = I (0 params, baseline de identidade)
  - Modelo B: Q = exp(sum alpha_m^fixo A_m) (Controle estático do Ciclo 21)
  - Modelo C-2: Micro-Gate com k=2 (66 params: 2x32 + 2)
  - Modelo C-4: Micro-Gate com k=4 (132 params: 4x32 + 4)
  - Modelo C-8: Micro-Gate com k=8 (264 params: 8x32 + 8)
  - Modelo D-4: Gate Externo com h in R^5120 (20.484 params: 4x5120 + 4)
  - Oráculo alpha*(x): Otimização direta por sequência held-out (limite superior)

U_c in R^(5120x32), W_opt in R^(32x5120) e norm_mod MANTIDOS RIGOROSAMENTE CONGELADOS.
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

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("CICLO 22: MICRO-GATE MULTIVARIADO DINÂMICO alpha(x) EM SO(32) E GENERALIZAÇÃO CAUSAL")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Horário: {time.strftime('%H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()

    # 1. Carregar Componentes Base
    print("\n[Passo 1/6] Carregando componentes do Qwen e congelando módulos de referência...")
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=16)
    cfg = comp["cfg"]
    lm_head_w = comp["lm_head_w"]
    norm_mod = comp["norm_mod"]
    for p in norm_mod.parameters():
        p.requires_grad = False
    norm_mod.eval()
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
    wiki_tr_targets = wiki_tokens[:8]
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

    h_wiki_tr = h_wiki_all[:8]
    h_wiki_te = h_wiki_all[16:24]

    # HF Code & GSM8K
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_tr = hf_data["h_in_63"][:8, :64].to(device=device, dtype=torch.bfloat16)
    code_tr_targets = hf_data["input_ids"][:8, :64].to(device)
    h_code_te = hf_data["h_in_63"][8:16, :64].to(device=device, dtype=torch.bfloat16)
    code_te_targets = hf_data["input_ids"][8:16, :64].to(device)

    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_tr = gsm_data["h_in_63"][:8, :64].to(device=device, dtype=torch.bfloat16)
    gsm_tr_targets = gsm_data["input_ids"][:8, :64].to(device)
    h_gsm_te = gsm_data["h_in_63"][8:16, :64].to(device=device, dtype=torch.bfloat16)
    gsm_te_targets = gsm_data["input_ids"][8:16, :64].to(device)

    # 2. Calibrar e Congelar W_opt (idêntico aos ciclos anteriores)
    p_data = torch.load("checkpoints/prof_targets_cache.pt", map_location=device)
    logits_prof_tr = p_data["logits_prof_tr"].to(device)
    prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
    y_tr_63 = prof_intermediates_tr[63].to(device).reshape(-1, 5120)

    W_opt = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    opt_w = torch.optim.AdamW([W_opt], lr=1e-4, weight_decay=1e-3)
    corr_tr0 = torch.matmul(torch.matmul(h_wiki_all[:16], U_base), W_opt)
    l_tr0 = F.linear(norm_mod(h_wiki_all[:16] + corr_tr0), lm_head_w)
    s_flat0 = l_tr0[:, :-1, :].reshape(-1, vocab_sz)
    t_flat0 = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
    y_flat0 = wiki_tokens[:16, 1:].contiguous().reshape(-1)
    loss0 = F.kl_div(F.log_softmax(s_flat0.float(), dim=-1), F.softmax(t_flat0.float(), dim=-1), reduction="batchmean") + \
            0.1 * F.cross_entropy(s_flat0.float(), y_flat0, reduction="mean") + \
            0.05 * F.mse_loss((h_wiki_all[:16] + corr_tr0).reshape(-1, 5120).float(), y_tr_63.float())
    loss0.backward()
    opt_w.step()

    W_opt.requires_grad = False
    U_base.requires_grad = False
    print("  [OK] U_c(32), W_opt e norm_mod RIGOROSAMENTE CONGELADOS.")

    # 3. Geradores Canônicos de Lie A_1 ... A_16
    print("\n[Passo 2/6] Sintetizando os geradores ortonormais canônicos de so(32)...")
    def compute_so32_grad(h_seqs, targets):
        a = torch.matmul(h_seqs, U_base)
        delta_h = torch.zeros_like(h_seqs, requires_grad=True)
        h_comb = h_seqs + delta_h
        h_norm = norm_mod(h_comb)
        logits = F.linear(h_norm, lm_head_w)
        l_pred = logits[:, :-1, :].reshape(-1, vocab_sz)
        tgt_flat = targets[:, 1:].contiguous().reshape(-1)
        loss = F.cross_entropy(l_pred.float(), tgt_flat, reduction="sum")
        loss.backward()
        b = torch.matmul(delta_h.grad, W_opt.t())
        a_tok = a[:, :-1, :].reshape(-1, 32)
        b_tok = b[:, :-1, :].reshape(-1, 32)
        idx_i, idx_j = torch.triu_indices(32, 32, offset=1, device=device)
        return 0.5 * (a_tok[:, idx_i] * b_tok[:, idx_j] - a_tok[:, idx_j] * b_tok[:, idx_i]).detach()

    g_w = compute_so32_grad(h_wiki_tr, wiki_tr_targets)
    g_c = compute_so32_grad(h_code_tr, code_tr_targets)
    g_g = compute_so32_grad(h_gsm_tr, gsm_tr_targets)
    g_all = torch.cat([g_w, g_c, g_g], dim=0)

    g_f = g_all.float()
    g_centered = g_f - g_f.mean(dim=0, keepdim=True)
    C = torch.matmul(g_centered.t(), g_centered) / (g_f.shape[0] - 1)
    eigenvals, eigenvecs = torch.linalg.eigh(C)
    eigenvecs = torch.flip(eigenvecs, dims=[1])
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
    A_stack = torch.stack(generators, dim=0) # [16, 32, 32]
    print(f"  [OK] {len(generators)} geradores ortonormais sintetizados com sucesso.")

    # 4. Definição das Classes de Gating Dinâmico
    class DynamicMicroGate32(nn.Module):
        def __init__(self, k, alpha_scale=2.5):
            super().__init__()
            self.k = k
            self.register_buffer("A_sub", A_stack[:k].float())
            self.w = nn.Parameter(torch.randn(k, 32, dtype=torch.float32) * 0.02)
            self.b = nn.Parameter(torch.zeros(k, dtype=torch.float32))
            self.alpha_scale = alpha_scale

        def forward(self, a):
            a_seq = a.mean(dim=1).float() # [B, 32]
            alpha = self.alpha_scale * torch.tanh(F.linear(a_seq, self.w, self.b)) # [B, k]
            A_comb = torch.einsum("bk,kij->bij", alpha, self.A_sub)
            Q = torch.linalg.matrix_exp(A_comb).to(a.dtype)
            a_rot = torch.matmul(a, Q)
            return a_rot, alpha

    class DynamicGate5120(nn.Module):
        def __init__(self, k, alpha_scale=2.5):
            super().__init__()
            self.k = k
            self.register_buffer("A_sub", A_stack[:k].float())
            self.w = nn.Parameter(torch.randn(k, 5120, dtype=torch.float32) * 0.002)
            self.b = nn.Parameter(torch.zeros(k, dtype=torch.float32))
            self.alpha_scale = alpha_scale

        def forward(self, h, a):
            with torch.no_grad():
                h_norm = norm_mod(h).mean(dim=1).float() # [B, 5120] detached!
            alpha = self.alpha_scale * torch.tanh(F.linear(h_norm, self.w, self.b))
            A_comb = torch.einsum("bk,kij->bij", alpha, self.A_sub)
            Q = torch.linalg.matrix_exp(A_comb).to(a.dtype)
            a_rot = torch.matmul(a, Q)
            return a_rot, alpha

    def eval_loss_and_ppl(h_seqs, targets, a_rot):
        corr = torch.matmul(a_rot, W_opt)
        h_out = norm_mod(h_seqs + corr)
        logits = F.linear(h_out, lm_head_w)
        l_pred = logits[:, :-1, :].reshape(-1, vocab_sz)
        tgt_flat = targets[:, 1:].contiguous().reshape(-1)
        ce = F.cross_entropy(l_pred.float(), tgt_flat, reduction="mean").item()
        ppl = math.exp(min(ce, 20.0))
        return ppl, ce

    # 5. Dataset de Calibração Multitarefa
    print("\n[Passo 3/6] Montando dataset multitarefa unificado de calibração (24 seqs)...")
    h_multi_tr = torch.cat([h_wiki_tr, h_code_tr, h_gsm_tr], dim=0)
    tgt_multi_tr = torch.cat([wiki_tr_targets, code_tr_targets, gsm_tr_targets], dim=0)
    a_multi_tr = torch.matmul(h_multi_tr, U_base)
    tgt_multi_flat = tgt_multi_tr[:, 1:].contiguous().reshape(-1)

    def train_gate(gate, num_steps=25, lr=0.03, is_5120=False):
        opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(num_steps):
            opt.zero_grad()
            if is_5120:
                a_rot, _ = gate(h_multi_tr, a_multi_tr)
            else:
                a_rot, _ = gate(a_multi_tr)
            corr = torch.matmul(a_rot, W_opt)
            l = F.linear(norm_mod(h_multi_tr + corr), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
            loss = F.cross_entropy(l.float(), tgt_multi_flat)
            loss.backward()
            opt.step()
        return gate

    # 6. Treinar Modelos
    print("\n[Passo 4/6] Treinando Micro-Gates Dinâmicos (Modelos C-2, C-4, C-8 e D-4)...")
    t_tr0 = time.time()
    gate_c2 = train_gate(DynamicMicroGate32(2).to(device))
    gate_c4 = train_gate(DynamicMicroGate32(4).to(device))
    gate_c8 = train_gate(DynamicMicroGate32(8).to(device))
    gate_d4 = train_gate(DynamicGate5120(4).to(device), is_5120=True)
    t_tr = time.time() - t_tr0
    print(f"  [OK] Treinamento dos 4 gates concluído em: {t_tr:.2f} s")

    # 7. Oráculo por Sequência Held-Out (Limite Superior Ótimo)
    def optimize_oracle(k, h_seqs, targets, num_steps=25, lr=0.04):
        B, S, D = h_seqs.shape
        a = torch.matmul(h_seqs, U_base)
        A_sub = A_stack[:k].float()
        alphas_star = []
        ppl_list = []
        ce_list = []
        for i in range(B):
            h_i = h_seqs[i:i+1]
            tgt_i = targets[i:i+1, 1:].contiguous().reshape(-1)
            a_i = a[i:i+1]
            alpha_p = nn.Parameter(torch.zeros(k, device=device, dtype=torch.float32))
            opt = torch.optim.AdamW([alpha_p], lr=lr, weight_decay=1e-4)
            for _ in range(num_steps):
                opt.zero_grad()
                A_comb = torch.einsum("k,kij->ij", alpha_p, A_sub)
                Q = torch.linalg.matrix_exp(A_comb).to(a_i.dtype)
                corr_i = torch.matmul(torch.matmul(a_i, Q), W_opt)
                l_i = F.linear(norm_mod(h_i + corr_i), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
                loss_i = F.cross_entropy(l_i.float(), tgt_i)
                loss_i.backward()
                opt.step()
            with torch.no_grad():
                A_comb = torch.einsum("k,kij->ij", alpha_p, A_sub)
                Q = torch.linalg.matrix_exp(A_comb).to(a_i.dtype)
                corr_i = torch.matmul(torch.matmul(a_i, Q), W_opt)
                l_i = F.linear(norm_mod(h_i + corr_i), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
                ce_val = F.cross_entropy(l_i.float(), tgt_i).item()
            alphas_star.append(alpha_p.detach().cpu().numpy())
            ce_list.append(ce_val)
            ppl_list.append(math.exp(ce_val))
        return float(np.mean(ppl_list)), float(np.mean(ce_list)), np.array(alphas_star)

    # 8. Avaliação em Held-Out
    test_domains = {
        "WikiText": (h_wiki_te, wiki_te_targets),
        "HF Code": (h_code_te, code_te_targets),
        "GSM8K": (h_gsm_te, gsm_te_targets)
    }

    c21_res_path = os.path.join(repo_root, "experiments", "cumulative_lie_modes_results.json")
    with open(c21_res_path, "r") as f:
        c21_data = json.load(f)

    print("\n[Passo 5/6] Avaliando modelos no conjunto de teste held-out (8 sequências por domínio)...")
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seq_len": 64,
            "held_out_sequences_per_domain": 8,
            "train_time_sec": float(t_tr)
        },
        "models": {
            "A_Identity": {},
            "B_Static_k4": {},
            "C2_Dynamic_66p": {},
            "C4_Dynamic_132p": {},
            "C8_Dynamic_264p": {},
            "D4_External_20484p": {},
            "Oracle_k4": {}
        },
        "trajectories": {}
    }

    print("-" * 115)
    print(f"{'Domínio':<10} | {'Mod A (Q=I)':<12} | {'Mod B (Fixo k4)':<15} | {'Mod C-2 (66p)':<15} | {'Mod C-4 (132p)':<15} | {'Mod C-8 (264p)':<15} | {'Mod D-4 (20k)':<15} | {'Oráculo k4':<12}")
    print("-" * 115)

    for dom, (h_te, tgt_te) in test_domains.items():
        a_te = torch.matmul(h_te, U_base)
        
        # Mod A
        ppl_a, ce_a = eval_loss_and_ppl(h_te, tgt_te, a_te)
        
        # Mod B (Fixo Ciclo 21 k=4)
        c21_dom = c21_data["domains"][dom]
        k_idx4 = c21_data["k_values"].index(4)
        ppl_b = c21_dom["ppl"][k_idx4]
        ce_b = c21_dom["ce"][k_idx4]
        
        # Mod C-2, C-4, C-8
        with torch.no_grad():
            a_rot_c2, alpha_c2 = gate_c2(a_te)
            ppl_c2, ce_c2 = eval_loss_and_ppl(h_te, tgt_te, a_rot_c2)
            
            a_rot_c4, alpha_c4 = gate_c4(a_te)
            ppl_c4, ce_c4 = eval_loss_and_ppl(h_te, tgt_te, a_rot_c4)
            
            a_rot_c8, alpha_c8 = gate_c8(a_te)
            ppl_c8, ce_c8 = eval_loss_and_ppl(h_te, tgt_te, a_rot_c8)
            
            a_rot_d4, alpha_d4 = gate_d4(h_te, a_te)
            ppl_d4, ce_d4 = eval_loss_and_ppl(h_te, tgt_te, a_rot_d4)
            
        # Oráculo
        ppl_ora, ce_ora, alphas_star = optimize_oracle(4, h_te, tgt_te)
        
        results["models"]["A_Identity"][dom] = {"ppl": ppl_a, "ce": ce_a}
        results["models"]["B_Static_k4"][dom] = {"ppl": ppl_b, "ce": ce_b}
        results["models"]["C2_Dynamic_66p"][dom] = {"ppl": ppl_c2, "ce": ce_c2, "delta_ppl": ppl_c2 - ppl_a}
        results["models"]["C4_Dynamic_132p"][dom] = {"ppl": ppl_c4, "ce": ce_c4, "delta_ppl": ppl_c4 - ppl_a}
        results["models"]["C8_Dynamic_264p"][dom] = {"ppl": ppl_c8, "ce": ce_c8, "delta_ppl": ppl_c8 - ppl_a}
        results["models"]["D4_External_20484p"][dom] = {"ppl": ppl_d4, "ce": ce_d4, "delta_ppl": ppl_d4 - ppl_a}
        results["models"]["Oracle_k4"][dom] = {"ppl": ppl_ora, "ce": ce_ora, "delta_ppl": ppl_ora - ppl_a}
        
        results["trajectories"][dom] = {
            "alpha_pred_c4": alpha_c4.cpu().numpy().tolist(),
            "alpha_pred_d4": alpha_d4.cpu().numpy().tolist(),
            "alpha_oracle_k4": alphas_star.tolist()
        }
        
        print(f"{dom:<10} | {ppl_a:12.2f} | {ppl_b:15.2f} | {ppl_c2:15.2f} | {ppl_c4:15.2f} | {ppl_c8:15.2f} | {ppl_d4:15.2f} | {ppl_ora:12.2f}")

    print("-" * 115)

    # Salvar JSON
    res_path = os.path.join(repo_root, "experiments", "multivariate_lie_gating_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {res_path}")

    # 9. Figuras Científicas
    print("\n[Passo 6/6] Gerando figuras científicas de alta resolução (300 DPI)...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    dom_list = ["WikiText", "HF Code", "GSM8K"]
    model_names = ["Mod A (Q=I)", "Mod B (Fixo k4)", "Mod C-2 (66p)", "Mod C-4 (132p)", "Mod C-8 (264p)", "Mod D-4 (20k)", "Oráculo k4"]
    colors = ["#718096", "#4A5568", "#4299E1", "#48BB78", "#9F7AEA", "#ED8936", "#F56565"]
    
    for ax, dom in zip(axes, dom_list):
        vals = [
            results["models"]["A_Identity"][dom]["ppl"],
            results["models"]["B_Static_k4"][dom]["ppl"],
            results["models"]["C2_Dynamic_66p"][dom]["ppl"],
            results["models"]["C4_Dynamic_132p"][dom]["ppl"],
            results["models"]["C8_Dynamic_264p"][dom]["ppl"],
            results["models"]["D4_External_20484p"][dom]["ppl"],
            results["models"]["Oracle_k4"][dom]["ppl"]
        ]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.9, width=0.6, edgecolor="#CBD5E0", linewidth=0.8)
        ax.set_title(f"Domínio: {dom}", fontsize=13, fontweight="bold", color="#E2E8F0", pad=10)
        ax.set_ylabel("Perplexidade (PPL)" if dom == "WikiText" else "", fontsize=11, color="#CBD5E0")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9, color="#CBD5E0")
        ax.grid(True, linestyle="--", alpha=0.2, axis="y")
        
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", 
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#FFFFFF")
            
    fig.suptitle(r"Ciclo 22: Comparação PPL — Micro-Gates Dinâmicos $a \in \mathbb{R}^{32}$ vs Linha de Base Externa e Oráculo", 
                 fontsize=14, fontweight="bold", color="#FFFFFF", y=1.02)
    plt.tight_layout()
    plot1_path = os.path.join(repo_root, "experiments", "so32_gating_ppl_comparison.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 1 salva em: {plot1_path}")

    # FIGURA 2: Trajetórias e Correlação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    dom_markers = {"WikiText": "o", "HF Code": "s", "GSM8K": "^"}
    dom_palette = {"WikiText": "#4299E1", "HF Code": "#48BB78", "GSM8K": "#ED8936"}
    
    for dom in dom_list:
        p_alphas = np.array(results["trajectories"][dom]["alpha_pred_c4"]) # [8, 4]
        o_alphas = np.array(results["trajectories"][dom]["alpha_oracle_k4"]) # [8, 4]
        ax1.scatter(p_alphas[:, 0], p_alphas[:, 1], color=dom_palette[dom], marker=dom_markers[dom], 
                    s=80, alpha=0.85, edgecolors="#FFFFFF", linewidth=1.0, label=f"{dom} (Pred C-4)")
        ax1.scatter(o_alphas[:, 0], o_alphas[:, 1], color=dom_palette[dom], marker="x", 
                    s=70, alpha=0.5, linewidth=1.5, label=f"{dom} (Oráculo)")
        
    ax1.set_title(r"Distribuição Geodésica no Subespaço de Lie $(\alpha_1, \alpha_2)$", fontsize=12, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel(r"Modo 1 ($\alpha_1$, radianos)", fontsize=11, color="#CBD5E0")
    ax1.set_ylabel(r"Modo 2 ($\alpha_2$, radianos)", fontsize=11, color="#CBD5E0")
    ax1.axhline(0, color="#718096", linestyle=":", alpha=0.5)
    ax1.axvline(0, color="#718096", linestyle=":", alpha=0.5)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.legend(fontsize=8.5, loc="best", framealpha=0.3)

    params_count = [0, 0, 66, 132, 264, 20484]
    m_labels = ["Mod A\n(Q=I)", "Mod B\n(Fixo)", "Mod C-2\n(66p)", "Mod C-4\n(132p)", "Mod C-8\n(264p)", "Mod D-4\n(20.484p)"]
    
    ppl_a_code = results["models"]["A_Identity"]["HF Code"]["ppl"]
    ppl_ora_code = results["models"]["Oracle_k4"]["HF Code"]["ppl"]
    max_rec_code = max(ppl_a_code - ppl_ora_code, 1e-4)
    
    rec_pcts = [
        0.0,
        ((ppl_a_code - results["models"]["B_Static_k4"]["HF Code"]["ppl"]) / max_rec_code) * 100.0,
        ((ppl_a_code - results["models"]["C2_Dynamic_66p"]["HF Code"]["ppl"]) / max_rec_code) * 100.0,
        ((ppl_a_code - results["models"]["C4_Dynamic_132p"]["HF Code"]["ppl"]) / max_rec_code) * 100.0,
        ((ppl_a_code - results["models"]["C8_Dynamic_264p"]["HF Code"]["ppl"]) / max_rec_code) * 100.0,
        ((ppl_a_code - results["models"]["D4_External_20484p"]["HF Code"]["ppl"]) / max_rec_code) * 100.0
    ]
    
    b2 = ax2.bar(range(len(rec_pcts)), rec_pcts, color=["#718096", "#4A5568", "#4299E1", "#48BB78", "#9F7AEA", "#ED8936"], 
                 width=0.6, edgecolor="#CBD5E0", linewidth=0.8)
    ax2.set_title(r"Eficiência de Recuperação Causal em HF Code (% do Oráculo)", fontsize=12, fontweight="bold", color="#E2E8F0")
    ax2.set_ylabel("% Recuperação vs Oráculo", fontsize=11, color="#CBD5E0")
    ax2.set_xticks(range(len(rec_pcts)))
    ax2.set_xticklabels(m_labels, fontsize=9.5, color="#CBD5E0")
    ax2.grid(True, linestyle="--", alpha=0.2, axis="y")
    
    for bar, pct, p_cnt in zip(b2, rec_pcts, params_count):
        y_pos = max(bar.get_height() + 1.5, 2.0)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f"{pct:.1f}%\n({p_cnt} p)", 
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#FFFFFF")
        
    fig.suptitle(r"Ciclo 22: Geometria de Roteamento Dinâmico em $\mathfrak{so}(32)$ e Compressão Paramétrica", 
                 fontsize=14, fontweight="bold", color="#FFFFFF", y=1.02)
    plt.tight_layout()
    plot2_path = os.path.join(repo_root, "experiments", "so32_gating_alpha_trajectories.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 2 salva em: {plot2_path}")

    total_time = time.time() - t_start
    print("\n" + "=" * 115)
    print(f"CICLO 22 EXECUTADO COM SUCESSO! Tempo total: {total_time:.2f} s")
    print("=" * 115)

if __name__ == "__main__":
    main()
