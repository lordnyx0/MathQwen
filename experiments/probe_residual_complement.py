#!/usr/bin/env python3
r"""
experiments/probe_residual_complement.py

CICLO 23: Sonda Funcional do Espaço Residual Externo a U_c(32)
--------------------------------------------------------------
Objetivo:
Testar rigorosamente se existe informação funcional relevante no complemento ortogonal U_c^\perp,
mantendo o melhor estado interno obtido nos ciclos anteriores RIGOROSAMENTE CONGELADO.

Pergunta Central:
    Existe capacidade funcional importante em U_c^\perp que não pode ser recuperada
    por coordenatização interna em U_c?

Decomposição:
    P_c = U_c @ U_c^T in R^(5120x5120)
    P_perp = I - P_c
    U_c^T @ U_perp = 0 (verificado com precisão de máquina < 1e-6)

Condições Avaliadas:
    - Baseline Congelado: r = 0 (PPL de referência com U_c(32) e W_opt congelados)
    - Subespaço Residual Estruturado: U_perp(r) para r in {1, 2, 4, 8, 16, 32}
    - Controle Aleatório em U_c^\perp: U_rand(r) para r in {1, 2, 4, 8, 16, 32}
    - Teste Causal Unidimensional: Delta PPL_i para cada uma das 32 direções marginais

Critério Decisivo:
    Classificação formal nos Cenários A (irrelevante), B (moderado) ou C (forte).
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
    print("CICLO 23: SONDA FUNCIONAL DO ESPAÇO RESIDUAL EXTERNO A U_c(32)")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Horário: {time.strftime('%H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()

    # ---------------------------------------------------------
    # 1. Carregar Componentes Base & Congelar Módulos
    # ---------------------------------------------------------
    print("\n[Passo 1/6] Carregando componentes do Qwen e congelando módulos de referência...")
    comp = load_qwen_reference_components(device=device, seq_len=64, num_seqs=16)
    cfg = comp["cfg"]
    lm_head_w = comp["lm_head_w"]
    norm_mod = comp["norm_mod"]
    for p in norm_mod.parameters():
        p.requires_grad = False
    norm_mod.eval()
    vocab_sz = lm_head_w.shape[0]

    # Carregar U_c(32) e garantir ortonormalidade exata
    func_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    saved_func = torch.load(func_path, map_location=device)
    U_base = saved_func["U_f"].to(device=device, dtype=torch.float32)
    U_base, _ = torch.linalg.qr(U_base)
    U_base = U_base.to(dtype=torch.bfloat16)

    # Validar ortonormalidade de U_c
    eye_err = torch.norm(torch.matmul(U_base.t().float(), U_base.float()) - torch.eye(32, device=device)).item()
    print(f"  [OK] U_c(32) carregado. Erro de ortonormalidade ||U_c^T U_c - I|| = {eye_err:.2e}")

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

    # ---------------------------------------------------------
    # 2. Calibrar e Congelar W_opt (Baseline r = 0)
    # ---------------------------------------------------------
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
    print("  [OK] U_c(32) e W_opt RIGOROSAMENTE CONGELADOS.")

    # Função de avaliação pura
    def evaluate_ppl_and_ce(h_seqs, targets, delta_extra=None):
        a = torch.matmul(h_seqs, U_base)
        corr_base = torch.matmul(a, W_opt)
        h_total = h_seqs + corr_base
        if delta_extra is not None:
            h_total = h_total + delta_extra
        h_out = norm_mod(h_total)
        logits = F.linear(h_out, lm_head_w)
        l_pred = logits[:, :-1, :].reshape(-1, vocab_sz)
        tgt_flat = targets[:, 1:].contiguous().reshape(-1)
        ce = F.cross_entropy(l_pred.float(), tgt_flat, reduction="mean").item()
        ppl = math.exp(min(ce, 20.0))
        return ppl, ce

    # Validar PPL Baseline (r = 0)
    ppl_w0, ce_w0 = evaluate_ppl_and_ce(h_wiki_te, wiki_te_targets)
    ppl_c0, ce_c0 = evaluate_ppl_and_ce(h_code_te, code_te_targets)
    ppl_g0, ce_g0 = evaluate_ppl_and_ce(h_gsm_te, gsm_te_targets)
    print(f"\n  --- BASELINE CONGELADO (r = 0) ---")
    print(f"  WikiText PPL(0): {ppl_w0:.2f}")
    print(f"  HF Code  PPL(0): {ppl_c0:.2f}")
    print(f"  GSM8K    PPL(0): {ppl_g0:.2f}")

    # ---------------------------------------------------------
    # 3. Construção do Complemento U_c^perp e Extração Espectral
    # ---------------------------------------------------------
    print("\n[Passo 2/6] Decompondo gradientes residuais em U_c^perp e sintetizando base estruturada...")
    
    # Dataset multitarefa de treino (24 seqs)
    h_multi_tr = torch.cat([h_wiki_tr, h_code_tr, h_gsm_tr], dim=0)
    tgt_multi_tr = torch.cat([wiki_tr_targets, code_tr_targets, gsm_tr_targets], dim=0)
    tgt_multi_flat = tgt_multi_tr[:, 1:].contiguous().reshape(-1)

    # Computar gradiente da perda w.r.t h
    delta_h = torch.zeros_like(h_multi_tr, requires_grad=True)
    a_tr = torch.matmul(h_multi_tr, U_base)
    corr_tr = torch.matmul(a_tr, W_opt)
    h_out_tr = norm_mod(h_multi_tr + corr_tr + delta_h)
    logits_tr = F.linear(h_out_tr, lm_head_w)
    loss_tr = F.cross_entropy(logits_tr[:, :-1, :].reshape(-1, vocab_sz).float(), tgt_multi_flat, reduction="sum")
    loss_tr.backward()
    grad_h = delta_h.grad # [24, 64, 5120]

    # Projetar no complemento ortogonal: g_perp = g - (g @ U_c) @ U_c^T
    g_flat = grad_h.view(-1, 5120).float() # [N, 5120]
    g_c_proj = torch.matmul(g_flat, U_base.float()) # [N, 32]
    g_perp = g_flat - torch.matmul(g_c_proj, U_base.t().float()) # [N, 5120]

    # Verificação de ortogonalidade de máquina: g_perp @ U_c deve ser zero
    ortho_leak = torch.norm(torch.matmul(g_perp, U_base.float())).item()
    print(f"  [OK] Gradientes projetados em U_c^perp. Vazamento residual ||g_perp @ U_c|| = {ortho_leak:.2e}")

    # Extrair os 32 autovetores dominantes de covariância em U_c^perp via SVD
    g_perp_cent = g_perp - g_perp.mean(dim=0, keepdim=True)
    _, S_vals, Vh = torch.linalg.svd(g_perp_cent, full_matrices=False)
    # S_vals**2 / (N-1) dá os autovalores de covariância
    eigenvals_perp = ((S_vals ** 2) / (g_perp.shape[0] - 1)).cpu().numpy()
    
    # Base ortonormal estruturada U_perp_struct in R^(5120 x 32)
    V_top32 = Vh[:32].t() # [5120, 32]
    # Garantir ortogonalidade exata a U_c via QR no complemento
    V_ortho = V_top32 - torch.matmul(U_base.float(), torch.matmul(U_base.t().float(), V_top32))
    U_perp_struct, _ = torch.linalg.qr(V_ortho)
    U_perp_struct = U_perp_struct.to(device=device, dtype=torch.bfloat16)

    # Validar ortogonalidade estrita de U_perp_struct
    leak_struct = torch.norm(torch.matmul(U_base.t().float(), U_perp_struct.float())).item()
    print(f"  [OK] Base estruturada U_perp(32) sintetizada. ||U_c^T U_perp_struct|| = {leak_struct:.2e}")

    # ---------------------------------------------------------
    # 4. Construção do Controle B: Base Aleatória em U_c^perp
    # ---------------------------------------------------------
    print("\n[Passo 3/6] Sintetizando base de controle aleatória em U_c^perp...")
    torch.manual_seed(42)
    G_rand = torch.randn(5120, 32, device=device, dtype=torch.float32)
    G_perp = G_rand - torch.matmul(U_base.float(), torch.matmul(U_base.t().float(), G_rand))
    U_perp_rand, _ = torch.linalg.qr(G_perp)
    U_perp_rand = U_perp_rand.to(device=device, dtype=torch.bfloat16)
    leak_rand = torch.norm(torch.matmul(U_base.t().float(), U_perp_rand.float())).item()
    print(f"  [OK] Base aleatória U_rand(32) sintetizada. ||U_c^T U_perp_rand|| = {leak_rand:.2e}")

    # ---------------------------------------------------------
    # 5. Teste Causal Unidimensional: Delta PPL_i para i in 1..32
    # ---------------------------------------------------------
    print("\n[Passo 4/6] Executando teste causal unidimensional para cada direção v_i in U_c^perp...")
    single_dir_results = []
    
    test_domains = {
        "WikiText": (h_wiki_te, wiki_te_targets),
        "HF Code": (h_code_te, code_te_targets),
        "GSM8K": (h_gsm_te, gsm_te_targets)
    }

    def train_residual_linear(U_sub, num_steps=20, lr=0.01):
        r_dim = U_sub.shape[1]
        W_sub = nn.Parameter(torch.zeros(r_dim, 5120, device=device, dtype=torch.bfloat16))
        opt_sub = torch.optim.AdamW([W_sub], lr=lr, weight_decay=1e-4)
        a_sub_tr = torch.matmul(h_multi_tr, U_sub)
        
        for _ in range(num_steps):
            opt_sub.zero_grad()
            delta_sub = torch.matmul(a_sub_tr, W_sub)
            a_base = torch.matmul(h_multi_tr, U_base)
            h_tot = h_multi_tr + torch.matmul(a_base, W_opt) + delta_sub
            l_tr = F.linear(norm_mod(h_tot), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
            loss = F.cross_entropy(l_tr.float(), tgt_multi_flat)
            loss.backward()
            opt_sub.step()
            
        return W_sub.detach()

    for i in range(32):
        v_i = U_perp_struct[:, i:i+1] # [5120, 1]
        W_i = train_residual_linear(v_i, num_steps=15, lr=0.01)
        
        dir_res = {"dim": i + 1, "eigenval": float(eigenvals_perp[i]), "ppl": {}, "delta_ppl": {}}
        for dom, (h_te, tgt_te) in test_domains.items():
            with torch.no_grad():
                a_i_te = torch.matmul(h_te, v_i)
                delta_i_te = torch.matmul(a_i_te, W_i)
                ppl_val, _ = evaluate_ppl_and_ce(h_te, tgt_te, delta_extra=delta_i_te)
                base_ppl = ppl_w0 if dom == "WikiText" else (ppl_c0 if dom == "HF Code" else ppl_g0)
                dir_res["ppl"][dom] = ppl_val
                dir_res["delta_ppl"][dom] = ppl_val - base_ppl
        single_dir_results.append(dir_res)
        
    print(f"  [OK] 32 direções individuais avaliadas.")
    top3_code = sorted(single_dir_results, key=lambda x: x["delta_ppl"]["HF Code"])[:3]
    print(f"  Top-3 direções para HF Code:")
    for d in top3_code:
        print(f"    Dimensão {d['dim']:2d}: Delta PPL Code = {d['delta_ppl']['HF Code']:+6.2f} (lambda = {d['eigenval']:.4e})")

    # ---------------------------------------------------------
    # 6. Sondagem Cumulativa: r in {0, 1, 2, 4, 8, 16, 32}
    # ---------------------------------------------------------
    r_values = [0, 1, 2, 4, 8, 16, 32]
    print("\n[Passo 5/6] Executando varredura cumulativa PPL(r) para Estruturado vs Aleatório...")
    
    cumulative_results = {
        "r_values": r_values,
        "baseline": {"WikiText": ppl_w0, "HF Code": ppl_c0, "GSM8K": ppl_g0},
        "structured": {dom: {"ppl": [], "delta_ppl": []} for dom in test_domains},
        "random": {dom: {"ppl": [], "delta_ppl": []} for dom in test_domains}
    }

    # Inserir r = 0
    for dom in test_domains:
        b_ppl = cumulative_results["baseline"][dom]
        cumulative_results["structured"][dom]["ppl"].append(b_ppl)
        cumulative_results["structured"][dom]["delta_ppl"].append(0.0)
        cumulative_results["random"][dom]["ppl"].append(b_ppl)
        cumulative_results["random"][dom]["delta_ppl"].append(0.0)

    print("-" * 115)
    print(f"{'Rank r':<8} | {'Wiki (Estrut)':<15} | {'Wiki (Aleat)':<15} | {'Code (Estrut)':<15} | {'Code (Aleat)':<15} | {'GSM (Estrut)':<15} | {'GSM (Aleat)':<15}")
    print("-" * 115)

    for r in r_values[1:]:
        # 1. Estruturado
        U_sub_str = U_perp_struct[:, :r]
        W_sub_str = train_residual_linear(U_sub_str, num_steps=20, lr=0.01)
        
        # 2. Aleatório
        U_sub_rnd = U_perp_rand[:, :r]
        W_sub_rnd = train_residual_linear(U_sub_rnd, num_steps=20, lr=0.01)
        
        row_str = f"{r:<8} | "
        for dom, (h_te, tgt_te) in test_domains.items():
            b_ppl = cumulative_results["baseline"][dom]
            with torch.no_grad():
                # Estruturado
                a_str = torch.matmul(h_te, U_sub_str)
                d_str = torch.matmul(a_str, W_sub_str)
                ppl_str, _ = evaluate_ppl_and_ce(h_te, tgt_te, delta_extra=d_str)
                cumulative_results["structured"][dom]["ppl"].append(ppl_str)
                cumulative_results["structured"][dom]["delta_ppl"].append(ppl_str - b_ppl)
                
                # Aleatório
                a_rnd = torch.matmul(h_te, U_sub_rnd)
                d_rnd = torch.matmul(a_rnd, W_sub_rnd)
                ppl_rnd, _ = evaluate_ppl_and_ce(h_te, tgt_te, delta_extra=d_rnd)
                cumulative_results["random"][dom]["ppl"].append(ppl_rnd)
                cumulative_results["random"][dom]["delta_ppl"].append(ppl_rnd - b_ppl)
                
            row_str += f"{ppl_str:7.2f} ({ppl_str - b_ppl:+5.1f}) | {ppl_rnd:7.2f} ({ppl_rnd - b_ppl:+5.1f}) | "
            
        print(row_str)

    print("-" * 115)

    # ---------------------------------------------------------
    # 7. Diagnóstico e Cenário Decisivo
    # ---------------------------------------------------------
    code_gain_str32 = cumulative_results["structured"]["HF Code"]["delta_ppl"][-1]
    code_gain_rnd32 = cumulative_results["random"]["HF Code"]["delta_ppl"][-1]
    gsm_gain_str32 = cumulative_results["structured"]["GSM8K"]["delta_ppl"][-1]
    gsm_gain_rnd32 = cumulative_results["random"]["GSM8K"]["delta_ppl"][-1]

    print("\n[Diagnóstico Decisivo]")
    print(f"  HF Code r=32:  Delta Estruturado = {code_gain_str32:+6.2f} pts | Delta Aleatório = {code_gain_rnd32:+6.2f} pts")
    print(f"  GSM8K   r=32:  Delta Estruturado = {gsm_gain_str32:+6.2f} pts | Delta Aleatório = {gsm_gain_rnd32:+6.2f} pts")

    # Classificação
    if abs(code_gain_str32) < 5.0 and abs(gsm_gain_str32) < 3.0:
        cenario = "CENÁRIO A — RESIDUAL IRRELEVANTE (Capacidade funcional está 100% contida em U_c)"
    elif abs(code_gain_str32 - code_gain_rnd32) < 4.0:
        cenario = "CENÁRIO A* — GANHO DE CAPACIDADE GENÉRICA (O ganho estruturado é indistinguível do controle aleatório)"
    elif code_gain_str32 < -25.0:
        cenario = "CENÁRIO C — RESIDUAL FORTE (U_c^perp contém informação funcional indispensável)"
    else:
        cenario = "CENÁRIO B — RESIDUAL MODERADO (Ganhos moderados que justificam arquitetura híbrida U_c + U_perp)"

    print(f"\n  >>> CLASSIFICAÇÃO: {cenario} <<<")

    # Salvar Resultados JSON
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seq_len": 64,
            "held_out_sequences_per_domain": 8,
            "cenario": cenario
        },
        "cumulative_sweep": cumulative_results,
        "single_direction_causal": single_dir_results,
        "orthogonality_checks": {
            "eye_err_Uc": eye_err,
            "leak_struct": leak_struct,
            "leak_rand": leak_rand
        }
    }

    res_json_path = os.path.join(repo_root, "experiments", "probe_residual_complement_results.json")
    with open(res_json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {res_json_path}")

    # ---------------------------------------------------------
    # 8. Figuras Científicas (300 DPI)
    # ---------------------------------------------------------
    print("\n[Passo 6/6] Gerando figuras científicas de alta resolução (300 DPI)...")
    plt.style.use("dark_background")

    # FIGURA 1: Curvas de PPL(r) e Delta PPL(r)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    dom_titles = ["WikiText", "HF Code", "GSM8K"]
    
    for ax, dom in zip(axes, dom_titles):
        str_ppl = cumulative_results["structured"][dom]["ppl"]
        rnd_ppl = cumulative_results["random"][dom]["ppl"]
        b_val = cumulative_results["baseline"][dom]
        
        ax.plot(r_values, str_ppl, "o-", color="#48BB78", linewidth=2.2, markersize=6, label=r"Estruturado $U_\perp(r)$")
        ax.plot(r_values, rnd_ppl, "s--", color="#F56565", linewidth=1.8, markersize=5, alpha=0.8, label=r"Controle Aleatório em $U_c^\perp$")
        ax.axhline(b_val, color="#CBD5E0", linestyle=":", linewidth=1.5, label=f"Baseline $r=0$ ({b_val:.1f})")
        
        ax.set_title(f"Domínio: {dom}", fontsize=13, fontweight="bold", color="#E2E8F0", pad=10)
        ax.set_xlabel(r"Rank Residual $r$ em $U_c^\perp$", fontsize=11, color="#CBD5E0")
        ax.set_ylabel("Perplexidade (PPL)" if dom == "WikiText" else "", fontsize=11, color="#CBD5E0")
        ax.set_xticks(r_values)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=9, loc="best", framealpha=0.3)

    fig.suptitle(r"Ciclo 23: Sonda Funcional em $U_c^\perp$ — Estruturado vs Controle Aleatório ($r \in \{0, 1, 2, 4, 8, 16, 32\}$)", 
                 fontsize=14, fontweight="bold", color="#FFFFFF", y=1.02)
    plt.tight_layout()
    plot1_path = os.path.join(repo_root, "experiments", "so32_residual_ppl_curve.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 1 salva em: {plot1_path}")

    # FIGURA 2: Espectro de Autovalores e Ganho Causal Individual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # Espectro de Covariância em U_c^perp
    dims_x = np.arange(1, 33)
    ev_norm = eigenvals_perp[:32] / eigenvals_perp[0]
    ax1.bar(dims_x, ev_norm, color="#4299E1", alpha=0.85, width=0.7, edgecolor="#CBD5E0", linewidth=0.6)
    ax1.set_title(r"Espectro de Covariância de Gradientes no Complemento $U_c^\perp$", fontsize=12, fontweight="bold", color="#E2E8F0")
    ax1.set_xlabel("Dimensão Ortogonal $i$", fontsize=11, color="#CBD5E0")
    ax1.set_ylabel(r"Autovalor Normalizado ($\lambda_i / \lambda_1$)", fontsize=11, color="#CBD5E0")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.25)

    # Ganho Individual Delta PPL_i para Code e GSM
    delta_code_i = [d["delta_ppl"]["HF Code"] for d in single_dir_results]
    delta_gsm_i = [d["delta_ppl"]["GSM8K"] for d in single_dir_results]
    
    width = 0.4
    ax2.bar(dims_x - width/2, delta_code_i, width=width, color="#48BB78", alpha=0.85, label=r"HF Code $\Delta$PPL")
    ax2.bar(dims_x + width/2, delta_gsm_i, width=width, color="#ED8936", alpha=0.85, label=r"GSM8K $\Delta$PPL")
    ax2.axhline(0, color="#718096", linestyle=":", alpha=0.7)
    ax2.set_title(r"Efeito Causal Individual por Direção Residual $\Delta\text{PPL}_i$", fontsize=12, fontweight="bold", color="#E2E8F0")
    ax2.set_xlabel("Dimensão Ortogonal $i$", fontsize=11, color="#CBD5E0")
    ax2.set_ylabel(r"$\Delta\text{PPL}_i$ (Negativo = Melhora)", fontsize=11, color="#CBD5E0")
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.legend(fontsize=9.5, loc="best", framealpha=0.3)

    fig.suptitle(r"Ciclo 23: Sensibilidade Espectral e Granularidade Causal no Espaço Residual Externo $U_c^\perp$", 
                 fontsize=14, fontweight="bold", color="#FFFFFF", y=1.02)
    plt.tight_layout()
    plot2_path = os.path.join(repo_root, "experiments", "so32_residual_eigenvalues_and_causal.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 2 salva em: {plot2_path}")

    total_time = time.time() - t_start
    print("\n" + "=" * 115)
    print(f"CICLO 23 EXECUTADO COM SUCESSO! Tempo total: {total_time:.2f} s")
    print("=" * 115)

if __name__ == "__main__":
    main()
