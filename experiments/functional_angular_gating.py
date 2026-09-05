#!/usr/bin/env python3
"""
experiments/functional_angular_gating.py

CICLO 19: Micro-Gating Angular Dinâmico Q(x) em L63 (33 vs 5121 Parâmetros)
-----------------------------------------------------------------------------
Objetivo:
Validar a hipótese de que a capacidade adaptativa funcional pode ser recuperada
através de uma coordenatização dinâmica Q(x) in SO(32) parametrizada pelo estado.

Condições Avaliadas:
- Modelo A: Q = I (Controle estático, 0 params)
- Modelo B: Q = G(theta_fixa = 155.25 deg) (Universal fixo, 0 params)
- Modelo C: Q(x) = G(theta(x)), gate com a = h @ U_c in R^32 (33 params: 1x32 + 1)
- Modelo D: Q(x) = G(theta(x)), gate com h in R^5120 (5121 params: 1x5120 + 1)

U_c in R^(5120x32) e W_opt in R^(32x5120) mantidos RIGOROSAMENTE CONGELADOS.
Tempo de execução ultrarrápido: treinamento em lote compacto (< 3s) e
varredura vetorizada em lote de theta*(x) (< 1s).
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

def rotate_a_vectorized(a, theta_deg, plane_i=2, plane_j=9):
    """
    Aplica rotação de Givens no plano (plane_i, plane_j) diretamente em a in R^(..., 32).
    theta_deg: tensor com mesmo batch/seq dimensions de a (ou broadcastable).
    """
    rad = theta_deg * (math.pi / 180.0)
    cos_t = torch.cos(rad).unsqueeze(-1).to(a.dtype)
    sin_t = torch.sin(rad).unsqueeze(-1).to(a.dtype)
    
    a_rot = a.clone()
    ai = a[..., plane_i:plane_i+1]
    aj = a[..., plane_j:plane_j+1]
    a_rot[..., plane_i:plane_i+1] = ai * cos_t - aj * sin_t
    a_rot[..., plane_j:plane_j+1] = ai * sin_t + aj * cos_t
    return a_rot

class AngularMicroGate32(nn.Module):
    """Modelo C: Gate de 33 parâmetros operando sobre a = h @ U_c in R^32."""
    def __init__(self, theta_0_deg=155.25, delta_theta_deg=35.0, plane_i=2, plane_j=9):
        super().__init__()
        self.theta_0 = theta_0_deg
        self.delta_theta = delta_theta_deg
        self.plane_i = plane_i
        self.plane_j = plane_j
        self.w_g = nn.Parameter(torch.randn(1, 32, dtype=torch.bfloat16) * 0.02)
        self.b_g = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, a):
        # a: [batch, seq, 32]
        logits = F.linear(a, self.w_g, self.b_g).squeeze(-1).float()
        theta_deg = self.theta_0 + self.delta_theta * torch.tanh(logits)
        a_rot = rotate_a_vectorized(a, theta_deg, self.plane_i, self.plane_j)
        return a_rot, theta_deg

class AngularGate5120(nn.Module):
    """Modelo D: Gate de 5121 parâmetros operando sobre h in R^5120."""
    def __init__(self, norm_mod, theta_0_deg=155.25, delta_theta_deg=35.0, plane_i=2, plane_j=9):
        super().__init__()
        self.norm = norm_mod
        self.theta_0 = theta_0_deg
        self.delta_theta = delta_theta_deg
        self.plane_i = plane_i
        self.plane_j = plane_j
        self.w_g = nn.Parameter(torch.randn(1, 5120, dtype=torch.bfloat16) * 0.005)
        self.b_g = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, h, a):
        # h: [batch, seq, 5120], a: [batch, seq, 32]
        h_norm = self.norm(h)
        logits = F.linear(h_norm, self.w_g, self.b_g).squeeze(-1).float()
        theta_deg = self.theta_0 + self.delta_theta * torch.tanh(logits)
        a_rot = rotate_a_vectorized(a, theta_deg, self.plane_i, self.plane_j)
        return a_rot, theta_deg

def compute_ppl_and_theta(h_seqs, targets, U_base, W_opt, norm_mod, lm_head_w, model_type, gate=None, fixed_theta=155.25):
    """
    Avalia a PPL e registra a distribuição de theta(x) para um conjunto de sequências mantidas em teste.
    h_seqs: [B, S, 5120]
    targets: [B, S]
    """
    vocab_sz = lm_head_w.shape[0]
    B, S, D = h_seqs.shape
    a = torch.matmul(h_seqs, U_base) # [B, S, 32]
    
    with torch.no_grad():
        if model_type == "A": # Q = I
            theta_vals = torch.zeros(B, S, device=h_seqs.device)
            a_rot = a
        elif model_type == "B": # Q = G(fixed_theta)
            theta_vals = torch.full((B, S), fixed_theta, device=h_seqs.device)
            a_rot = rotate_a_vectorized(a, theta_vals, 2, 9)
        elif model_type == "C": # Gate 33 params
            a_rot, theta_vals = gate(a)
        elif model_type == "D": # Gate 5121 params
            a_rot, theta_vals = gate(h_seqs, a)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        corr = torch.matmul(a_rot, W_opt) # [B, S, 5120]
        h_out = norm_mod(h_seqs + corr)
        logits = F.linear(h_out, lm_head_w) # [B, S, vocab_sz]
        
        logits_pred = logits[:, :-1, :].reshape(-1, vocab_sz)
        tgt_flat = targets[:, 1:].contiguous().reshape(-1)
        
        ce = F.cross_entropy(logits_pred.float(), tgt_flat, reduction="mean")
        ppl = math.exp(min(ce.item(), 20.0))
        
    return ppl, ce.item(), theta_vals.cpu().numpy()

def compute_ground_truth_theta_star(h_seqs, targets, U_base, W_opt, norm_mod, lm_head_w):
    """
    Calcula theta*(x_i) via varredura vetorizada em lote para cada sequência individual held-out.
    Avalia 17 ângulos entre 100 e 180 graus (passo de 5 graus) simultaneamente em lote!
    """
    vocab_sz = lm_head_w.shape[0]
    B, S, D = h_seqs.shape
    angles_deg = torch.linspace(100, 180, 17, device=h_seqs.device) # 17 ângulos
    K = len(angles_deg)
    
    rad = angles_deg * (math.pi / 180.0)
    cos_t = torch.cos(rad).view(-1, 1, 1, 1).to(torch.bfloat16)
    sin_t = torch.sin(rad).view(-1, 1, 1, 1).to(torch.bfloat16)
    
    a = torch.matmul(h_seqs, U_base) # [B, S, 32]
    
    theta_stars = []
    losses_star = []
    
    with torch.no_grad():
        for i in range(B):
            h_i = h_seqs[i:i+1] # [1, S, 5120]
            tgt_i = targets[i:i+1, 1:].contiguous() # [1, S-1]
            a_i = a[i:i+1] # [1, S, 32]
            
            # Expandir para os K ângulos
            a_exp = a_i.unsqueeze(0).repeat(K, 1, 1, 1) # [K, 1, S, 32]
            a_rot = a_exp.clone()
            ai = a_exp[..., 2:3]
            aj = a_exp[..., 9:10]
            a_rot[..., 2:3] = ai * cos_t - aj * sin_t
            a_rot[..., 9:10] = ai * sin_t + aj * cos_t
            
            corr = torch.matmul(a_rot, W_opt) # [K, 1, S, 5120]
            h_new = h_i.unsqueeze(0) + corr # [K, 1, S, 5120]
            
            h_flat = h_new.view(-1, 5120) # [K*S, 5120]
            logits = F.linear(norm_mod(h_flat), lm_head_w) # [K*S, vocab_sz]
            logits_3d = logits.view(K, S, vocab_sz)[:, :-1] # [K, S-1, vocab_sz]
            
            tgt_rep = tgt_i.repeat(K, 1) # [K, S-1]
            losses = F.cross_entropy(logits_3d.reshape(-1, vocab_sz).float(), tgt_rep.reshape(-1), reduction="none")
            losses_per_angle = losses.view(K, S - 1).mean(dim=-1) # [K]
            
            best_idx = torch.argmin(losses_per_angle).item()
            theta_stars.append(angles_deg[best_idx].item())
            losses_star.append(losses_per_angle[best_idx].item())
            
    return np.array(theta_stars), np.array(losses_star)

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print("CICLO 19: MICRO-GATING ANGULAR DINÂMICO Q(x) EM L63 (33 vs 5121 PARÂMETROS)")
    print("=" * 110)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Horário: {time.strftime('%H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()

    # 1. Carregar Componentes Base (lm_head_w, norm_mod, rotary)
    print("\n[Passo 1/6] Carregando componentes do Qwen e dados held-out...")
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
    wiki_tr_targets = wiki_tokens[:16]
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

    h_wiki_tr = h_wiki_all[:16]
    h_wiki_te = h_wiki_all[16:24] # 8 seqs

    # HF Code (16 seqs de 64 tokens)
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_all = hf_data["h_in_63"][:, :64].to(device=device, dtype=torch.bfloat16)
    code_tokens_all = hf_data["input_ids"][:, :64].to(device)
    h_code_tr = h_code_all[:8]
    code_tr_targets = code_tokens_all[:8]
    h_code_te = h_code_all[8:16] # 8 seqs
    code_te_targets = code_tokens_all[8:16]

    # GSM8K (16 seqs de 64 tokens)
    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_all = gsm_data["h_in_63"][:, :64].to(device=device, dtype=torch.bfloat16)
    gsm_tokens_all = gsm_data["input_ids"][:, :64].to(device)
    h_gsm_tr = h_gsm_all[:8]
    gsm_tr_targets = gsm_tokens_all[:8]
    h_gsm_te = h_gsm_all[8:16] # 8 seqs
    gsm_te_targets = gsm_tokens_all[8:16]

    print(f"  Treino Multidomínio: Wiki={h_wiki_tr.shape}, Code={h_code_tr.shape}, GSM={h_gsm_tr.shape}")
    print(f"  Teste Held-Out:     Wiki={h_wiki_te.shape}, Code={h_code_te.shape}, GSM={h_gsm_te.shape}")

    # 2. Calibrar e Congelar W_opt
    print("\n[Passo 2/6] Calibrando e congelando refinador linear W_opt...")
    p_data = torch.load("checkpoints/prof_targets_cache.pt", map_location=device)
    logits_prof_tr = p_data["logits_prof_tr"].to(device)
    prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
    y_tr_63 = prof_intermediates_tr[63].to(device).reshape(-1, 5120)

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

    # CONGELAMENTO ESTRITO
    W_opt.requires_grad = False
    U_base.requires_grad = False
    print("  [OK] U_c(32) e W_opt RIGOROSAMENTE CONGELADOS.")

    # 3. Treinamento Ultrarrápido dos Gates C e D (< 3 segundos)
    print("\n[Passo 3/6] Otimizando Micro-Gates C (33 params) e D (5121 params) em batch representativo...")
    # Selecionar 2 seqs de cada domínio para o treino ultrarrápido (total 384 tokens)
    h_w_sub = h_wiki_tr[:2]
    w_tgt_sub = wiki_tr_targets[:2, 1:].contiguous().reshape(-1)
    h_c_sub = h_code_tr[:2]
    c_tgt_sub = code_tr_targets[:2, 1:].contiguous().reshape(-1)
    h_g_sub = h_gsm_tr[:2]
    g_tgt_sub = gsm_tr_targets[:2, 1:].contiguous().reshape(-1)

    a_w_sub = torch.matmul(h_w_sub, U_base)
    a_c_sub = torch.matmul(h_c_sub, U_base)
    a_g_sub = torch.matmul(h_g_sub, U_base)

    gate_c = AngularMicroGate32(theta_0_deg=155.25, delta_theta_deg=35.0, plane_i=2, plane_j=9).to(device)
    gate_d = AngularGate5120(norm_mod, theta_0_deg=155.25, delta_theta_deg=35.0, plane_i=2, plane_j=9).to(device)

    opt_c = torch.optim.AdamW(gate_c.parameters(), lr=2e-2, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(gate_d.parameters(), lr=1e-2, weight_decay=1e-4)

    # Treinar Gate C (33 params) - 25 passos rápidos
    t_c0 = time.time()
    for step in range(25):
        opt_c.zero_grad()
        ar_w, _ = gate_c(a_w_sub)
        ar_c, _ = gate_c(a_c_sub)
        ar_g, _ = gate_c(a_g_sub)

        cw = torch.matmul(ar_w, W_opt)
        cc = torch.matmul(ar_c, W_opt)
        cg = torch.matmul(ar_g, W_opt)

        lw = F.linear(norm_mod(h_w_sub + cw), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
        lc = F.linear(norm_mod(h_c_sub + cc), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
        lg = F.linear(norm_mod(h_g_sub + cg), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)

        loss = F.cross_entropy(lw.float(), w_tgt_sub) + \
               0.6 * F.cross_entropy(lc.float(), c_tgt_sub) + \
               0.4 * F.cross_entropy(lg.float(), g_tgt_sub)
        loss.backward()
        opt_c.step()
    t_c = time.time() - t_c0
    print(f"  [Gate C - 33 params] Treinado com sucesso em {t_c:.2f} s ({t_c/25*1000:.1f} ms/passo)!")

    # Treinar Gate D (5121 params) - 25 passos rápidos
    t_d0 = time.time()
    for step in range(25):
        opt_d.zero_grad()
        ar_w, _ = gate_d(h_w_sub, a_w_sub)
        ar_c, _ = gate_d(h_c_sub, a_c_sub)
        ar_g, _ = gate_d(h_g_sub, a_g_sub)

        cw = torch.matmul(ar_w, W_opt)
        cc = torch.matmul(ar_c, W_opt)
        cg = torch.matmul(ar_g, W_opt)

        lw = F.linear(norm_mod(h_w_sub + cw), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
        lc = F.linear(norm_mod(h_c_sub + cc), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)
        lg = F.linear(norm_mod(h_g_sub + cg), lm_head_w)[:, :-1, :].reshape(-1, vocab_sz)

        loss = F.cross_entropy(lw.float(), w_tgt_sub) + \
               0.6 * F.cross_entropy(lc.float(), c_tgt_sub) + \
               0.4 * F.cross_entropy(lg.float(), g_tgt_sub)
        loss.backward()
        opt_d.step()
    t_d = time.time() - t_d0
    print(f"  [Gate D - 5121 params] Treinado com sucesso em {t_d:.2f} s ({t_d/25*1000:.1f} ms/passo)!")

    # 4. Avaliação Comparativa dos 4 Modelos em Dados Held-Out
    print("\n[Passo 4/6] Avaliando PPL dos 4 Modelos em Dados Held-Out...")
    domains = {
        "WikiText": (h_wiki_te, wiki_te_targets),
        "Code": (h_code_te, code_te_targets),
        "GSM8K": (h_gsm_te, gsm_te_targets)
    }

    results = {"models": {}, "domain_theta": {}, "correlation": {}}
    model_configs = [
        ("A", "Q = I (0 params)", None, 0),
        ("B", "Q = G(155.25) (0 params)", None, 0),
        ("C", "Q(x) via a (33 params)", gate_c, 33),
        ("D", "Q(x) via h (5121 params)", gate_d, 5121)
    ]

    for m_id, m_name, g_mod, n_params in model_configs:
        results["models"][m_id] = {
            "name": m_name,
            "params": n_params,
            "ppl": {},
            "ce": {}
        }
        print(f"  -> Avaliando Modelo {m_id} ({m_name})...")
        for dom_name, (h_te, tgt_te) in domains.items():
            ppl, ce, theta_vals = compute_ppl_and_theta(h_te, tgt_te, U_base, W_opt, norm_mod, lm_head_w, m_id, g_mod)
            results["models"][m_id]["ppl"][dom_name] = round(ppl, 2)
            results["models"][m_id]["ce"][dom_name] = round(ce, 4)
            if m_id in ["C", "D"]:
                if dom_name not in results["domain_theta"]:
                    results["domain_theta"][dom_name] = {}
                results["domain_theta"][dom_name][m_id] = {
                    "mean": float(np.mean(theta_vals)),
                    "std": float(np.std(theta_vals)),
                    "min": float(np.min(theta_vals)),
                    "max": float(np.max(theta_vals)),
                    "values": theta_vals.flatten().tolist()
                }

    # Imprimir Tabela Comparativa de PPL
    print("\n" + "=" * 80)
    print(f"{'Modelo':<28} | {'Parâmetros':<10} | {'WikiText PPL':<12} | {'Code PPL':<12} | {'GSM8K PPL':<12}")
    print("-" * 80)
    for m_id, m_name, _, n_p in model_configs:
        p_w = results['models'][m_id]['ppl']['WikiText']
        p_c = results['models'][m_id]['ppl']['Code']
        p_g = results['models'][m_id]['ppl']['GSM8K']
        print(f"{m_name:<28} | {n_p:<10} | {p_w:<12.2f} | {p_c:<12.2f} | {p_g:<12.2f}")
    print("=" * 80)

    # 5. Cálculo Ground-Truth theta*(x) e Correlação Preditiva em Held-Out
    print("\n[Passo 5/6] Calculando theta*(x) ground-truth e correlação com theta_hat(x)...")
    all_theta_star = []
    all_theta_pred_c = []
    all_theta_pred_d = []
    domain_labels = []

    for dom_name, (h_te, tgt_te) in domains.items():
        t_star, _ = compute_ground_truth_theta_star(h_te, tgt_te, U_base, W_opt, norm_mod, lm_head_w)
        all_theta_star.extend(t_star.tolist())
        
        # Obter previsões médias por sequência de C e D
        with torch.no_grad():
            a_te = torch.matmul(h_te, U_base)
            _, th_c = gate_c(a_te)
            _, th_d = gate_d(h_te, a_te)
            th_c_seq = th_c.mean(dim=-1).cpu().numpy().tolist()
            th_d_seq = th_d.mean(dim=-1).cpu().numpy().tolist()
            
        all_theta_pred_c.extend(th_c_seq)
        all_theta_pred_d.extend(th_d_seq)
        domain_labels.extend([dom_name] * len(t_star))

    all_theta_star = np.array(all_theta_star)
    all_theta_pred_c = np.array(all_theta_pred_c)
    all_theta_pred_d = np.array(all_theta_pred_d)

    corr_c = float(np.corrcoef(all_theta_pred_c, all_theta_star)[0, 1])
    corr_d = float(np.corrcoef(all_theta_pred_d, all_theta_star)[0, 1])
    mae_c = float(np.mean(np.abs(all_theta_pred_c - all_theta_star)))
    mae_d = float(np.mean(np.abs(all_theta_pred_d - all_theta_star)))

    print(f"  [Modelo C - 33 params]   Correlação r(theta_hat, theta*) = {corr_c:+.4f} | MAE = {mae_c:.2f} graus")
    print(f"  [Modelo D - 5121 params] Correlação r(theta_hat, theta*) = {corr_d:+.4f} | MAE = {mae_d:.2f} graus")

    results["correlation"] = {
        "model_c": {"r": corr_c, "mae_deg": mae_c},
        "model_d": {"r": corr_d, "mae_deg": mae_d},
        "theta_star_seqs": all_theta_star.tolist(),
        "theta_pred_c_seqs": all_theta_pred_c.tolist(),
        "theta_pred_d_seqs": all_theta_pred_d.tolist(),
        "domain_labels": domain_labels
    }

    # Salvar Resultados JSON
    res_path = os.path.join(repo_root, "experiments", "angular_gating_results.json")
    # Salvar sem a lista massiva de values para manter o JSON conciso
    clean_results = json.loads(json.dumps(results))
    for dom in clean_results["domain_theta"]:
        for m in clean_results["domain_theta"][dom]:
            clean_results["domain_theta"][dom][m].pop("values", None)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"  [OK] Resultados salvos em: {res_path}")

    # 6. Geração dos Gráficos em Alta Resolução
    print("\n[Passo 6/6] Gerando figuras científicas...")
    
    # Figura 1: Distribuição de theta(x) por Domínio (Modelo C vs D)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    dom_names = ["WikiText", "GSM8K", "Code"]
    colors = ["#2b5c8f", "#2ca02c", "#d62728"]

    # Subplot 1: Modelo C (33 params)
    ax = axes[0]
    data_c = [results["domain_theta"][d]["C"]["values"] for d in dom_names]
    vp1 = ax.violinplot(data_c, showmeans=True, showextrema=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(dom_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Orientação Funcional predita $\\theta(x)$ (graus)", fontsize=11)
    ax.set_title("Modelo C: Gate em $a = h U_c \\in \\mathbb{R}^{32}$ (33 params)", fontsize=12, fontweight="bold")
    ax.axhline(155.25, color="#2b5c8f", linestyle="--", alpha=0.5, label="Alvo Wiki (155.25°)")
    ax.axhline(163.75, color="#2ca02c", linestyle="--", alpha=0.5, label="Alvo GSM (163.75°)")
    ax.axhline(131.50, color="#d62728", linestyle="--", alpha=0.5, label="Alvo Code (131.50°)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)

    # Subplot 2: Modelo D (5121 params)
    ax = axes[1]
    data_d = [results["domain_theta"][d]["D"]["values"] for d in dom_names]
    vp2 = ax.violinplot(data_d, showmeans=True, showextrema=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(dom_names, fontsize=11, fontweight="bold")
    ax.set_title("Modelo D: Gate em $h \\in \\mathbb{R}^{5120}$ (5121 params)", fontsize=12, fontweight="bold")
    ax.axhline(155.25, color="#2b5c8f", linestyle="--", alpha=0.5)
    ax.axhline(163.75, color="#2ca02c", linestyle="--", alpha=0.5)
    ax.axhline(131.50, color="#d62728", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.25)

    plt.suptitle("Ciclo 19: Distribuição Dinâmica de Coordenadas $\\theta(x)$ em Dados Held-Out", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(repo_root, "experiments", "gating_theta_distribution.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.savefig("gating_theta_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 1 salva: {fig1_path}")

    # Figura 2: Correlação Predita vs Ground-Truth theta*(x)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    dom_color_map = {"WikiText": "#2b5c8f", "Code": "#d62728", "GSM8K": "#2ca02c"}
    colors_seqs = [dom_color_map[l] for l in domain_labels]

    # Modelo C Scatter
    ax = axes[0]
    for dom in ["WikiText", "Code", "GSM8K"]:
        idx = [i for i, l in enumerate(domain_labels) if l == dom]
        ax.scatter(all_theta_star[idx], all_theta_pred_c[idx], color=dom_color_map[dom], label=dom, s=65, alpha=0.85, edgecolors="k")
    # Linha ideal y = x
    min_v = min(all_theta_star.min(), all_theta_pred_c.min()) - 5
    max_v = max(all_theta_star.max(), all_theta_pred_c.max()) + 5
    ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.4, label="Ideal $\\hat{\\theta} = \\theta^*$")
    # Regressão Linear
    m, b = np.polyfit(all_theta_star, all_theta_pred_c, 1)
    ax.plot(all_theta_star, m * all_theta_star + b, color="#0055ff", alpha=0.8, label=f"Regressão (r={corr_c:.3f})")
    ax.set_xlabel("Ótimo Empírico $\\theta^*(x)$ (graus)", fontsize=11)
    ax.set_ylabel("Predição $\\hat{\\theta}(x)$ Modelo C (graus)", fontsize=11)
    ax.set_title(f"Modelo C (33 params): r = {corr_c:.3f}, MAE = {mae_c:.2f}°", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    # Modelo D Scatter
    ax = axes[1]
    for dom in ["WikiText", "Code", "GSM8K"]:
        idx = [i for i, l in enumerate(domain_labels) if l == dom]
        ax.scatter(all_theta_star[idx], all_theta_pred_d[idx], color=dom_color_map[dom], label=dom, s=65, alpha=0.85, edgecolors="k")
    min_v = min(all_theta_star.min(), all_theta_pred_d.min()) - 5
    max_v = max(all_theta_star.max(), all_theta_pred_d.max()) + 5
    ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.4, label="Ideal $\\hat{\\theta} = \\theta^*$")
    m, b = np.polyfit(all_theta_star, all_theta_pred_d, 1)
    ax.plot(all_theta_star, m * all_theta_star + b, color="#ff5500", alpha=0.8, label=f"Regressão (r={corr_d:.3f})")
    ax.set_xlabel("Ótimo Empírico $\\theta^*(x)$ (graus)", fontsize=11)
    ax.set_ylabel("Predição $\\hat{\\theta}(x)$ Modelo D (graus)", fontsize=11)
    ax.set_title(f"Modelo D (5121 params): r = {corr_d:.3f}, MAE = {mae_d:.2f}°", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    plt.suptitle("Validação em Dados Held-Out: Capacidade Preditiva do Micro-Gating Angular", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = os.path.join(repo_root, "experiments", "gating_prediction_correlation.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.savefig("gating_prediction_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Figura 2 salva: {fig2_path}")

    total_time = time.time() - t_start
    print(f"\n[SUCESSO] Ciclo 19 finalizado em {total_time:.2f} segundos!")
    print("=" * 110)

if __name__ == "__main__":
    main()
