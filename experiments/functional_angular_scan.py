# -*- coding: utf-8 -*-
"""
EXPERIMENTO DE MAPEAMENTO ANGULAR EM L63: FASES A, B e D
- Varredura 360° (passo 30°, 12 pontos) em 8 planos bidimensionais canônicos de Givens em U_c/U_f (32 canais).
- Forward caching ultrarrápido em GPU (sem backprop).
- Registra PPL(theta), Delta PPL(theta), Top-1, KL, Cosine Sim e distancia Grassmanniana.
- Classifica o vale de cada plano e constrói o ranking formal.
- Gera os gráficos: functional_angular_curves.png e functional_angular_plan_ranking.png.
- Salva experiments/functional_angular_results.json.
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

def apply_givens_rotation(U: torch.Tensor, i: int, j: int, theta_rad: float):
    """
    Aplica uma rotação canônica de Givens no plano formado pelas colunas (i, j) de U.
    u_i(theta) =  u_i cos(theta) + u_j sin(theta)
    u_j(theta) = -u_i sin(theta) + u_j cos(theta)
    """
    U_rot = U.clone()
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    ui = U[:, i].clone()
    uj = U[:, j].clone()
    U_rot[:, i] = c * ui + s * uj
    U_rot[:, j] = -s * ui + c * uj
    return U_rot

def compute_grassmann_distance(U1: torch.Tensor, U2: torch.Tensor):
    """
    Distância geodésica no Grassmanniano Gr(r, d):
    d_Gr(U1, U2) = sqrt(sum_k theta_k^2)
    """
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    thetas = torch.acos(S) # radianos
    return torch.norm(thetas).item()

def classify_curve(angles_deg, ppl_vals):
    p_min = min(ppl_vals)
    p_max = max(ppl_vals)
    delta = p_max - p_min
    if delta < 0.15:
        return "Aproximadamente Plana"
    min_idx = int(np.argmin(ppl_vals))
    # Testar se há múltiplos vales locais
    valleys = 0
    N = len(ppl_vals)
    for idx in range(N):
        prev_v = ppl_vals[(idx - 1) % N]
        next_v = ppl_vals[(idx + 1) % N]
        curr_v = ppl_vals[idx]
        if curr_v < prev_v and curr_v < next_v:
            valleys += 1
    if valleys == 1:
        return "Vale Único"
    elif valleys > 1:
        return f"Múltiplos Vales ({valleys})"
    return "Comportamento Monotônico / Irregular"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("CICLO DE ENGENHARIA: MAPEAMENTO ANGULAR DO NUCLEO COMPARTILHADO EM L63 (FASES A, B, D)")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Baseline Canônico SVD-256 Uniforme: PPL = 192.68")
    print("Pure Universal U_c(32) (Doc 15):     PPL = 186.63")
    print("Melhor Atual U_c Q (Doc 16):         PPL = 186.55")
    print("Recorde L63 Monolitico (Doc 13):      PPL = 179.94")

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

    # Forward da camada 63 do Atlas para obter h_63_raw congelado
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

    # 2. Carregar a Base Funcional U_f(32) e Calibrar o Peso W de Referência no Ponto 0
    func_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    saved_func = torch.load(func_path, map_location=device)
    U_base = saved_func["U_f"].to(device=device, dtype=torch.bfloat16) # (5120, 32)
    print(f"  -> U_base (32 canais) carregado: shape = {U_base.shape}")

    # Calibração do peso ótimo W* em theta = 0
    print("\n[Passo 1/4] Calibrando vetor de pesos de subida W* de referência em theta = 0...")
    W_opt = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([W_opt], lr=1e-4, weight_decay=1e-3)
    h_tr = h_63_raw[:num_train]
    corr_tr = torch.matmul(torch.matmul(h_tr, U_base), W_opt)
    l_tr = F.linear(norm_mod(h_tr + corr_tr), lm_head_w)
    s_flat = l_tr[:, :-1, :].reshape(-1, vocab_sz)
    t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
    y_flat = train_targets.reshape(-1)

    loss = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean") + \
           0.1 * F.cross_entropy(s_flat.float(), y_flat, reduction="mean") + \
           0.05 * F.mse_loss((h_tr + corr_tr).reshape(-1, 5120).float(), y_tr_63.float())
    loss.backward()
    optimizer.step()

    # Avaliar no ponto inicial theta = 0
    h_te = h_63_raw[num_train:]
    with torch.no_grad():
        corr_0 = torch.matmul(torch.matmul(h_te, U_base), W_opt)
        l_te_0 = F.linear(norm_mod(h_te + corr_0), lm_head_w)
        m_0 = compute_metrics(l_te_0, test_targets, teacher_logits=logits_prof_test)

    ppl_ref = m_0["ppl"]
    print(f"  -> Ponto de Referência PPL(0): {ppl_ref:.2f} | Top-1: {m_0['top1']:.2f}% | KL: {m_0['kl_div']:.4f}")
    print("-" * 85)

    # 3. Definir os 8 Planos Canônicos 2D de Givens a Mapear
    # Índices 0-based: u_4 -> 3, u_10 -> 9, u_5 -> 4, u_3 -> 2, u_7 -> 6, u_1 -> 0, u_2 -> 1
    candidate_planes = [
        {"name": "(u_4, u_10)", "i": 3, "j": 9, "desc": "Top 1 e Top 2 de Alavancagem"},
        {"name": "(u_4, u_5)",  "i": 3, "j": 4, "desc": "Top 1 e Top 3 de Alavancagem"},
        {"name": "(u_10, u_5)", "i": 9, "j": 4, "desc": "Top 2 e Top 3 de Alavancagem"},
        {"name": "(u_3, u_4)",  "i": 2, "j": 3, "desc": "Top 4 e Top 1 de Alavancagem"},
        {"name": "(u_3, u_10)", "i": 2, "j": 9, "desc": "Top 4 e Top 2 de Alavancagem"},
        {"name": "(u_5, u_7)",  "i": 4, "j": 6, "desc": "Top 3 e Top 5 de Alavancagem"},
        {"name": "(u_1, u_4)",  "i": 0, "j": 3, "desc": "Máxima Energia (u1) vs Máxima Alavancagem (u4)"},
        {"name": "(u_1, u_2)",  "i": 0, "j": 1, "desc": "Dois Modos Dominantes do Erro Euclidiano"}
    ]

    angles_deg_scan = list(range(0, 360, 30)) # 12 pontos: 0, 30, 60, ..., 330
    scan_results = {}
    plan_summaries = []

    print("\n[Passo 2/4] Executando Varredura 360° (passo 30°, 12 pontos por plano)...")

    for p_idx, plan in enumerate(candidate_planes, 1):
        p_name = plan["name"]
        i, j = plan["i"], plan["j"]
        print(f"\n--- Plano {p_idx}/8: {p_name} [{plan['desc']}] ---")

        curve_records = []
        for deg in angles_deg_scan:
            theta_rad = deg * (math.pi / 180.0)
            U_rot = apply_givens_rotation(U_base, i, j, theta_rad)

            # Verificar ortonormalidade
            orth_err = torch.dist(torch.matmul(U_rot.t(), U_rot), torch.eye(32, device=device)).item()

            # Forward pass cached ultrarrápido
            with torch.no_grad():
                corr_rot = torch.matmul(torch.matmul(h_te, U_rot), W_opt)
                l_rot = F.linear(norm_mod(h_te + corr_rot), lm_head_w)
                m_rot = compute_metrics(l_rot, test_targets, teacher_logits=logits_prof_test)

            d_gr = compute_grassmann_distance(U_base, U_rot)
            delta_ppl = m_rot["ppl"] - ppl_ref

            curve_records.append({
                "theta_deg": deg,
                "ppl": m_rot["ppl"],
                "delta_ppl": delta_ppl,
                "nll": m_rot["nll"],
                "top1": m_rot["top1"],
                "kl": m_rot["kl_div"],
                "cos_sim": m_rot["cos_sim"],
                "d_gr": d_gr,
                "orth_err": orth_err
            })

        # Checar periodicidade em 360°
        U_360 = apply_givens_rotation(U_base, i, j, 2.0 * math.pi)
        with torch.no_grad():
            corr_360 = torch.matmul(torch.matmul(h_te, U_360), W_opt)
            l_360 = F.linear(norm_mod(h_te + corr_360), lm_head_w)
            m_360 = compute_metrics(l_360, test_targets)
        period_diff = abs(m_360["ppl"] - ppl_ref)

        # FASE B: Encontrar o Melhor Vale
        ppl_vals = [r["ppl"] for r in curve_records]
        min_idx = int(np.argmin(ppl_vals))
        best_rec = curve_records[min_idx]
        curve_class = classify_curve(angles_deg_scan, ppl_vals)

        print(f"  -> Melhor Ângulo theta*: {best_rec['theta_deg']:3d} deg | PPL*: {best_rec['ppl']:.2f} (Delta: {best_rec['delta_ppl']:+6.2f}) | Top-1: {best_rec['top1']:.2f}% | Perfil: {curve_class} | Period diff: {period_diff:.2e}")

        summary = {
            "plane": p_name,
            "desc": plan["desc"],
            "best_theta": best_rec["theta_deg"],
            "best_ppl": best_rec["ppl"],
            "delta_ppl": best_rec["delta_ppl"],
            "top1": best_rec["top1"],
            "kl": best_rec["kl"],
            "d_gr": best_rec["d_gr"],
            "curve_class": curve_class,
            "period_diff": period_diff
        }
        plan_summaries.append(summary)
        scan_results[p_name] = {
            "meta": plan,
            "summary": summary,
            "curve": curve_records
        }

    # 4. FASE D: Ranking Formal dos Planos
    print("\n" + "=" * 115)
    print("FASE D: RANKING FORMAL DOS PLANOS CANONICOS 2D DE GIVENS")
    print("=" * 115)
    print(f"{'Ranking':<8} | {'Plano':<16} | {'Melhor theta*':<14} | {'PPL Min':<10} | {'Delta PPL':<14} | {'Top-1':<10} | {'Classificação'}")
    print("-" * 115)

    ranked_plans = sorted(plan_summaries, key=lambda x: x["best_ppl"])
    for rank, p in enumerate(ranked_plans, 1):
        print(f"{rank:<8} | {p['plane']:<16} | {p['best_theta']:>3d} deg{'':<8} | {p['best_ppl']:<10.2f} | {p['delta_ppl']:+6.2f} pts{'':<6} | {p['top1']:<9.2f}% | {p['curve_class']}")
    print("-" * 115)

    best_overall_plan = ranked_plans[0]
    print(f"\n[DESTAQUE] Plano Mais Promissor: {best_overall_plan['plane']} em theta* = {best_overall_plan['best_theta']} deg (PPL = {best_overall_plan['best_ppl']:.2f}, Delta = {best_overall_plan['delta_ppl']:+6.2f})")

    # 5. Salvar JSON de Resultados
    json_path = os.path.join(repo_root, "experiments", "functional_angular_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "ppl_reference": ppl_ref,
            "ranking": ranked_plans,
            "plans_data": scan_results
        }, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {json_path}")

    # 6. Salvar Gráfico 1: functional_angular_curves.png (8 trajetórias de PPL)
    print("\n[Passo 4/4] Gerando gráficos publicáveis...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.suptitle("Mapeamento Angular 360° em Planos Canônicos 2D de Givens em L63", fontsize=16, fontweight="bold")

    for idx, plan in enumerate(candidate_planes):
        ax = axes[idx // 4, idx % 4]
        p_name = plan["name"]
        rec = scan_results[p_name]["curve"]
        th_plot = [r["theta_deg"] for r in rec]
        ppl_plot = [r["ppl"] for r in rec]

        ax.plot(th_plot, ppl_plot, marker="o", lw=2, color="tab:blue", label="PPL(theta)")
        ax.axhline(ppl_ref, color="red", linestyle="--", label=f"Ref (theta=0: {ppl_ref:.2f})")
        
        b_th = scan_results[p_name]["summary"]["best_theta"]
        b_p = scan_results[p_name]["summary"]["best_ppl"]
        ax.scatter([b_th], [b_p], color="green", s=120, zorder=5, label=f"Min: {b_p:.2f} ({b_th}°)")

        ax.set_title(f"{p_name}\n[{plan['desc']}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Ângulo theta (graus)", fontweight="bold")
        ax.set_ylabel("Perplexidade (PPL)", fontweight="bold")
        ax.set_xticks(range(0, 361, 60))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    chart1_path = os.path.join(repo_root, "experiments", "functional_angular_curves.png")
    fig.savefig(chart1_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "functional_angular_curves.png"), dpi=300)
    plt.close(fig)
    print(f"  [OK] Salvo: {chart1_path} e raiz do repositório.")

    # 7. Salvar Gráfico 2: functional_angular_plan_ranking.png (ranking de ganhos)
    fig2, ax_rk = plt.subplots(figsize=(12, 6))
    plan_names = [p["plane"] for p in ranked_plans]
    deltas = [p["delta_ppl"] for p in ranked_plans]
    colors_rk = ["tab:green" if d < 0 else "tab:red" for d in deltas]

    bars = ax_rk.barh(range(len(plan_names)), deltas, color=colors_rk, alpha=0.85, edgecolor="k")
    ax_rk.set_yticks(range(len(plan_names)))
    ax_rk.set_yticklabels(plan_names, fontsize=11, fontweight="bold")
    ax_rk.axvline(0.0, color="black", linestyle="-", lw=1)
    ax_rk.set_xlabel("Delta PPL vs Referência (theta = 0)", fontsize=12, fontweight="bold")
    ax_rk.set_title("Ranking dos Planos Canônicos 2D por Ganho de PPL (Menor é Melhor)", fontsize=14, fontweight="bold")
    ax_rk.invert_yaxis()
    ax_rk.grid(True, alpha=0.3)

    for bar, p in zip(bars, ranked_plans):
        w = bar.get_width()
        x_pos = w - 0.05 if w < 0 else w + 0.05
        ha = "right" if w < 0 else "left"
        ax_rk.text(x_pos, bar.get_y() + bar.get_height()/2.0, f"Delta: {p['delta_ppl']:+.2f} (PPL: {p['best_ppl']:.2f} em {p['best_theta']}°)", ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart2_path = os.path.join(repo_root, "experiments", "functional_angular_plan_ranking.png")
    fig2.savefig(chart2_path, dpi=300)
    fig2.savefig(os.path.join(repo_root, "functional_angular_plan_ranking.png"), dpi=300)
    plt.close(fig2)
    print(f"  [OK] Salvo: {chart2_path} e raiz do repositório.")
    print("=" * 115)

if __name__ == "__main__":
    main()
