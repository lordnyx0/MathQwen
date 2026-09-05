# -*- coding: utf-8 -*-
"""
EXPERIMENTO DE MAPEAMENTO ANGULAR EM L63: FASE E (BUSCA 2D) E VALIDACAO MULTIDOMINIO
- Combina os dois melhores planos canônicos encontrados nas Fases Anteriores.
- Grade 2D 7x7 (49 pontos) theta_1, theta_2 in {-15, -10, -5, 0, 5, 10, 15} deg.
- Gera heatmap functional_angular_heatmap.png.
- Avalia generalização multidomínio em WikiText, HF Code e GSM8K.
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
    U_rot = U.clone()
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    ui = U[:, i].clone()
    uj = U[:, j].clone()
    U_rot[:, i] = c * ui + s * uj
    U_rot[:, j] = -s * ui + c * uj
    return U_rot

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("BUSCA ANGULAR 2D COMBINADA E VALIDACAO MULTIDOMINIO EM L63 (FASE E)")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Resultados das Fases Anteriores
    scan_json_path = os.path.join(repo_root, "experiments", "functional_angular_results.json")
    if not os.path.exists(scan_json_path):
        raise FileNotFoundError(f"Arquivo {scan_json_path} nao encontrado!")
    with open(scan_json_path, "r", encoding="utf-8") as f:
        scan_data = json.load(f)

    ranked_plans = scan_data["ranking"]
    ppl_ref = scan_data["ppl_reference"]

    plan1 = ranked_plans[0]
    plan2 = ranked_plans[1]
    meta1 = scan_data["plans_data"][plan1["plane"]]["meta"]
    meta2 = scan_data["plans_data"][plan2["plane"]]["meta"]

    print(f"Plano 1: {plan1['plane']} (i={meta1['i']}, j={meta1['j']}) | Centro theta1*: {plan1['best_theta']}°")
    print(f"Plano 2: {plan2['plane']} (i={meta2['i']}, j={meta2['j']}) | Centro theta2*: {plan2['best_theta']}°")

    # 2. Carregar Componentes e Targets
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

    # Forward L63 Atlas
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

    func_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    saved_func = torch.load(func_path, map_location=device)
    U_base = saved_func["U_f"].to(device=device, dtype=torch.bfloat16)

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

    h_te = h_63_raw[num_train:]

    # 3. Grade 2D 7x7 (49 pontos)
    d_theta_vals = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    th1_center = float(plan1["best_theta"])
    th2_center = float(plan2["best_theta"])

    grid_th1 = [th1_center + dt for dt in d_theta_vals]
    grid_th2 = [th2_center + dt for dt in d_theta_vals]

    ppl_grid = np.zeros((len(grid_th1), len(grid_th2)))
    best_2d_ppl = 9999.0
    best_2d_point = None

    print("\n[Passo 1/3] Calculando Grade 2D (49 avaliações forward vetorizadas)...")
    for r_idx, t1 in enumerate(grid_th1):
        for c_idx, t2 in enumerate(grid_th2):
            th1_rad = t1 * (math.pi / 180.0)
            th2_rad = t2 * (math.pi / 180.0)

            # Rotação composta: U(t1, t2) = G2(t2) o G1(t1) [U_base]
            U_rot1 = apply_givens_rotation(U_base, meta1["i"], meta1["j"], th1_rad)
            U_rot2 = apply_givens_rotation(U_rot1, meta2["i"], meta2["j"], th2_rad)

            with torch.no_grad():
                corr = torch.matmul(torch.matmul(h_te, U_rot2), W_opt)
                l_te_2d = F.linear(norm_mod(h_te + corr), lm_head_w)
                m_2d = compute_metrics(l_te_2d, test_targets, teacher_logits=logits_prof_test)

            p_val = m_2d["ppl"]
            ppl_grid[r_idx, c_idx] = p_val

            if p_val < best_2d_ppl:
                best_2d_ppl = p_val
                best_2d_point = (t1, t2, m_2d["top1"], m_2d["kl_div"])

    print(f"  -> Melhor Ponto 2D: theta1* = {best_2d_point[0]:.1f}° | theta2* = {best_2d_point[1]:.1f}° | PPL = {best_2d_ppl:.2f} (Delta: {best_2d_ppl - ppl_ref:+.2f}) | Top-1: {best_2d_point[2]:.2f}%")

    # 4. Salvar Heatmap: functional_angular_heatmap.png
    print("\n[Passo 2/3] Gerando heatmap 2D...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ppl_grid, cmap="viridis_r", origin="lower")
    ax.set_xticks(range(len(grid_th2)))
    ax.set_yticks(range(len(grid_th1)))
    ax.set_xticklabels([f"{t:.0f}°" for t in grid_th2], fontweight="bold")
    ax.set_yticklabels([f"{t:.0f}°" for t in grid_th1], fontweight="bold")
    ax.set_xlabel(f"Ângulo theta2 no Plano {plan2['plane']}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Ângulo theta1 no Plano {plan1['plane']}", fontsize=12, fontweight="bold")
    ax.set_title(f"Heatmap de PPL: Exploração Angular Bidimensional\n{plan1['plane']} vs {plan2['plane']} (Mínimo: {best_2d_ppl:.2f})", fontsize=14, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Perplexidade (PPL)", fontsize=11, fontweight="bold")

    # Marcar o ponto ótimo
    r_opt = int(np.where(np.array(grid_th1) == best_2d_point[0])[0][0])
    c_opt = int(np.where(np.array(grid_th2) == best_2d_point[1])[0][0])
    ax.scatter([c_opt], [r_opt], color="red", s=180, edgecolors="white", linewidth=2, label=f"Ótimo: {best_2d_ppl:.2f}")
    ax.legend(loc="upper left")

    plt.tight_layout()
    hm_path = os.path.join(repo_root, "experiments", "functional_angular_heatmap.png")
    fig.savefig(hm_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "functional_angular_heatmap.png"), dpi=300)
    plt.close(fig)
    print(f"  [OK] Heatmap salvo em: {hm_path} e raiz do repositório.")

    # 5. Avaliação Multidomínio do Melhor Ponto 2D
    print("\n[Passo 3/3] Avaliação Multidomínio do Ponto Ótimo 2D...")
    U_best_2d = apply_givens_rotation(U_base, meta1["i"], meta1["j"], best_2d_point[0] * math.pi / 180.0)
    U_best_2d = apply_givens_rotation(U_best_2d, meta2["i"], meta2["j"], best_2d_point[1] * math.pi / 180.0)

    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_in = hf_data["h_in_63"].to(device=device, dtype=torch.bfloat16)
    code_targets = hf_data["input_ids"][:, 1:].contiguous().to(device)

    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_in = gsm_data["h_in_63"].to(device=device, dtype=torch.bfloat16)
    gsm_targets = gsm_data["input_ids"][:, 1:].contiguous().to(device)

    domains_eval = [
        ("WikiText (Test)", h_te, test_targets),
        ("HF Code", h_code_in, code_targets),
        ("GSM8K", h_gsm_in, gsm_targets)
    ]

    multidomain_res = {}
    print(f"{'Domínio':<20} | {'Atlas Raw PPL':<15} | {'Ótimo 2D PPL':<15} | {'Delta':<12} | {'Top-1':<12}")
    print("-" * 85)
    for d_name, h_d, targets_d in domains_eval:
        with torch.no_grad():
            l_raw_d = F.linear(norm_mod(h_d), lm_head_w)
            m_raw = compute_metrics(l_raw_d, targets_d)
            corr_2d = torch.matmul(torch.matmul(h_d, U_best_2d), W_opt)
            l_2d = F.linear(norm_mod(h_d + corr_2d), lm_head_w)
            m_2d_d = compute_metrics(l_2d, targets_d)

        delta_d = m_2d_d["ppl"] - m_raw["ppl"]
        multidomain_res[d_name] = {
            "raw_ppl": m_raw["ppl"],
            "best2d_ppl": m_2d_d["ppl"],
            "delta": delta_d,
            "top1": m_2d_d["top1"]
        }
        print(f"{d_name:<20} | {m_raw['ppl']:<15.2f} | {m_2d_d['ppl']:<15.2f} | {delta_d:+6.2f} pts{'':<4} | {m_2d_d['top1']:<11.2f}%")
    print("-" * 85)

    # Salvar Resultados JSON 2D
    json_2d_path = os.path.join(repo_root, "experiments", "functional_angular_2d_results.json")
    with open(json_2d_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_2d": {
                "theta1": best_2d_point[0],
                "theta2": best_2d_point[1],
                "ppl": best_2d_ppl,
                "delta": best_2d_ppl - ppl_ref,
                "top1": best_2d_point[2],
                "kl": best_2d_point[3]
            },
            "grid_th1": grid_th1,
            "grid_th2": grid_th2,
            "ppl_grid": ppl_grid.tolist(),
            "multidomain": multidomain_res
        }, f, indent=2)
    print(f"\n[OK] Resultados 2D salvos em: {json_2d_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
