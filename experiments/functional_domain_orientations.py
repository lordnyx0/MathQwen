# -*- coding: utf-8 -*-
"""
EXPERIMENTO DE ORIENTAÇÕES FUNCIONAIS MULTIDOMÍNIO EM L63
- Mapeamento angular 360° em WikiText, HF Code e GSM8K nos 8 planos canônicos de U_c(32).
- Determinação precisa dos ângulos ótimos theta*_Wiki, theta*_Code, theta*_GSM.
- Cálculo de distâncias geodésicas de Grassmann d_Gr(U_Wiki, U_Code), d_Gr(U_Wiki, U_GSM), d_Gr(U_Code, U_GSM).
- Matriz de transferência cruzada 3x3 entre domínios.
- Geração de gráficos: domain_angular_trajectories.png e domain_grassmann_matrix.png.
- Salva experiments/domain_orientations_results.json.
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

def compute_canonical_angles(U1: torch.Tensor, U2: torch.Tensor):
    """
    Retorna os ângulos canônicos/principais em radianos e graus entre U1 e U2.
    """
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    thetas_rad = torch.acos(S)
    thetas_deg = thetas_rad * (180.0 / math.pi)
    return thetas_rad, thetas_deg

def compute_grassmann_metrics(U1: torch.Tensor, U2: torch.Tensor):
    thetas_rad, thetas_deg = compute_canonical_angles(U1, U2)
    d_geo_rad = torch.norm(thetas_rad).item()
    d_geo_deg = d_geo_rad * (180.0 / math.pi)
    d_chord = torch.norm(torch.sin(thetas_rad)).item()
    max_angle_deg = torch.max(thetas_deg).item()
    mean_cos = torch.mean(torch.cos(thetas_rad)).item()
    return {
        "d_geodesic_rad": d_geo_rad,
        "d_geodesic_deg": d_geo_deg,
        "d_chord": d_chord,
        "max_angle_deg": max_angle_deg,
        "mean_cosine_overlap": mean_cos
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("CICLO DE ENGENHARIA: MAPEAMENTO ANGULAR MULTIDOMÍNIO E GEOMETRIA DE GRASSMANN EM L63")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    snapshot_dir = get_snapshot_dir()

    # 1. Carregar Dados de Avaliação dos 3 Domínios
    print("\n[Passo 1/6] Carregando ativações e targets dos 3 domínios (WikiText, HF Code, GSM8K)...")
    
    # WikiText
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

    # Forward L63 Atlas para WikiText
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

    h_tr = h_63_raw[:num_train]
    h_te_wiki = h_63_raw[num_train:]

    # HF Code
    hf_data = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location=device)
    h_code_in = hf_data["h_in_63"].to(device=device, dtype=torch.bfloat16)
    code_targets = hf_data["input_ids"][:, 1:].contiguous().to(device)

    # GSM8K
    gsm_data = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location=device)
    h_gsm_in = gsm_data["h_in_63"].to(device=device, dtype=torch.bfloat16)
    gsm_targets = gsm_data["input_ids"][:, 1:].contiguous().to(device)

    print(f"  WikiText Test:  {h_te_wiki.shape}  | Targets: {test_targets.shape}")
    print(f"  HF Code:        {h_code_in.shape} | Targets: {code_targets.shape}")
    print(f"  GSM8K:          {h_gsm_in.shape}  | Targets: {gsm_targets.shape}")

    # 2. Carregar a Base Funcional U_base(32) e Peso W_opt Compartilhado
    print("\n[Passo 2/6] Carregando base funcional U_c(32) e calibrando refinador base W*...")
    func_path = os.path.join(repo_root, "checkpoints", "Uc_functional_r32.pt")
    saved_func = torch.load(func_path, map_location=device)
    U_base = saved_func["U_f"].to(device=device, dtype=torch.bfloat16)

    W_opt = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([W_opt], lr=1e-4, weight_decay=1e-3)
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

    # Baselines em theta = 0
    domains = {
        "WikiText": {"h": h_te_wiki, "targets": test_targets, "teacher": logits_prof_test},
        "HF_Code": {"h": h_code_in, "targets": code_targets, "teacher": None},
        "GSM8K": {"h": h_gsm_in, "targets": gsm_targets, "teacher": None}
    }

    base_metrics = {}
    raw_metrics = {}
    print("\n--- Baselines Iniciais (Sem Refinador vs Refinador em theta = 0) ---")
    print(f"{'Domínio':<15} | {'Raw PPL':<15} | {'PPL(0)':<15} | {'Delta PPL(0)':<15} | {'Top-1(0)':<12}")
    print("-" * 80)
    for d_name, d_info in domains.items():
        with torch.no_grad():
            l_raw = F.linear(norm_mod(d_info["h"]), lm_head_w)
            m_raw = compute_metrics(l_raw, d_info["targets"], d_info["teacher"])
            corr_0 = torch.matmul(torch.matmul(d_info["h"], U_base), W_opt)
            l_0 = F.linear(norm_mod(d_info["h"] + corr_0), lm_head_w)
            m_0 = compute_metrics(l_0, d_info["targets"], d_info["teacher"])
        raw_metrics[d_name] = m_raw
        base_metrics[d_name] = m_0
        delta_0 = m_0["ppl"] - m_raw["ppl"]
        print(f"{d_name:<15} | {m_raw['ppl']:<15.2f} | {m_0['ppl']:<15.2f} | {delta_0:+14.2f} pts | {m_0['top1']:<11.2f}%")
    print("-" * 80)

    # 3. Varredura Angular Multidomínio 360° nos 8 Planos Canônicos (Resolução 10°, 36 pontos)
    candidate_planes = [
        {"name": "(u_4, u_10)", "i": 3, "j": 9, "desc": "Top 1 e Top 2 de Alavancagem"},
        {"name": "(u_4, u_5)",  "i": 3, "j": 4, "desc": "Top 1 e Top 3 de Alavancagem"},
        {"name": "(u_10, u_5)", "i": 9, "j": 4, "desc": "Top 2 e Top 3 de Alavancagem"},
        {"name": "(u_3, u_4)",  "i": 2, "j": 3, "desc": "Top 4 e Top 1 de Alavancagem"},
        {"name": "(u_3, u_10)", "i": 2, "j": 9, "desc": "Top 4 e Top 2 de Alavancagem (Ótimo Wiki)"},
        {"name": "(u_5, u_7)",  "i": 4, "j": 6, "desc": "Top 3 e Top 5 de Alavancagem"},
        {"name": "(u_1, u_4)",  "i": 0, "j": 3, "desc": "Máxima Energia vs Máxima Alavancagem"},
        {"name": "(u_1, u_2)",  "i": 0, "j": 1, "desc": "Modos de Máxima Energia Euclidiana"}
    ]

    angles_scan = list(range(0, 360, 10)) # 36 pontos
    scan_results = {}
    domain_plane_optima = {d: {} for d in domains.keys()}

    print(f"\n[Passo 3/6] Varrendo 360° (passo 10°, 36 pontos) em 8 planos para todos os 3 domínios...")
    t_scan0 = time.time()

    for p_idx, plan in enumerate(candidate_planes, 1):
        p_name = plan["name"]
        i, j = plan["i"], plan["j"]
        scan_results[p_name] = {
            "meta": plan,
            "angles": angles_scan,
            "WikiText": {"ppl": [], "top1": []},
            "HF_Code": {"ppl": [], "top1": []},
            "GSM8K": {"ppl": [], "top1": []}
        }

        best_pts = {d: {"theta": 0, "ppl": 99999.0, "top1": 0.0} for d in domains.keys()}

        for deg in angles_scan:
            th_rad = deg * (math.pi / 180.0)
            U_rot = apply_givens_rotation(U_base, i, j, th_rad)

            with torch.no_grad():
                # WikiText
                corr_w = torch.matmul(torch.matmul(h_te_wiki, U_rot), W_opt)
                l_w = F.linear(norm_mod(h_te_wiki + corr_w), lm_head_w)
                mw = compute_metrics(l_w, test_targets, logits_prof_test)
                scan_results[p_name]["WikiText"]["ppl"].append(mw["ppl"])
                scan_results[p_name]["WikiText"]["top1"].append(mw["top1"])
                if mw["ppl"] < best_pts["WikiText"]["ppl"]:
                    best_pts["WikiText"] = {"theta": deg, "ppl": mw["ppl"], "top1": mw["top1"]}

                # HF Code
                corr_c = torch.matmul(torch.matmul(h_code_in, U_rot), W_opt)
                l_c = F.linear(norm_mod(h_code_in + corr_c), lm_head_w)
                mc = compute_metrics(l_c, code_targets)
                scan_results[p_name]["HF_Code"]["ppl"].append(mc["ppl"])
                scan_results[p_name]["HF_Code"]["top1"].append(mc["top1"])
                if mc["ppl"] < best_pts["HF_Code"]["ppl"]:
                    best_pts["HF_Code"] = {"theta": deg, "ppl": mc["ppl"], "top1": mc["top1"]}

                # GSM8K
                corr_g = torch.matmul(torch.matmul(h_gsm_in, U_rot), W_opt)
                l_g = F.linear(norm_mod(h_gsm_in + corr_g), lm_head_w)
                mg = compute_metrics(l_g, gsm_targets)
                scan_results[p_name]["GSM8K"]["ppl"].append(mg["ppl"])
                scan_results[p_name]["GSM8K"]["top1"].append(mg["top1"])
                if mg["ppl"] < best_pts["GSM8K"]["ppl"]:
                    best_pts["GSM8K"] = {"theta": deg, "ppl": mg["ppl"], "top1": mg["top1"]}

        for d in domains.keys():
            domain_plane_optima[d][p_name] = best_pts[d]

        print(f"  Plano {p_idx}/8: {p_name:<12} | "
              f"Wiki: th*={best_pts['WikiText']['theta']:3d}° (PPL {best_pts['WikiText']['ppl']:.2f}) | "
              f"Code: th*={best_pts['HF_Code']['theta']:3d}° (PPL {best_pts['HF_Code']['ppl']:.2f}) | "
              f"GSM: th*={best_pts['GSM8K']['theta']:3d}° (PPL {best_pts['GSM8K']['ppl']:.2f})")

    print(f"[OK] Varredura 360° concluída em {time.time()-t_scan0:.2f}s.")

    # 4. Refinamento de Alta Precisão (0.25°) nos Planos Prioritários: (u3, u10) e (u10, u5)
    print("\n[Passo 4/6] Refinamento fino (0.25°) para cada domínio nos planos chave...")
    priority_planes = ["(u_3, u_10)", "(u_10, u_5)"]
    refined_domain_optima = {}

    for d_name in domains.keys():
        refined_domain_optima[d_name] = {}
        for p_name in priority_planes:
            plan_meta = next(p for p in candidate_planes if p["name"] == p_name)
            i, j = plan_meta["i"], plan_meta["j"]
            initial_best_th = domain_plane_optima[d_name][p_name]["theta"]

            # Refinamento Nível 1: +/- 10° com passo de 1°
            th_range_1 = [initial_best_th + d for d in range(-10, 11, 1)]
            best_r1 = {"theta": initial_best_th, "ppl": 99999.0, "top1": 0.0}
            for t_deg in th_range_1:
                U_r = apply_givens_rotation(U_base, i, j, t_deg * math.pi / 180.0)
                with torch.no_grad():
                    corr_r = torch.matmul(torch.matmul(domains[d_name]["h"], U_r), W_opt)
                    l_r = F.linear(norm_mod(domains[d_name]["h"] + corr_r), lm_head_w)
                    mr = compute_metrics(l_r, domains[d_name]["targets"], domains[d_name]["teacher"])
                if mr["ppl"] < best_r1["ppl"]:
                    best_r1 = {"theta": t_deg, "ppl": mr["ppl"], "top1": mr["top1"]}

            # Refinamento Nível 2: +/- 2° com passo de 0.25°
            th_range_2 = [best_r1["theta"] + d * 0.25 for d in range(-8, 9)]
            best_r2 = {"theta": best_r1["theta"], "ppl": 99999.0, "top1": 0.0}
            for t_deg in th_range_2:
                U_r = apply_givens_rotation(U_base, i, j, t_deg * math.pi / 180.0)
                with torch.no_grad():
                    corr_r = torch.matmul(torch.matmul(domains[d_name]["h"], U_r), W_opt)
                    l_r = F.linear(norm_mod(domains[d_name]["h"] + corr_r), lm_head_w)
                    mr = compute_metrics(l_r, domains[d_name]["targets"], domains[d_name]["teacher"])
                if mr["ppl"] < best_r2["ppl"]:
                    best_r2 = {"theta": t_deg, "ppl": mr["ppl"], "top1": mr["top1"], "kl": mr.get("kl_div", 0.0)}

            refined_domain_optima[d_name][p_name] = best_r2
            print(f"  -> {d_name:<10} no Plano {p_name}: theta* = {best_r2['theta']:6.2f}° | PPL = {best_r2['ppl']:8.2f} (Delta: {best_r2['ppl'] - raw_metrics[d_name]['ppl']:+6.2f}) | Top-1: {best_r2['top1']:.2f}%")

    # 5. Geometria de Grassmann e Distâncias Geodésicas entre Subespaços de Domínio
    print("\n[Passo 5/6] Calculando distâncias geodésicas de Grassmann e métricas de alinhamento...")
    # Consideramos o plano canônico principal (u_3, u_10) onde WikiText atinge seu recorde
    primary_plane = next(p for p in candidate_planes if p["name"] == "(u_3, u_10)")
    pi, pj = primary_plane["i"], primary_plane["j"]

    th_wiki_opt = refined_domain_optima["WikiText"]["(u_3, u_10)"]["theta"]
    th_code_opt = refined_domain_optima["HF_Code"]["(u_3, u_10)"]["theta"]
    th_gsm_opt  = refined_domain_optima["GSM8K"]["(u_3, u_10)"]["theta"]

    U_wiki = apply_givens_rotation(U_base, pi, pj, th_wiki_opt * math.pi / 180.0)
    U_code = apply_givens_rotation(U_base, pi, pj, th_code_opt * math.pi / 180.0)
    U_gsm  = apply_givens_rotation(U_base, pi, pj, th_gsm_opt  * math.pi / 180.0)

    domain_bases = {
        "WikiText": U_wiki,
        "HF_Code": U_code,
        "GSM8K": U_gsm
    }

    # Pares de Domínio
    domain_pairs = [
        ("WikiText", "HF_Code"),
        ("WikiText", "GSM8K"),
        ("HF_Code", "GSM8K")
    ]

    grassmann_results = {}
    print("\n--- Distâncias Geodésicas de Grassmann e Separação Angular no Plano (u_3, u_10) ---")
    print(f"{'Par de Domínios':<25} | {'Delta Theta*':<15} | {'d_Gr(1) Modo Ativo':<20} | {'d_Gr Geodésica':<18} | {'Overlap Cosseno':<15}")
    print("-" * 105)

    for d1, d2 in domain_pairs:
        th1 = refined_domain_optima[d1]["(u_3, u_10)"]["theta"]
        th2 = refined_domain_optima[d2]["(u_3, u_10)"]["theta"]
        delta_th = abs(th1 - th2)
        if delta_th > 180.0:
            delta_th = 360.0 - delta_th

        # Distância de Grassmann entre os modos ativos individuais (Gr(1, 5120))
        u1_active = domain_bases[d1][:, pi]
        u2_active = domain_bases[d2][:, pi]
        cos_active = abs(torch.dot(u1_active.float(), u2_active.float()).item())
        d_gr_mode = math.acos(min(1.0, cos_active)) * (180.0 / math.pi)

        # Distância no espaço 32D com rotação
        # Notar que entre bases truncadas top-k (k=3, contendo u3, mas antes de u10):
        # O subespaço de 3 canais gira exatamente pelo ângulo de Givens!
        U1_k3 = domain_bases[d1][:, :3] # contém u1, u2, u3 rotacionado
        U2_k3 = domain_bases[d2][:, :3]
        gm_k3 = compute_grassmann_metrics(U1_k3, U2_k3)

        # Distância de Lie SO(32)
        d_so32 = math.sqrt(2.0) * (delta_th * math.pi / 180.0)

        grassmann_results[f"{d1}_vs_{d2}"] = {
            "theta1": th1,
            "theta2": th2,
            "delta_theta_deg": delta_th,
            "d_gr_mode1_deg": d_gr_mode,
            "d_gr_subspace_deg": gm_k3["d_geodesic_deg"],
            "d_chord": gm_k3["d_chord"],
            "d_so32_lie": d_so32,
            "mean_cosine": gm_k3["mean_cosine_overlap"]
        }

        print(f"{d1 + ' vs ' + d2:<25} | {delta_th:6.2f}°{'':<8} | {d_gr_mode:6.2f}° ({cos_active:.4f}){'':<6} | {gm_k3['d_geodesic_deg']:6.2f}° ({gm_k3['d_geodesic_rad']:.4f} rad) | {gm_k3['mean_cosine_overlap']:.6f}")

    print("-" * 105)

    # Matriz de Transferência Cruzada 3x3
    print("\n--- Matriz de Transferência Cruzada de Orientação Funcional (PPL) ---")
    print(f"{'Avaliação no Dado':<20} | {'Usando Q_Wiki':<18} | {'Usando Q_Code':<18} | {'Usando Q_GSM':<18} | {'Raw (Sem Corr)':<15}")
    print("-" * 100)

    transfer_matrix = {target_d: {} for target_d in domains.keys()}
    for target_d, d_info in domains.items():
        row_str = f"{target_d:<20} | "
        for source_d, U_src in domain_bases.items():
            with torch.no_grad():
                corr_x = torch.matmul(torch.matmul(d_info["h"], U_src), W_opt)
                l_x = F.linear(norm_mod(d_info["h"] + corr_x), lm_head_w)
                mx = compute_metrics(l_x, d_info["targets"], d_info["teacher"])
            transfer_matrix[target_d][source_d] = {
                "ppl": mx["ppl"],
                "delta": mx["ppl"] - raw_metrics[target_d]["ppl"],
                "top1": mx["top1"]
            }
            # Marcar diagonal em negrito
            mark = "*" if target_d == source_d else " "
            row_str += f"{mx['ppl']:8.2f}{mark} ({mx['ppl'] - raw_metrics[target_d]['ppl']:+6.2f}) | "
        row_str += f"{raw_metrics[target_d]['ppl']:8.2f}"
        print(row_str)
    print("-" * 100)
    print("(* indica orientação ótima própria do domínio)")

    # 6. Gráficos Comparativos
    print("\n[Passo 6/6] Gerando figuras: domain_angular_trajectories.png e domain_grassmann_matrix.png...")
    
    # Figura 1: Trajetórias Angulares Comparativas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    planes_to_plot = ["(u_3, u_10)", "(u_10, u_5)", "(u_4, u_10)", "(u_1, u_2)"]

    for ax_idx, p_plot in enumerate(planes_to_plot):
        r = ax_idx // 2
        c = ax_idx % 2
        ax = axes[r, c]
        p_data = scan_results[p_plot]

        th_axis = p_data["angles"]
        # Normalizar Delta PPL para cada domínio para colocar na mesma escala visual
        w_ppl = np.array(p_data["WikiText"]["ppl"])
        c_ppl = np.array(p_data["HF_Code"]["ppl"])
        g_ppl = np.array(p_data["GSM8K"]["ppl"])

        delta_w = w_ppl - base_metrics["WikiText"]["ppl"]
        delta_c = c_ppl - base_metrics["HF_Code"]["ppl"]
        delta_g = g_ppl - base_metrics["GSM8K"]["ppl"]

        ax.plot(th_axis, delta_w, color="#1f77b4", linewidth=2.5, marker="o", markersize=4, label=f"WikiText (ref: {base_metrics['WikiText']['ppl']:.2f})")
        ax.plot(th_axis, delta_c, color="#d62728", linewidth=2.5, marker="s", markersize=4, label=f"HF Code (ref: {base_metrics['HF_Code']['ppl']:.2f})")
        ax.plot(th_axis, delta_g, color="#2ca02c", linewidth=2.5, marker="^", markersize=4, label=f"GSM8K (ref: {base_metrics['GSM8K']['ppl']:.2f})")

        # Marcar ótimos refinados se for (u_3, u_10)
        if p_plot == "(u_3, u_10)":
            ax.axvline(th_wiki_opt, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Ótimo Wiki ({th_wiki_opt:.1f}°)")
            ax.axvline(th_code_opt, color="#d62728", linestyle="--", alpha=0.7, label=f"Ótimo Code ({th_code_opt:.1f}°)")
            ax.axvline(th_gsm_opt,  color="#2ca02c", linestyle="--", alpha=0.7, label=f"Ótimo GSM ({th_gsm_opt:.1f}°)")

        ax.set_title(f"Plano {p_plot} [{p_data['meta']['desc']}]", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ângulo theta de Givens (graus)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Delta PPL vs theta = 0 (pontos)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

    plt.suptitle("Trajetórias Angulares Multidomínio em L63: A Variedade Compartilhada e as Faixas Específicas", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(repo_root, "experiments", "domain_angular_trajectories.png")
    fig.savefig(fig1_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "domain_angular_trajectories.png"), dpi=300)
    plt.close(fig)
    print(f"  [OK] Gráfico de trajetórias salvo em: {fig1_path}")

    # Figura 2: Matriz de Grassmann e Transferência Cruzada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Matriz de Distância Geodésica de Grassmann (graus)
    d_names_list = ["WikiText", "HF_Code", "GSM8K"]
    dist_matrix = np.zeros((3, 3))
    for i_idx, d1 in enumerate(d_names_list):
        for j_idx, d2 in enumerate(d_names_list):
            if i_idx == j_idx:
                dist_matrix[i_idx, j_idx] = 0.0
            else:
                pair_k = f"{d1}_vs_{d2}" if f"{d1}_vs_{d2}" in grassmann_results else f"{d2}_vs_{d1}"
                dist_matrix[i_idx, j_idx] = grassmann_results[pair_k]["d_gr_subspace_deg"]

    im1 = ax1.imshow(dist_matrix, cmap="YlOrRd", vmin=0, vmax=180)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(d_names_list, fontweight="bold")
    ax1.set_yticklabels(d_names_list, fontweight="bold")
    for i_idx in range(3):
        for j_idx in range(3):
            val = dist_matrix[i_idx, j_idx]
            color = "white" if val > 90 else "black"
            ax1.text(j_idx, i_idx, f"{val:.1f}°", ha="center", va="center", color=color, fontweight="bold", fontsize=12)
    ax1.set_title("Distância Geodésica de Grassmann d_Gr\n(Separação Subespacial Efetiva)", fontsize=13, fontweight="bold")
    fig.colorbar(im1, ax=ax1, label="Distância Geodésica (graus)")

    # Subplot 2: Matriz de Transferência Cruzada (Delta PPL vs Raw)
    transfer_delta_matrix = np.zeros((3, 3))
    for i_idx, target_d in enumerate(d_names_list):
        for j_idx, source_d in enumerate(d_names_list):
            transfer_delta_matrix[i_idx, j_idx] = transfer_matrix[target_d][source_d]["delta"]

    im2 = ax2.imshow(transfer_delta_matrix, cmap="coolwarm", vmin=-15, vmax=40)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels([f"Q_{d}" for d in d_names_list], fontweight="bold")
    ax2.set_yticklabels(d_names_list, fontweight="bold")
    for i_idx in range(3):
        for j_idx in range(3):
            val = transfer_delta_matrix[i_idx, j_idx]
            color = "white" if abs(val) > 20 else "black"
            ax2.text(j_idx, i_idx, f"{val:+.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)
    ax2.set_title("Matriz de Transferência Cruzada (Delta PPL vs Raw)\n(Negativo = Ganho, Positivo = Degradação)", fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=ax2, label="Delta PPL")

    plt.suptitle("Geometria Subespacial e Especialização Funcional por Domínio em L63", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(repo_root, "experiments", "domain_grassmann_matrix.png")
    fig.savefig(fig2_path, dpi=300)
    fig.savefig(os.path.join(repo_root, "domain_grassmann_matrix.png"), dpi=300)
    plt.close(fig)
    print(f"  [OK] Gráfico de matriz salvo em: {fig2_path}")

    # 7. Salvar Dump JSON
    output_json_path = os.path.join(repo_root, "experiments", "domain_orientations_results.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "raw_metrics": raw_metrics,
            "base_metrics": base_metrics,
            "domain_plane_optima": domain_plane_optima,
            "refined_domain_optima": refined_domain_optima,
            "grassmann_results": grassmann_results,
            "transfer_matrix": transfer_matrix
        }, f, indent=2)
    print(f"\n[OK] Resultados completos salvos em: {output_json_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
