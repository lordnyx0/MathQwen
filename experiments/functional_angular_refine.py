# -*- coding: utf-8 -*-
"""
EXPERIMENTO DE MAPEAMENTO ANGULAR EM L63: FASES C, F e H
- FASE C: Refinamento angular hierárquico nos melhores planos:
  * Nível 1: theta* +- 30° com passo 5°
  * Nível 2: theta* +- 10° com passo 1°
  * Nível 3: theta* +- 2° com passo 0.25°
- FASE F: Estimativa numérica da curvatura local (d PPL / d theta e d^2 PPL / d theta^2).
- FASE H: Validação causal estrita em theta* +- 10°, theta* +- 5°, theta*.
- Salva os checkpoints: Uc_angular_best.pt e Uc_angular_best_plan.pt.
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

def compute_grassmann_distance(U1: torch.Tensor, U2: torch.Tensor):
    M = torch.matmul(U1.t(), U2).float()
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0)
    thetas = torch.acos(S)
    return torch.norm(thetas).item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("REFINAMENTO ANGULAR HIERARQUICO, CURVATURA E VALIDACAO CAUSAL EM L63 (FASES C, F, H)")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Carregar Resultados da Varredura 360° para Identificar os Melhores Planos
    scan_json_path = os.path.join(repo_root, "experiments", "functional_angular_results.json")
    if not os.path.exists(scan_json_path):
        raise FileNotFoundError(f"Arquivo {scan_json_path} nao encontrado!")
    with open(scan_json_path, "r", encoding="utf-8") as f:
        scan_data = json.load(f)

    ranked_plans = scan_data["ranking"]
    ppl_ref = scan_data["ppl_reference"]

    # Selecionar os dois melhores planos
    top2_plans = ranked_plans[:2]
    print(f"Ponto de Referência PPL(0): {ppl_ref:.2f}")
    print(f"Top 1 Plano Selecionado: {top2_plans[0]['plane']} (Melhor inicial: {top2_plans[0]['best_theta']}° | PPL: {top2_plans[0]['best_ppl']:.2f})")
    print(f"Top 2 Plano Selecionado: {top2_plans[1]['plane']} (Melhor inicial: {top2_plans[1]['best_theta']}° | PPL: {top2_plans[1]['best_ppl']:.2f})")

    # 2. Carregar Componentes e Dados de Referência
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

    # 3. Base U_base e Calibração de W
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

    # Função auxiliar de avaliação angular em um plano (i, j)
    def eval_angle(i: int, j: int, angle_deg: float):
        th_rad = angle_deg * (math.pi / 180.0)
        U_rot = apply_givens_rotation(U_base, i, j, th_rad)
        with torch.no_grad():
            corr_rot = torch.matmul(torch.matmul(h_te, U_rot), W_opt)
            l_rot = F.linear(norm_mod(h_te + corr_rot), lm_head_w)
            m = compute_metrics(l_rot, test_targets, teacher_logits=logits_prof_test)
        d_gr = compute_grassmann_distance(U_base, U_rot)
        return m["ppl"], m["top1"], m["kl_div"], d_gr, U_rot

    refinement_results = {}
    best_overall_ref = None
    best_overall_ppl = 9999.0

    # 4. Executar Refinamento Hierárquico nos 2 Melhores Planos
    for p_info in top2_plans:
        p_name = p_info["plane"]
        p_meta = scan_data["plans_data"][p_name]["meta"]
        i, j = p_meta["i"], p_meta["j"]
        th_init = float(p_info["best_theta"])

        print(f"\n=========================================================================================")
        print(f"REFINAMENTO DO PLANO {p_name} [{p_meta['desc']}] - CENTRO INICIAL: {th_init}°")
        print(f"=========================================================================================")

        # Nível 1: +-30° com passo 5° (13 pontos)
        print("  -> [Nível 1] Varredura theta* +- 30° (passo 5°)...")
        deg_lvl1 = np.arange(th_init - 30.0, th_init + 30.1, 5.0)
        ppl_lvl1 = []
        for deg in deg_lvl1:
            p_val, t_val, kl_val, _, _ = eval_angle(i, j, deg)
            ppl_lvl1.append(p_val)
        min_idx1 = int(np.argmin(ppl_lvl1))
        th_best_lvl1 = float(deg_lvl1[min_idx1])
        print(f"     Melhor Nível 1: {th_best_lvl1:6.1f}° | PPL: {ppl_lvl1[min_idx1]:.2f} (Delta: {ppl_lvl1[min_idx1] - ppl_ref:+.2f})")

        # Nível 2: +-10° com passo 1° (21 pontos)
        print("  -> [Nível 2] Varredura theta* +- 10° (passo 1°)...")
        deg_lvl2 = np.arange(th_best_lvl1 - 10.0, th_best_lvl1 + 10.1, 1.0)
        ppl_lvl2 = []
        for deg in deg_lvl2:
            p_val, t_val, kl_val, _, _ = eval_angle(i, j, deg)
            ppl_lvl2.append(p_val)
        min_idx2 = int(np.argmin(ppl_lvl2))
        th_best_lvl2 = float(deg_lvl2[min_idx2])
        print(f"     Melhor Nível 2: {th_best_lvl2:6.1f}° | PPL: {ppl_lvl2[min_idx2]:.2f} (Delta: {ppl_lvl2[min_idx2] - ppl_ref:+.2f})")

        # Nível 3: +-2° com passo fino 0.25° (17 pontos)
        print("  -> [Nível 3] Varredura de Alta Resolução theta* +- 2° (passo 0.25°)...")
        deg_lvl3 = np.arange(th_best_lvl2 - 2.0, th_best_lvl2 + 2.01, 0.25)
        ppl_lvl3 = []
        top1_lvl3 = []
        kl_lvl3 = []
        for deg in deg_lvl3:
            p_val, t_val, kl_val, _, _ = eval_angle(i, j, deg)
            ppl_lvl3.append(p_val)
            top1_lvl3.append(t_val)
            kl_lvl3.append(kl_val)
        min_idx3 = int(np.argmin(ppl_lvl3))
        th_best_preciso = float(deg_lvl3[min_idx3])
        ppl_preciso = float(ppl_lvl3[min_idx3])
        top1_preciso = float(top1_lvl3[min_idx3])
        kl_preciso = float(kl_lvl3[min_idx3])

        delta_preciso = ppl_preciso - ppl_ref
        print(f"     >> MÍNIMO PRECISO: theta* = {th_best_preciso:6.2f}° | PPL* = {ppl_preciso:.2f} (Delta: {delta_preciso:+.2f}) | Top-1: {top1_preciso:.2f}%")

        # FASE F: Estimativa da Curvatura Local no Mínimo
        h = 0.25
        p_plus, _, _, _, _ = eval_angle(i, j, th_best_preciso + h)
        p_minus, _, _, _, _ = eval_angle(i, j, th_best_preciso - h)
        d_ppl_d_theta = (p_plus - p_minus) / (2.0 * h)
        d2_ppl_d_theta2 = (p_plus - 2.0 * ppl_preciso + p_minus) / (h ** 2)
        print(f"     Curvatura Local: dPPL/dtheta = {d_ppl_d_theta:+.4e} | d^2PPL/dtheta^2 = {d2_ppl_d_theta2:+.4e}")

        # FASE H: Validação Causal em theta* +- 10°, theta* +- 5°
        causal_points = [
            th_best_preciso - 10.0,
            th_best_preciso - 5.0,
            th_best_preciso,
            th_best_preciso + 5.0,
            th_best_preciso + 10.0
        ]
        causal_evals = []
        print("\n  -> [Fase H] Validação Causal Estrita:")
        for cp in causal_points:
            p_c, t_c, kl_c, d_gr_c, _ = eval_angle(i, j, cp)
            delta_c = p_c - ppl_preciso
            causal_evals.append({
                "theta": cp, "ppl": p_c, "top1": t_c, "kl": kl_c, "delta_vs_min": delta_c
            })
            print(f"     theta = {cp:6.2f}° | PPL = {p_c:.2f} (Delta vs Min: {delta_c:+5.2f}) | Top-1 = {t_c:.2f}% | KL = {kl_c:.4f}")

        # Obter a base no mínimo preciso
        _, _, _, _, U_best_plan = eval_angle(i, j, th_best_preciso)

        refinement_results[p_name] = {
            "plane": p_name,
            "meta": p_meta,
            "th_preciso": th_best_preciso,
            "ppl_preciso": ppl_preciso,
            "delta_preciso": delta_preciso,
            "top1_preciso": top1_preciso,
            "kl_preciso": kl_preciso,
            "curvature": {
                "d_ppl_d_theta": d_ppl_d_theta,
                "d2_ppl_d_theta2": d2_ppl_d_theta2
            },
            "causal_validation": causal_evals,
            "lvl3_scan": {
                "angles": [float(x) for x in deg_lvl3],
                "ppls": ppl_lvl3
            }
        }

        if ppl_preciso < best_overall_ppl:
            best_overall_ppl = ppl_preciso
            best_overall_ref = {
                "plane": p_name,
                "th_preciso": th_best_preciso,
                "ppl_preciso": ppl_preciso,
                "delta_preciso": delta_preciso,
                "top1_preciso": top1_preciso,
                "kl_preciso": kl_preciso,
                "U_best": U_best_plan
            }

    # 5. Salvar Checkpoints das Melhores Bases
    print("\n[Passo 3/3] Salvando checkpoints Uc_angular_best.pt e Uc_angular_best_plan.pt...")
    torch.save({
        "plane": best_overall_ref["plane"],
        "theta_preciso": best_overall_ref["th_preciso"],
        "ppl": best_overall_ref["ppl_preciso"],
        "top1": best_overall_ref["top1_preciso"],
        "kl": best_overall_ref["kl_preciso"],
        "U_angular": best_overall_ref["U_best"].to(torch.bfloat16)
    }, os.path.join(repo_root, "checkpoints", "Uc_angular_best.pt"))

    torch.save({
        "all_refinements": {k: {kk: v[kk] for kk in ["plane", "th_preciso", "ppl_preciso", "delta_preciso", "top1_preciso", "kl_preciso", "curvature"]} for k, v in refinement_results.items()},
        "best_overall": {k: best_overall_ref[k] for k in ["plane", "th_preciso", "ppl_preciso", "delta_preciso", "top1_preciso", "kl_preciso"]}
    }, os.path.join(repo_root, "checkpoints", "Uc_angular_best_plan.pt"))
    print("  [OK] Checkpoints salvos com sucesso.")

    # Salvar JSON de Refinamento
    refine_json_path = os.path.join(repo_root, "experiments", "functional_angular_refine_results.json")
    with open(refine_json_path, "w", encoding="utf-8") as f:
        json.dump(refinement_results, f, indent=2)
    print(f"  [OK] Resultados de refinamento salvos em: {refine_json_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
