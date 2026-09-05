# -*- coding: utf-8 -*-
"""
Teste Cientifico Rigoroso: Otimizacao Terminal na Camada Final Real (Layer 63).
Em L63, a saida do estabilizador vai diretamente para RMSNorm e LMHead.
Nao existem camadas subsequentes para sofrer distorcao semantica.
Pergunta: 192.68 -> ?
"""
import os
import sys
import time
import copy
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as st
from datasets import load_dataset
from transformers import AutoTokenizer

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
from atlas.residual import LinearResidualStabilizer

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

def run_layer63_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("EXPERIMENTO CIENTIFICO: OTIMIZACAO TERMINAL NA CAMADA FINAL (L63)")
    print("=" * 95)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print("Baseline Canônico: Linear r=256 Uniforme (PPL = 192.68)")
    print("Pergunta: 192.68 -> ?")

    snapshot_dir = get_snapshot_dir()
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
    x_0 = F.embedding(input_ids.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, total_seqs, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    # 1. Carregar caches
    bases_cache_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    raw_bases = torch.load(bases_cache_path, map_location=device)
    bases_cache = {c_idx: {k: v.to(device=device, dtype=torch.bfloat16) for k, v in b.items()} for c_idx, b in raw_bases.items()}

    prof_cache_path = os.path.join(repo_root, "checkpoints", "prof_targets_cache.pt")
    p_data = torch.load(prof_cache_path, map_location=device)
    prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
    logits_prof_tr = p_data["logits_prof_tr"].to(device)
    logits_prof_test = p_data["logits_prof_test"].to(device)

    stab_base_path = os.path.join(repo_root, "checkpoints", "svd256_baseline_stabilizers.pt")
    raw_stabs = torch.load(stab_base_path, map_location=device)
    stabilizers_base = {}
    for l in range(64):
        stab = LinearResidualStabilizer(
            raw_stabs[l]["W_down"],
            raw_stabs[l]["W_up"],
            alpha=raw_stabs[l].get("alpha", 1.0)
        ).to(device=device, dtype=torch.bfloat16)
        stab.freeze()
        stabilizers_base[l] = stab

    # 2. Obter ativacao de entrada de L63
    h63_cache_path = os.path.join(repo_root, "checkpoints", "h_stud_63_cache.pt")
    if os.path.exists(h63_cache_path):
        print(f"Carregando entrada de L63 do cache: {h63_cache_path}...")
        h_in_63 = torch.load(h63_cache_path, map_location=device)
    else:
        h48_cache_path = os.path.join(repo_root, "checkpoints", "h_stud_48_cache.pt")
        print(f"Carregando h_stud_48 e propagando L48..L62...")
        t0 = time.time()
        h_curr = torch.load(h48_cache_path, map_location=device)

        for chart_idx in range(12, 16):
            start_l = chart_idx * 4
            end_l = start_l + 4
            bases = bases_cache[chart_idx]
            for l in range(start_l, end_l):
                if l == 63:
                    break
                layer_mod = load_layer_module(l, cfg, device, snapshot_dir=snapshot_dir)
                lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
                with st.safe_open(lp, framework="pt") as f:
                    is_attn = (l % 4 == 3)
                    proj_name = "self_attn.o_proj" if is_attn else "linear_attn.out_proj"
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.{proj_name}.weight"), f.get_tensor(f"model.language_model.layers.{l}.{proj_name}.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    cw = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}
                p_w = project_layer_weights(cw, bases)
                if cw["is_attn"]:
                    layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
                else:
                    layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
                layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
                layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
                layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])
                with torch.no_grad():
                    out_s = layer_mod(h_curr, position_embeddings=pos_emb)
                    h_out = out_s[0] if isinstance(out_s, tuple) else out_s
                    h_curr = stabilizers_base[l](h_out)
                del layer_mod
                torch.cuda.empty_cache()

        h_in_63 = h_curr.clone()
        torch.save(h_in_63.cpu(), h63_cache_path)
        print(f"  -> Entrada de L63 gerada e cacheada em {time.time()-t0:.1f}s em: {h63_cache_path}")

    # 3. Forward pass da Layer 63 (Atlas projetado)
    l63_mod = load_layer_module(63, cfg, device, snapshot_dir=snapshot_dir)
    chart15_bases = bases_cache[15]
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

    with torch.no_grad():
        out_s = l63_mod(h_in_63, position_embeddings=pos_emb)
        h_63_raw = out_s[0] if isinstance(out_s, tuple) else out_s
    del l63_mod
    torch.cuda.empty_cache()

    # 4. Avaliar Baseline Inicial com Stab 63
    stab_base_63 = stabilizers_base[63]
    with torch.no_grad():
        h_corr_base = stab_base_63(h_63_raw)
        logits_tr_base = F.linear(comp["norm_mod"](h_corr_base[:num_train]), comp["lm_head_w"])
        logits_te_base = F.linear(comp["norm_mod"](h_corr_base[num_train:]), comp["lm_head_w"])

    m_tr_base = compute_metrics(logits_tr_base, train_targets)
    m_te_base = compute_metrics(logits_te_base, test_targets, teacher_logits=logits_prof_test)

    print("\n" + "-" * 70)
    print("Baseline Canônico SVD-256 Confirmado:")
    print(f"  Train: NLL = {m_tr_base['nll']:.4f} | PPL = {m_tr_base['ppl']:.2f} | Top-1 = {m_tr_base['top1']:.2f}%")
    print(f"  Test:  NLL = {m_te_base['nll']:.4f} | PPL = {m_te_base['ppl']:.2f} | Top-1 = {m_te_base['top1']:.2f}% | Cosine = {m_te_base['cos_sim']:.4f} | KL = {m_te_base['kl_div']:.4f}")
    print("-" * 70)

    # 5. Otimizacao Terminal de L63 com Multiplas Taxas de Aprendizado
    vocab_sz = comp["lm_head_w"].shape[0]
    y_tr_63 = prof_intermediates_tr[63].to(device).reshape(-1, 5120)

    lrs_to_test = [1e-4, 3e-5, 1e-5]
    all_runs = {}

    for lr in lrs_to_test:
        print(f"\n>>> Testando Otimizacao Terminal em L63 com lr = {lr:.0e} (AdamW)...")
        stab_opt = copy.deepcopy(stab_base_63)
        stab_opt.unfreeze()
        optimizer = torch.optim.AdamW(stab_opt.parameters(), lr=lr, weight_decay=1e-3)

        records = []
        for step in range(1, 26):
            optimizer.zero_grad()
            h_tr_in = h_63_raw[:num_train]
            h_corr = stab_opt(h_tr_in)

            x_norm = comp["norm_mod"](h_corr)
            logits = F.linear(x_norm, comp["lm_head_w"])

            s_flat = logits[:, :-1, :].reshape(-1, vocab_sz)
            t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
            y_flat = train_targets.reshape(-1)

            loss_kl = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean")
            loss_ce = F.cross_entropy(s_flat.float(), y_flat, reduction="mean")
            loss_mse = F.mse_loss(h_corr.reshape(-1, 5120).float(), y_tr_63.float())
            loss_total = loss_kl + 0.1 * loss_ce + 0.05 * loss_mse
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(stab_opt.parameters(), max_norm=1.0)
            optimizer.step()

            if step in [1, 2, 3, 5, 10, 15, 20, 25]:
                stab_opt.freeze()
                with torch.no_grad():
                    h_eval = stab_opt(h_63_raw)
                    l_tr = F.linear(comp["norm_mod"](h_eval[:num_train]), comp["lm_head_w"])
                    l_te = F.linear(comp["norm_mod"](h_eval[num_train:]), comp["lm_head_w"])
                ev_tr = compute_metrics(l_tr, train_targets)
                ev_te = compute_metrics(l_te, test_targets, teacher_logits=logits_prof_test)
                delta_ppl = ev_te["ppl"] - m_te_base["ppl"]
                rec = {
                    "step": step,
                    "loss": loss_total.item(),
                    "tr_nll": ev_tr["nll"],
                    "tr_ppl": ev_tr["ppl"],
                    "te_nll": ev_te["nll"],
                    "te_ppl": ev_te["ppl"],
                    "te_top1": ev_te["top1"],
                    "te_cos": ev_te["cos_sim"],
                    "te_kl": ev_te["kl_div"],
                    "delta_ppl": delta_ppl
                }
                records.append(rec)
                print(f"  Step {step:02d} | Loss: {loss_total.item():.4f} | Train PPL: {ev_tr['ppl']:.2f} | Test PPL: {ev_te['ppl']:.2f} (Delta: {delta_ppl:+.2f}) | Top-1: {ev_te['top1']:.2f}% | KL: {ev_te['kl_div']:.4f}")
                stab_opt.unfreeze()

        all_runs[str(lr)] = records

    # 6. Tabela Final Consolidada de L63
    print("\n" + "=" * 120)
    print("                     TABELA CONSOLIDADA: OTIMIZACAO TERMINAL NA CAMADA FINAL (L63)")
    print("=" * 120)
    header = f"{'Configuracao':<40} | {'Loss':<8} | {'Train PPL':<12} | {'Test PPL':<12} | {'Delta PPL':<12} | {'Top-1':<8} | {'KL':<8}"
    print(header)
    print("-" * 120)
    print(f"{'Baseline SVD-256 (Passo 0)':<40} | {'---':<8} | {m_tr_base['ppl']:<12.2f} | {m_te_base['ppl']:<12.2f} | {'BASELINE':<12} | {m_te_base['top1']:<7.2f}% | {m_te_base['kl_div']:<8.4f}")
    for lr_str, recs in all_runs.items():
        for r in recs:
            if r["step"] in [1, 5, 15, 25]:
                label = f"L63 AdamW (lr={lr_str}, step={r['step']})"
                print(f"{label:<40} | {r['loss']:<8.4f} | {r['tr_ppl']:<12.2f} | {r['te_ppl']:<12.2f} | {r['delta_ppl']:<+12.2f} | {r['te_top1']:<7.2f}% | {r['te_kl']:<8.4f}")
    print("=" * 120)

    out_file = os.path.join(repo_root, "experiments", "layer63_terminal_optimization_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {"train": m_tr_base, "test": m_te_base},
            "runs": all_runs
        }, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {out_file}")

if __name__ == "__main__":
    run_layer63_experiment()
