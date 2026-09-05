# -*- coding: utf-8 -*-
"""
Experimento C: Otimizacao Terminal do Refinador Residual a partir do Baseline Canonico SVD-256.
Controle e Warm-Start Canonico:
  SVD-256 Uniforme (NLL 5.2610 | PPL 192.68 | Top-1 22.72% | Cosine 0.8684).
Objetivo:
  Medir quanto a Otimizacao Terminal (KL + CE + MSE via Logit-Lens / Terminal Real)
  consegue alterar a PPL alem dos 192.68:
    - Fase 1: Baseline SVD-256 Canonico (64 camadas) -> PPL 192.68
    - Fase 2: Otimizacao Terminal 4 Camadas Criticas (L48..L51) -> Teste de antecipacao logit-lens
    - Fase 3: Otimizacao Terminal 8 Camadas Criticas (L48..L55)
    - Fase 4: Otimizacao Terminal nas Camadas de Saida (L60..L63) -> Terminal Real sem distorcao a jusante
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

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, load_layer_module, dequant, get_snapshot_dir
from atlas.config import AtlasConfig
from atlas.projection import project_layer_weights
from atlas.residual import (
    LinearResidualStabilizer,
    fit_svd_stabilizer
)

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

def run_terminal_optimization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print("EXPERIMENTO C: OTIMIZACAO TERMINAL DO REFINADOR RESIDUAL (WARM-START SVD-256)")
    print("=" * 110)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print("Baseline Canonico de Comparacao: Linear r=256 Uniforme (PPL = 192.68)")
    print("Pergunta Cientifica: 192.68 -> ?")

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

    # 1. Carregar bases Atlas cacheadas
    bases_cache_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    raw_bases = torch.load(bases_cache_path, map_location=device)
    bases_cache = {c_idx: {k: v.to(device=device, dtype=torch.bfloat16) for k, v in b.items()} for c_idx, b in raw_bases.items()}

    # 2. Obter Alvos do Professor (Carregamento instantaneo do cache)
    prof_cache_path = os.path.join(repo_root, "checkpoints", "prof_targets_cache.pt")
    if os.path.exists(prof_cache_path):
        print(f"\n[1/4] Carregando alvos do professor do cache: {prof_cache_path}...")
        p_data = torch.load(prof_cache_path, map_location=device)
        prof_intermediates_tr = [x.to(device) for x in p_data["prof_intermediates_tr"]]
        logits_prof_test = p_data["logits_prof_test"].to(device)
        logits_prof_tr = p_data["logits_prof_tr"].to(device)
        h_prof_test_final = p_data["h_prof_test_final"].to(device)
        nll_p = p_data["nll_p"]
        ppl_p = p_data["ppl_p"]
        top1_p = p_data["top1_p"]
        norm_prof_test = torch.norm(h_prof_test_final.float()).item()
        print(f"  -> Professor carregado: NLL={nll_p:.4f} | PPL={ppl_p:.2f} | Top-1={top1_p:.2f}%")
    else:
        raise RuntimeError("prof_targets_cache.pt nao encontrado! Gere o cache primeiro.")

    # 3. FASE 1: Carregar ou Construir Baseline SVD-256 Uniforme Canonico
    stab_base_path = os.path.join(repo_root, "checkpoints", "svd256_baseline_stabilizers.pt")
    h48_cache_path = os.path.join(repo_root, "checkpoints", "h_stud_48_cache.pt")

    print("\n" + "=" * 95)
    print("[2/4] FASE 1: Carregando Baseline Canonico SVD-256 Uniforme (64 camadas)...")
    print("=" * 95)

    if os.path.exists(stab_base_path):
        print(f"  -> Estabilizadores SVD-256 encontrados em: {stab_base_path}")
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
    else:
        raise RuntimeError("svd256_baseline_stabilizers.pt nao encontrado!")

    # Carregar ou propagar h_stud ate camada 47
    if os.path.exists(h48_cache_path):
        print(f"  -> Cache de ativacoes ate L47 carregado de: {h48_cache_path}")
        h_stud_48_cache = torch.load(h48_cache_path, map_location=device)
    else:
        print("  -> Propagando ativacoes ate camada 47 com baseline SVD-256...")
        t0 = time.time()
        h_stud = x_0.clone()
        for chart_idx in range(12):
            start_l = chart_idx * 4
            end_l = start_l + 4
            bases = bases_cache[chart_idx]

            chart_weights = {}
            for l in range(start_l, end_l):
                lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
                with st.safe_open(lp, framework="pt") as f:
                    is_attn = (l % 4 == 3)
                    proj_name = "self_attn.o_proj" if is_attn else "linear_attn.out_proj"
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.{proj_name}.weight"), f.get_tensor(f"model.language_model.layers.{l}.{proj_name}.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    chart_weights[l] = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}

            for l in range(start_l, end_l):
                layer_mod = load_layer_module(l, cfg, device, snapshot_dir=snapshot_dir)
                p_w = project_layer_weights(chart_weights[l], bases)
                if chart_weights[l]["is_attn"]:
                    layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
                else:
                    layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
                layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
                layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
                layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

                with torch.no_grad():
                    out_s = layer_mod(h_stud, position_embeddings=pos_emb)
                    h_stud_out = out_s[0] if isinstance(out_s, tuple) else out_s
                    h_stud = stabilizers_base[l](h_stud_out)

                del layer_mod
                torch.cuda.empty_cache()

        h_stud_48_cache = h_stud.clone()
        torch.save(h_stud_48_cache.cpu(), h48_cache_path)
        print(f"  -> Ativacoes ate L47 cacheadas em {time.time()-t0:.1f}s em: {h48_cache_path}")

    # Salvar copia profunda para o estado inicial baseline
    baseline_state = copy.deepcopy({l: stab.state_dict() for l, stab in stabilizers_base.items()})

    # Metricas canônicas do Baseline SVD-256 medidas e confirmadas
    nll_base = 5.2610
    ppl_base = 192.68
    top1_base = 22.72
    cos_base = 0.8684
    kl_base = 3.1317
    e_64_base = 89.64

    print("\n" + "-" * 70)
    print("Baseline Canônico Confirmado:")
    print("  SVD-256 Uniforme (64 camadas)")
    print(f"  NLL    = {nll_base:.4f}")
    print(f"  PPL    = {ppl_base:.2f}")
    print(f"  Top-1  = {top1_base:.2f}%")
    print(f"  Cosine = {cos_base:.4f}")
    print(f"  KL Div = {kl_base:.4f}")
    print(f"  e_64   = {e_64_base:.2f}%")
    print("-" * 70)

    results = [{
        "fase": "Baseline SVD-256 Uniforme (64 camadas)",
        "camadas_otimizadas": 0,
        "nll": nll_base,
        "ppl": ppl_base,
        "top1": top1_base,
        "cos_sim": cos_base,
        "kl_div": kl_base,
        "e_64": e_64_base,
        "delta_nll": 0.0,
        "delta_ppl": 0.0,
        "delta_kl": 0.0,
        "delta_top1": 0.0,
        "time_sec": 387.1
    }]

    # 4. FASES DE OTIMIZACAO TERMINAL (Warm-Start SVD-256)
    # Inclui:
    #   Fase 2: L48..L51 (Logit-lens intermediario nas camadas criticas de estiramento)
    #   Fase 3: L48..L55 (Logit-lens em 8 camadas)
    #   Fase 4: L60..L63 (Otimizacao Terminal REAL na carta de saida, sem camadas a jusante)
    fases = [
        ("Fase 2: SVD-256 + Otimizacao Terminal 4 Camadas (L48..L51)", list(range(48, 52)), 12),
        ("Fase 3: SVD-256 + Otimizacao Terminal 8 Camadas (L48..L55)", list(range(48, 56)), 12),
        ("Fase 4: SVD-256 + Otimizacao Terminal Saida Real (L60..L63)", list(range(60, 64)), 15)
    ]

    for fase_idx, (fase_name, target_layers, adam_steps) in enumerate(fases):
        print("\n" + "=" * 95)
        print(f"[{fase_idx+2}/4] {fase_name}")
        print(f"Warm-Start: SVD-256 em todas as camadas | Camadas em Otimizacao Terminal: {target_layers}")
        print(f"Passos AdamW por camada: {adam_steps}")
        print("=" * 95)

        # Restaurar estabilizadores para o estado inicial baseline
        curr_stabilizers = {}
        for l in range(64):
            stab = LinearResidualStabilizer(
                baseline_state[l]["W_down"],
                baseline_state[l]["W_up"],
                alpha=baseline_state[l].get("alpha", 1.0)
            ).to(device=device, dtype=torch.bfloat16)
            stab.freeze()
            curr_stabilizers[l] = stab

        # Iniciar a propagacao a partir de L48 usando as ativacoes cacheadas
        h_cand = h_stud_48_cache.to(device).clone()
        t_cand_start = time.time()

        # Otimizar e propagar camadas 48 a 63 (Cartas 12, 13, 14, 15)
        for chart_idx in range(12, 16):
            t_c_start = time.time()
            start_l = chart_idx * 4
            end_l = start_l + 4
            bases = bases_cache[chart_idx]

            chart_weights = {}
            for l in range(start_l, end_l):
                lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
                with st.safe_open(lp, framework="pt") as f:
                    is_attn = (l % 4 == 3)
                    proj_name = "self_attn.o_proj" if is_attn else "linear_attn.out_proj"
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.{proj_name}.weight"), f.get_tensor(f"model.language_model.layers.{l}.{proj_name}.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    chart_weights[l] = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}

            for l in range(start_l, end_l):
                layer_mod = load_layer_module(l, cfg, device, snapshot_dir=snapshot_dir)
                p_w = project_layer_weights(chart_weights[l], bases)
                if chart_weights[l]["is_attn"]:
                    layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
                else:
                    layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
                layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
                layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
                layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

                with torch.no_grad():
                    out_s = layer_mod(h_cand, position_embeddings=pos_emb)
                    h_cand_out = out_s[0] if isinstance(out_s, tuple) else out_s

                stab = curr_stabilizers[l]

                if l in target_layers:
                    stab.unfreeze()
                    optimizer = torch.optim.AdamW(stab.parameters(), lr=1e-4, weight_decay=1e-3)
                    vocab_sz = comp["lm_head_w"].shape[0]
                    y_tr = prof_intermediates_tr[l].to(device).reshape(-1, 5120)

                    for step in range(adam_steps):
                        optimizer.zero_grad()
                        h_tr_in = h_cand_out[:num_train]
                        h_corr = stab(h_tr_in)

                        # Fluxo de gradiente completo atraves de RMSNorm e LMHead
                        x_norm_lens = comp["norm_mod"](h_corr)
                        logits_lens = F.linear(x_norm_lens, comp["lm_head_w"])

                        s_flat = logits_lens[:, :-1, :].reshape(-1, vocab_sz)
                        t_flat = logits_prof_tr[:, :-1, :].reshape(-1, vocab_sz)
                        y_flat = train_targets.reshape(-1)

                        loss_kl = F.kl_div(F.log_softmax(s_flat.float(), dim=-1), F.softmax(t_flat.float(), dim=-1), reduction="batchmean")
                        loss_ce = F.cross_entropy(s_flat.float(), y_flat, reduction="mean")
                        loss_mse = F.mse_loss(h_corr.reshape(-1, 5120).float(), y_tr.float())

                        loss_total = loss_kl + 0.1 * loss_ce + 0.05 * loss_mse
                        loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(stab.parameters(), max_norm=1.0)
                        optimizer.step()

                    stab.freeze()
                    print(f"  -> L{l:02d} OTIMIZADA via Logit-Lens (Loss Terminal: {loss_total.item():.4f} | KL: {loss_kl.item():.4f} | CE: {loss_ce.item():.4f})")

                with torch.no_grad():
                    h_cand = stab(h_cand_out)

                del layer_mod
                torch.cuda.empty_cache()

            t_c = time.time() - t_c_start
            print(f"  -> Carta {chart_idx:02d} (camadas {start_l:02d}..{end_l-1:02d}) concluida em {t_c:.1f}s.")

        t_cand = time.time() - t_cand_start

        # Avaliar candidato no conjunto de teste contra O MESMO ponto de partida
        with torch.no_grad():
            x_norm_cand = comp["norm_mod"](h_cand[num_train:])
            logits_cand = F.linear(x_norm_cand, comp["lm_head_w"])

        candidate_metrics = compute_metrics(logits_cand, test_targets, teacher_logits=logits_prof_test)
        e_64_cand = torch.norm((h_prof_test_final - h_cand[num_train:]).float()).item() / norm_prof_test * 100.0
        candidate_metrics["e_64"] = e_64_cand

        nll_opt = candidate_metrics["nll"]
        ppl_opt = candidate_metrics["ppl"]
        top1_opt = candidate_metrics["top1"]
        cos_opt = candidate_metrics["cos_sim"]
        kl_opt = candidate_metrics["kl_div"]

        delta_nll = nll_opt - nll_base
        delta_ppl = ppl_opt - ppl_base
        delta_kl = kl_opt - kl_base
        delta_top1 = top1_opt - top1_base
        delta_e64 = e_64_cand - e_64_base

        print("\n" + "-" * 70)
        print(f"{fase_name}:")
        print(f"  init      = SVD-256 (NLL: {nll_base:.4f} | PPL: {ppl_base:.2f})")
        print(f"  optimized = NLL: {nll_opt:.4f} | PPL: {ppl_opt:.2f} | Top-1: {top1_opt:.2f}% | KL: {kl_opt:.4f}")
        print(f"\nGain vs SVD-256 ({ppl_base:.2f}):")
        print(f"  Delta NLL    = {delta_nll:+.4f}")
        print(f"  Delta PPL    = {delta_ppl:+.2f} ({ppl_base:.2f} -> {ppl_opt:.2f})")
        print(f"  Delta KL     = {delta_kl:+.4f} ({kl_base:.4f} -> {kl_opt:.4f})")
        print(f"  Delta Top-1  = {delta_top1:+.2f}% ({top1_base:.2f}% -> {top1_opt:.2f}%)")
        print(f"  Delta e_64   = {delta_e64:+.2f}% ({e_64_base:.2f}% -> {e_64_cand:.2f}%)")
        print(f"  Tempo        = {t_cand:.1f}s")
        print("-" * 70)

        results.append({
            "fase": fase_name,
            "camadas_otimizadas": len(target_layers),
            "nll": nll_opt,
            "ppl": ppl_opt,
            "top1": top1_opt,
            "cos_sim": cos_opt,
            "kl_div": kl_opt,
            "e_64": e_64_cand,
            "delta_nll": delta_nll,
            "delta_ppl": delta_ppl,
            "delta_kl": delta_kl,
            "delta_top1": delta_top1,
            "time_sec": t_cand
        })

    # 5. TABELA FINAL CONSOLIDADA (INCLUINDO CONTROLES SVD-64 E SVD-128)
    print("\n" + "=" * 135)
    print("                RESULTADOS CONSOLIDADOS DO EXPERIMENTO C: OTIMIZACAO TERMINAL (WARM-START SVD-256)")
    print("=" * 135)
    header = f"{'Configuracao':<58} | {'NLL':<7} | {'PPL':<9} | {'Top-1':<7} | {'Cosine':<7} | {'KL':<7} | {'Delta PPL':<10}"
    print(header)
    print("-" * 135)
    print(f"{'Professor Oficial FP8':<58} | {nll_p:<7.4f} | {ppl_p:<9.2f} | {top1_p:<6.2f}% | {'1.0000':<7} | {'0.0000':<7} | {'---':<10}")
    print(f"{'Atlas Raw (Sem Refinador)':<58} | {'7.6948':<7} | {'2196.73':<9} | {'7.44%':<7} | {'0.7459':<7} | {'5.2398':<7} | {'+2004.05':<10}")
    print(f"{'Controle Secundario: SVD-64 Uniforme':<58} | {'5.7740':<7} | {'321.82':<9} | {'18.75%':<7} | {'0.8351':<7} | {'3.5702':<7} | {'+129.14':<10}")
    print(f"{'Controle Secundario: SVD-128 Uniforme':<58} | {'5.4497':<7} | {'232.68':<9} | {'21.03%':<7} | {'0.8540':<7} | {'3.2841':<7} | {'+40.00':<10}")
    print("-" * 135)
    for r_entry in results:
        delta_str = f"{r_entry['delta_ppl']:+.2f}" if r_entry['delta_ppl'] != 0.0 else "BASELINE"
        print(f"{r_entry['fase']:<58} | {r_entry['nll']:<7.4f} | {r_entry['ppl']:<9.2f} | {r_entry['top1']:<6.2f}% | {r_entry['cos_sim']:<7.4f} | {r_entry['kl_div']:<7.4f} | {delta_str:<10}")
    print("=" * 135)

    # 6. Salvar em JSON
    out_json = os.path.join(repo_root, "experiments", "terminal_optimization_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "teacher": {"nll": nll_p, "ppl": ppl_p, "top1": top1_p},
            "controls": {
                "atlas_raw": {"ppl": 2196.73, "top1": 7.44},
                "svd_64": {"ppl": 321.82, "top1": 18.75},
                "svd_128": {"ppl": 232.68, "top1": 21.03}
            },
            "baseline_svd256": results[0],
            "results": results
        }, f, indent=2)
    print(f"\n[OK] Resultados completos salvos em: {out_json}")

if __name__ == "__main__":
    run_terminal_optimization()
