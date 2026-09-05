# -*- coding: utf-8 -*-
"""
Experimento A & Benchmark de Ablação do Refinador Residual (Versão Vetorizada de Passagem Única).
Compara no mesmo protocolo E2E e no mesmo conjunto cego de teste do WikiText-2:
  1. Atlas Raw (r=2048)
  2. Atlas + Linear rank-16
  3. Atlas + Linear rank-32
  4. Atlas + Linear rank-64
  5. Atlas + Linear rank-128
  6. Atlas + Linear rank-256
  7. Atlas + GELU rank-64
  8. Atlas + GELU rank-128
  9. Atlas + Alpha + Linear rank-64
 10. Atlas + Alpha + GELU rank-64

Otimização Estrutural: Processa o professor e as 10 trajetórias em paralelo na mesma camada,
eliminando a reabertura repetida dos arquivos safetensors e reduzindo o tempo de 50 min para ~3 min.
"""
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
import safetensors.torch as st
from datasets import load_dataset
from transformers import AutoTokenizer

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, load_layer_module, dequant, get_snapshot_dir
from atlas.config import AtlasConfig
from atlas.projection import compute_chart_atlas_bases, project_layer_weights
from atlas.residual import (
    fit_svd_stabilizer,
    fit_gelu_warmstart,
    fit_joint_alpha_stabilizer
)

def compute_ppl_and_metrics(logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor = None):
    vocab_size = logits.shape[-1]
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    targets_flat = targets.reshape(-1)
    nll = F.cross_entropy(logits_pred, targets_flat, reduction="mean").item()
    ppl = math.exp(nll)
    top1 = (logits_pred.argmax(dim=-1) == targets_flat).float().mean().item() * 100.0

    cos_sim = 0.0
    if teacher_logits is not None:
        t_pred = teacher_logits[:, :-1, :].reshape(-1, vocab_size).float()
        cos_sim = F.cosine_similarity(logits_pred, t_pred, dim=-1).mean().item()

    return nll, ppl, top1, cos_sim

def run_ablation_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print("EXPERIMENTO A: BENCHMARK DE ABLAÇÃO DO REFINADOR RESIDUAL (PASSAGEM ÚNICA CONJUNTA)")
    print("=" * 110)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    snapshot_dir = get_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 32
    seq_len = 64
    input_ids = tokens_all[:num_seqs * seq_len].view(num_seqs, seq_len).to(device)

    # Split: 16 seqs de treino (calibração) e 16 seqs de teste cego
    test_targets = input_ids[16:, 1:].contiguous()

    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=num_seqs)
    cfg = comp["cfg"]

    x_0 = F.embedding(input_ids.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    # 1. Carregar bases Atlas cacheadas
    bases_cache_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    if os.path.exists(bases_cache_path):
        print(f"Carregando bases Atlas cacheadas de {bases_cache_path}...")
        raw_bases = torch.load(bases_cache_path, map_location=device)
        bases_cache = {c_idx: {k: v.to(device=device, dtype=torch.bfloat16) for k, v in b.items()} for c_idx, b in raw_bases.items()}
    else:
        raise FileNotFoundError("atlas_bases.pt não encontrado.")

    # 2. Definição das 10 configurações
    configs = [
        {"name": "Atlas Raw", "type": "raw", "rank": 0, "alpha": False},
        {"name": "Linear r=16", "type": "linear", "rank": 16, "alpha": False},
        {"name": "Linear r=32", "type": "linear", "rank": 32, "alpha": False},
        {"name": "Linear r=64", "type": "linear", "rank": 64, "alpha": False},
        {"name": "Linear r=128", "type": "linear", "rank": 128, "alpha": False},
        {"name": "Linear r=256", "type": "linear", "rank": 256, "alpha": False},
        {"name": "GELU r=64", "type": "gelu", "rank": 64, "alpha": False},
        {"name": "GELU r=128", "type": "gelu", "rank": 128, "alpha": False},
        {"name": "Alpha+Linear r=64", "type": "alpha_linear", "rank": 64, "alpha": True},
        {"name": "Alpha+GELU r=64", "type": "alpha_gelu", "rank": 64, "alpha": True},
    ]

    # Inicializar os 10 estados do aluno e o estado do professor
    h_prof = x_0.clone()
    h_students = [x_0.clone() for _ in range(len(configs))]

    print(f"\nIniciando propagação conjunta pelas 64 camadas (1 Professor + {len(configs)} Alunos)...")
    torch.cuda.reset_peak_memory_stats()
    t_global_start = time.time()

    for chart_idx in range(16):
        t_chart_start = time.time()
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

            # A. Forward do Professor
            with torch.no_grad():
                out_p = layer_mod(h_prof, position_embeddings=pos_emb)
                h_prof_next = out_p[0] if isinstance(out_p, tuple) else out_p

            y_tr = h_prof_next[:16].reshape(-1, 5120)
            lambda_reg = 5e-3 if (48 <= l <= 55) else 1e-3

            # B. Configurar pesos projetados do Atlas no backbone
            p_w = project_layer_weights(chart_weights[l], bases)
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
            layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
            layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
            layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

            # C. Forward e Refinamento para cada uma das 10 configurações
            for c_idx, c in enumerate(configs):
                c_type = c["type"]
                r = c["rank"]
                h_in = h_students[c_idx]

                with torch.no_grad():
                    out_s = layer_mod(h_in, position_embeddings=pos_emb)
                    h_out = out_s[0] if isinstance(out_s, tuple) else out_s

                if c_type == "raw":
                    h_students[c_idx] = h_out
                else:
                    x_tr = h_out[:16].reshape(-1, 5120)

                    if c_type == "linear":
                        W_down, W_up = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                        corr = torch.matmul(torch.matmul(h_out, W_down), W_up)
                        h_students[c_idx] = h_out + corr
                    elif c_type == "gelu":
                        W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                        W_up = fit_gelu_warmstart(x_tr, y_tr, W_down, lambda_reg=lambda_reg)
                        z = F.gelu(torch.matmul(h_out.float(), W_down.float())).to(dtype=h_out.dtype)
                        corr = torch.matmul(z, W_up)
                        h_students[c_idx] = h_out + corr
                    elif c_type == "alpha_linear":
                        W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                        alpha_opt, W_up = fit_joint_alpha_stabilizer(x_tr, y_tr, W_down, is_gelu=False, lambda_reg=lambda_reg)
                        corr = torch.matmul(torch.matmul(h_out, W_down), W_up)
                        h_students[c_idx] = alpha_opt * h_out + corr
                    elif c_type == "alpha_gelu":
                        W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                        alpha_opt, W_up = fit_joint_alpha_stabilizer(x_tr, y_tr, W_down, is_gelu=True, lambda_reg=lambda_reg)
                        z = F.gelu(torch.matmul(h_out.float(), W_down.float())).to(dtype=h_out.dtype)
                        corr = torch.matmul(z, W_up)
                        h_students[c_idx] = alpha_opt * h_out + corr

            h_prof = h_prof_next
            del layer_mod
            torch.cuda.empty_cache()

        t_chart = time.time() - t_chart_start
        print(f"  -> Carta {chart_idx:02d} (camadas {start_l:02d}..{end_l-1:02d}) concluída em {t_chart:.1f}s.")

    t_total_propagation = time.time() - t_global_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"\n[OK] Propagação conjunta de 64 camadas finalizada em {t_total_propagation:.1f}s! Pico VRAM: {peak_vram_mb:.0f} MB")

    # 3. Avaliação Final nos 1.024 Tokens de Teste Cego ([16:])
    print("\nAvaliando métricas finais no conjunto de teste cego...")

    # Logits e métricas do Professor
    with torch.no_grad():
        x_norm_prof = comp["norm_mod"](h_prof[16:])
        logits_prof = F.linear(x_norm_prof, comp["lm_head_w"])
    nll_prof, ppl_prof, top1_prof, _ = compute_ppl_and_metrics(logits_prof, test_targets)
    h_prof_final = h_prof[16:].clone()
    norm_prof_final = torch.norm(h_prof_final.float()).item()

    results = []
    for c_idx, c in enumerate(configs):
        c_name = c["name"]
        c_type = c["type"]
        r = c["rank"]

        h_s_final = h_students[c_idx][16:].clone()
        with torch.no_grad():
            x_norm_stud = comp["norm_mod"](h_s_final)
            logits_stud = F.linear(x_norm_stud, comp["lm_head_w"])

        nll, ppl, top1, cos_sim = compute_ppl_and_metrics(logits_stud, test_targets, teacher_logits=logits_prof)
        e_64 = torch.norm((h_prof_final - h_s_final).float()).item() / norm_prof_final * 100.0

        p_refiner = 64 * (2 * 5120 * r + (1 if c["alpha"] else 0)) if r > 0 else 0
        p_refiner_m = p_refiner / 1e6

        results.append({
            "name": c_name,
            "type": c_type,
            "rank": r,
            "has_alpha": c["alpha"],
            "nll": nll,
            "ppl": ppl,
            "top1": top1,
            "cos_sim": cos_sim,
            "e_64": e_64,
            "params_refiner_m": p_refiner_m
        })

    # 4. Tabela Consolidada de Resultados
    print("\n" + "=" * 110)
    print("                     RESULTADOS DO EXPERIMENTO A: ABLAÇÃO DO REFINADOR RESIDUAL")
    print("=" * 110)
    header = f"{'Configuração':<22} | {'Params':<8} | {'NLL':<7} | {'PPL':<9} | {'Top-1':<7} | {'Cosine':<7} | {'e_64 (%)':<8}"
    print(header)
    print("-" * 110)
    print(f"{'Professor Oficial FP8':<22} | {'-':<8} | {nll_prof:<7.4f} | {ppl_prof:<9.2f} | {top1_prof:<6.2f}% | {'1.0000':<7} | {'0.00%':<8}")
    for r_entry in results:
        print(f"{r_entry['name']:<22} | +{r_entry['params_refiner_m']:<5.2f}M | {r_entry['nll']:<7.4f} | {r_entry['ppl']:<9.2f} | {r_entry['top1']:<6.2f}% | {r_entry['cos_sim']:<7.4f} | {r_entry['e_64']:<7.2f}%")
    print("=" * 110)

    # 5. Salvar em JSON
    out_json = os.path.join(repo_root, "experiments", "refiner_ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "teacher": {"nll": nll_prof, "ppl": ppl_prof, "top1": top1_prof},
            "ablation": results,
            "timing": {"total_seconds": t_total_propagation, "peak_vram_mb": peak_vram_mb}
        }, f, indent=2)
    print(f"\n[OK] Resultados completos salvos em: {out_json}")

if __name__ == "__main__":
    run_ablation_suite()
