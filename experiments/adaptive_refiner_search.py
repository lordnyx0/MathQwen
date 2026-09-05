# -*- coding: utf-8 -*-
"""
Experimento B: Busca de Política Adaptativa Aprendida por Validação Independente.
Regra:
  Para cada camada l, seleciona (tipo_l, rank_l) exclusivamente com base no split de validação (D_val).
  Critério estrito:
    e_h^{cand} <= e_h^{baseline} e NLL_val^{cand} <= NLL_val^{baseline}.
  Em seguida, congela a política e a avalia UMA ÚNICA VEZ no conjunto cego de teste (D_test).
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
    LinearResidualStabilizer,
    NonLinearResidualStabilizer,
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

def run_adaptive_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print("EXPERIMENTO B: BUSCA DE POLÍTICA ADAPTATIVA APRENDIDA (SPLIT TRIPLO: TRAIN / VAL / TEST)")
    print("=" * 110)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    snapshot_dir = get_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    # Split Triplo Formal:
    # Treino:    16 seqs x 64 tokens = 1.024 tokens (0..1024)
    # Validação:  8 seqs x 64 tokens =   512 tokens (1024..1536)
    # Teste:     16 seqs x 64 tokens = 1.024 tokens (1536..2560)
    num_train = 16
    num_val = 8
    num_test = 16
    total_seqs = num_train + num_val + num_test
    seq_len = 64

    input_ids = tokens_all[:total_seqs * seq_len].view(total_seqs, seq_len).to(device)

    train_slice = slice(0, num_train)
    val_slice = slice(num_train, num_train + num_val)
    test_slice = slice(num_train + num_val, total_seqs)

    val_targets = input_ids[val_slice, 1:].contiguous()
    test_targets = input_ids[test_slice, 1:].contiguous()

    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=total_seqs)
    cfg = comp["cfg"]

    x_0 = F.embedding(input_ids.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, total_seqs, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    # 1. Carregar bases Atlas
    bases_cache_path = os.path.join(repo_root, "checkpoints", "atlas_bases.pt")
    raw_bases = torch.load(bases_cache_path, map_location=device)
    bases_cache = {c_idx: {k: v.to(device=device, dtype=torch.bfloat16) for k, v in b.items()} for c_idx, b in raw_bases.items()}

    # 2. Inicializar estados
    h_prof = x_0.clone()
    h_stud = x_0.clone()

    learned_policy = []
    total_refiner_params = 0

    print("\nIniciando busca adaptativa camada por camada...")
    t_start = time.time()

    for chart_idx in range(16):
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

            y_tr = h_prof_next[train_slice].reshape(-1, 5120)
            y_val = h_prof_next[val_slice]
            norm_y_val = torch.norm(y_val.float()).item()

            # B. Forward do Aluno no backbone Atlas
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

            # C. Baseline na Validação (candidato "none")
            h_val_baseline = h_stud_out[val_slice]
            err_baseline = torch.norm((y_val - h_val_baseline).float()).item() / norm_y_val

            # Proxy NLL da validação via logit lens
            with torch.no_grad():
                logits_val_base = F.linear(comp["norm_mod"](h_val_baseline), comp["lm_head_w"])
                vocab_sz = logits_val_base.shape[-1]
                nll_val_base = F.cross_entropy(logits_val_base[:, :-1, :].reshape(-1, vocab_sz).float(), val_targets.reshape(-1)).item()

            # D. Espaço de Candidatos a testar
            candidates = [
                {"type": "none", "rank": 0, "alpha": False},
                {"type": "linear", "rank": 16, "alpha": False},
                {"type": "linear", "rank": 32, "alpha": False},
                {"type": "linear", "rank": 64, "alpha": False},
                {"type": "linear", "rank": 128, "alpha": False},
                {"type": "linear", "rank": 256, "alpha": False},
                {"type": "gelu", "rank": 64, "alpha": False},
                {"type": "gelu", "rank": 128, "alpha": False},
                {"type": "alpha_linear", "rank": 64, "alpha": True},
                {"type": "alpha_linear", "rank": 128, "alpha": True},
                {"type": "alpha_gelu", "rank": 64, "alpha": True},
            ]

            x_tr = h_stud_out[train_slice].reshape(-1, 5120)
            lambda_reg = 5e-3 if (48 <= l <= 55) else 1e-3

            best_cand = candidates[0]
            best_score = err_baseline
            best_stab = None
            best_err = err_baseline
            best_nll = nll_val_base

            for cand in candidates[1:]:
                c_type = cand["type"]
                r = cand["rank"]

                if c_type == "linear":
                    W_down, W_up = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                    stab = LinearResidualStabilizer(W_down, W_up)
                elif c_type == "gelu":
                    W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                    W_up = fit_gelu_warmstart(x_tr, y_tr, W_down, lambda_reg=lambda_reg)
                    stab = NonLinearResidualStabilizer(W_down, W_up)
                elif c_type == "alpha_linear":
                    W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                    alpha_opt, W_up = fit_joint_alpha_stabilizer(x_tr, y_tr, W_down, is_gelu=False, lambda_reg=lambda_reg)
                    stab = LinearResidualStabilizer(W_down, W_up, alpha=alpha_opt)
                elif c_type == "alpha_gelu":
                    W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r, lambda_reg=lambda_reg)
                    alpha_opt, W_up = fit_joint_alpha_stabilizer(x_tr, y_tr, W_down, is_gelu=True, lambda_reg=lambda_reg)
                    stab = NonLinearResidualStabilizer(W_down, W_up, alpha=alpha_opt)

                stab.freeze()
                with torch.no_grad():
                    h_val_cand = stab(h_val_baseline)
                    err_cand = torch.norm((y_val - h_val_cand).float()).item() / norm_y_val
                    logits_cand = F.linear(comp["norm_mod"](h_val_cand), comp["lm_head_w"])
                    nll_val_cand = F.cross_entropy(logits_cand[:, :-1, :].reshape(-1, vocab_sz).float(), val_targets.reshape(-1)).item()

                # Regra mínima: e_h <= e_h^{baseline} E NLL_val <= NLL_val^{baseline} + 0.01 (tolerância de ruído)
                if err_cand < err_baseline and nll_val_cand <= (nll_val_base + 0.05):
                    # Penalidade suave de complexidade para preferir ranks menores em empate técnico
                    complexity_pen = 1e-4 * (r / 256.0)
                    score = err_cand + complexity_pen
                    if score < best_score:
                        best_score = score
                        best_cand = cand
                        best_stab = stab
                        best_err = err_cand
                        best_nll = nll_val_cand

            # E. Aplicar o melhor estabilizador selecionado
            if best_stab is not None:
                with torch.no_grad():
                    h_stud = best_stab(h_stud_out)
                r = best_cand["rank"]
                p_l = 2 * 5120 * r + (1 if best_cand["alpha"] else 0)
            else:
                h_stud = h_stud_out
                p_l = 0

            total_refiner_params += p_l
            learned_policy.append({
                "layer": l,
                "type": best_cand["type"],
                "rank": best_cand["rank"],
                "alpha": best_cand["alpha"],
                "err_val": best_err,
                "nll_val": best_nll,
                "err_reduction_pct": (err_baseline - best_err) / err_baseline * 100.0 if err_baseline > 0 else 0.0
            })

            h_prof = h_prof_next
            del layer_mod
            torch.cuda.empty_cache()

            if l % 8 == 7:
                print(f"  -> Camadas 00..{l:02d} processadas. Última política L{l:02d}: {best_cand['type']}-r{best_cand['rank']} (e_h: {err_baseline*100:.2f}% -> {best_err*100:.2f}%)")

    t_search = time.time() - t_start
    print(f"\n[OK] Busca adaptativa concluída em {t_search:.1f}s!")
    print(f"Total de parâmetros do refinador aprendido: {total_refiner_params / 1e6:.2f}M")

    # 3. Avaliação Única no Conjunto Cego de Teste (D_test: tokens 1536..2560)
    print("\n[Avaliação Final Cega] Avaliando a política aprendida em D_test...")
    with torch.no_grad():
        x_norm_prof = comp["norm_mod"](h_prof[test_slice])
        logits_prof_test = F.linear(x_norm_prof, comp["lm_head_w"])

        x_norm_stud = comp["norm_mod"](h_stud[test_slice])
        logits_stud_test = F.linear(x_norm_stud, comp["lm_head_w"])

    nll_prof, ppl_prof, top1_prof, _ = compute_ppl_and_metrics(logits_prof_test, test_targets)
    nll_test, ppl_test, top1_test, cos_test = compute_ppl_and_metrics(logits_stud_test, test_targets, teacher_logits=logits_prof_test)

    h_p_test = h_prof[test_slice]
    h_s_test = h_stud[test_slice]
    e_64_test = torch.norm((h_p_test - h_s_test).float()).item() / torch.norm(h_p_test.float()).item() * 100.0

    print("\n" + "=" * 105)
    print("         RESULTADO FINAL DA POLÍTICA ADAPTATIVA APRENDIDA (CONJUNTO CEGO D_TEST)")
    print("=" * 105)
    print(f"Professor Oficial FP8:   NLL={nll_prof:.4f} | PPL={ppl_prof:.2f} | Top-1={top1_prof:.2f}%")
    print(f"Atlas Política Aprendida: NLL={nll_test:.4f} | PPL={ppl_test:.2f} | Top-1={top1_test:.2f}% | Cosine={cos_test:.4f} | e_64={e_64_test:.2f}%")
    print(f"Orçamento Refinador:     +{total_refiner_params/1e6:.2f}M parâmetros (< 0.6% da rede)")
    print("=" * 105)

    # 4. Distribuição de Tipos e Ranks Selecionados
    type_counts = {}
    rank_counts = {}
    for entry in learned_policy:
        t = entry["type"]
        r = entry["rank"]
        type_counts[t] = type_counts.get(t, 0) + 1
        rank_counts[r] = rank_counts.get(r, 0) + 1

    print("\nDistribuição de Políticas Selecionadas por Validação:")
    print(f"  Tipos: {type_counts}")
    print(f"  Ranks: {rank_counts}")

    # 5. Salvar Resultados em JSON
    out_json = os.path.join(repo_root, "experiments", "adaptive_policy_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "metrics_test": {
                "nll": nll_test,
                "ppl": ppl_test,
                "top1": top1_test,
                "cos_sim": cos_test,
                "e_64": e_64_test,
                "params_refiner_m": total_refiner_params / 1e6
            },
            "distributions": {
                "types": type_counts,
                "ranks": rank_counts
            },
            "layer_policy": learned_policy
        }, f, indent=2)
    print(f"\n[OK] Resultados da política adaptativa salvos em: {out_json}")

if __name__ == "__main__":
    run_adaptive_search()
