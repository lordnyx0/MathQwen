# -*- coding: utf-8 -*-
"""
Teste 2: Validação End-to-End das 64 Camadas com PPL Terminal Real.
Mede a perplexidade da rede inteira no conjunto de teste cego do WikiText-2.
"""
import sys
import os
import time
import math
import torch
import torch.nn.functional as F
import safetensors.torch as st
from datasets import load_dataset
from transformers import AutoTokenizer

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from reference.loader import load_qwen_reference_components, load_layer_module, dequant, get_snapshot_dir
from atlas.config import AtlasConfig
from atlas.projection import compute_chart_atlas_bases, project_layer_weights
from atlas.residual import fit_svd_stabilizer

def compute_ppl_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    """Calcula NLL e PPL causal exata a partir dos logits."""
    vocab_size = logits.shape[-1]
    logits_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    targets_flat = targets.reshape(-1)
    nll = F.cross_entropy(logits_pred, targets_flat, reduction="mean").item()
    ppl = math.exp(nll)
    top1 = (logits_pred.argmax(dim=-1) == targets_flat).float().mean().item() * 100.0
    return nll, ppl, top1

def run_64layer_end_to_end_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("TESTE 2: BENCHMARK END-TO-END DAS 64 CAMADAS: ESTABILIZADOR LINEAR (SVD-64)")
    print("=" * 95)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Dados: 16 seqs de 64 tokens = 1.024 tokens de treino (calibração)
    #           16 seqs de 64 tokens = 1.024 tokens de teste cego (avaliação PPL)
    snapshot_dir = get_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 32
    seq_len = 64
    input_ids = tokens_all[:num_seqs * seq_len].view(num_seqs, seq_len).to(device)

    train_ids = input_ids[:16]
    test_ids = input_ids[16:]
    test_targets = test_ids[:, 1:].contiguous()

    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=num_seqs)
    cfg = comp["cfg"]

    x_0 = F.embedding(input_ids.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    # -------------------------------------------------------------
    # Execução das 64 camadas em streaming
    # Mantém os estados de:
    # 1. Professor Oficial
    # 2. Atlas Raw (r=2048 sem estabilizador)
    # 3. Atlas + SVD-64 (estabilizador linear nas 64 camadas)
    # -------------------------------------------------------------
    h_prof = x_0.clone()
    h_raw = x_0.clone()
    h_svd = x_0.clone()

    r_base = 2048
    r_corr = 64

    print("\nExecutando streaming pelas 64 camadas...")
    t_start = time.time()

    for chart_idx in range(16):
        start_l = chart_idx * 4
        end_l = start_l + 4

        chart_weights = {}
        for l in range(start_l, end_l):
            lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(lp, framework="pt") as f:
                is_attn = (l % 4 == 3)
                if is_attn:
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight_scale_inv"))
                else:
                    mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight_scale_inv"))
                gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv"))
                up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv"))
                down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv"))
                chart_weights[l] = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}

        bases = compute_chart_atlas_bases(chart_weights, r_base=r_base, device=device)

        for l in range(start_l, end_l):
            layer_mod = load_layer_module(l, cfg, device, snapshot_dir=snapshot_dir)

            # 1. Forward Professor Oficial
            with torch.no_grad():
                out = layer_mod(h_prof, position_embeddings=pos_emb)
                h_prof_next = out[0] if isinstance(out, tuple) else out

            # 2. Configurar pesos Atlas
            p_w = project_layer_weights(chart_weights[l], bases)
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
            layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
            layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
            layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

            # 3. Forward Atlas Raw
            with torch.no_grad():
                out = layer_mod(h_raw, position_embeddings=pos_emb)
                h_raw_next = out[0] if isinstance(out, tuple) else out

            # 4. Forward Atlas SVD-64
            with torch.no_grad():
                out = layer_mod(h_svd, position_embeddings=pos_emb)
                h_svd_in = out[0] if isinstance(out, tuple) else out

                # Calibração SVD analítica na partição de treino:
                x_tr = h_svd_in[:16].reshape(-1, 5120)
                y_tr = h_prof_next[:16].reshape(-1, 5120)
                W_down, W_up = fit_svd_stabilizer(x_tr, y_tr, r_corr=r_corr)

                # Aplica correção em toda a sequência:
                corr = torch.matmul(torch.matmul(h_svd_in, W_down), W_up)
                h_svd_next = h_svd_in + corr

            h_prof = h_prof_next
            h_raw = h_raw_next
            h_svd = h_svd_next

            del layer_mod
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} concluída.")

    t_total = time.time() - t_start
    print(f"\n[OK] Streaming completo de 64 camadas concluído em {t_total:.2f}s!")

    # -------------------------------------------------------------
    # Avaliação Terminal das Projeções de Logits no Teste Cego
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("              RESULTADOS FINAIS END-TO-END NO CONJUNTO DE TESTE CEGO")
    print("=" * 95)

    with torch.no_grad():
        # 1. Professor Oficial
        norm_prof = comp["norm_mod"](h_prof[16:])
        logits_prof = F.linear(norm_prof, comp["lm_head_w"])
        nll_prof, ppl_prof, top1_prof = compute_ppl_from_logits(logits_prof, test_targets)

        # 2. Atlas Raw
        norm_raw = comp["norm_mod"](h_raw[16:])
        logits_raw = F.linear(norm_raw, comp["lm_head_w"])
        nll_raw, ppl_raw, top1_raw = compute_ppl_from_logits(logits_raw, test_targets)

        # 3. Atlas + SVD-64
        norm_svd = comp["norm_mod"](h_svd[16:])
        logits_svd = F.linear(norm_svd, comp["lm_head_w"])
        nll_svd, ppl_svd, top1_svd = compute_ppl_from_logits(logits_svd, test_targets)

    print(f"{'Configuração':<35} | {'NLL':<10} | {'PPL Terminal':<15} | {'Top-1 Accuracy':<15}")
    print("-" * 95)
    print(f"{'Professor Oficial FP8 (Ref)':<35} | {nll_prof:8.4f}   | {ppl_prof:12.2f}    | {top1_prof:6.2f}%")
    print(f"{'Atlas Assimétrico Raw (r=2048)':<35} | {nll_raw:8.4f}   | {ppl_raw:12.2f}    | {top1_raw:6.2f}%")
    print(f"{'Atlas + SVD-64 (Estabilizado 64 Camadas)':<35} | {nll_svd:8.4f}   | {ppl_svd:12.2f}    | {top1_svd:6.2f}%")
    print("=" * 95)

    delta_ppl = ((ppl_svd - ppl_raw) / ppl_raw) * 100.0
    print(f"\n[OK] Redução de PPL pelo estabilizador SVD-64: {delta_ppl:.2f}% (de {ppl_raw:.1f} para {ppl_svd:.2f})!")

if __name__ == "__main__":
    run_64layer_end_to_end_benchmark()
