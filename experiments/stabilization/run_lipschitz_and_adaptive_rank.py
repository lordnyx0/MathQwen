# -*- coding: utf-8 -*-
"""Diagnóstico de Lipschitz e Benchmark de Rank Adaptativo por Profundidade:
1. Medição empírica do ganho local de Lipschitz G_l (aleatório e direcional) e L_l (sub-operador) nas 64 camadas.
2. Comparação direta entre 4 políticas:
   - Uniforme r=1536 (~12.3B params)
   - Uniforme r=2048 (~14.3B params)
   - Adaptativo com Reforço: r=1024 (L0..47), r=3072 (L48..55), r=1536 (L56..63) (~11.45B params)
   - Adaptativo com Ilha Densa: r=1024 (L0..47, L56..63), Denso Original (L48..55) (~11.9B params)
3. Avaliação final de PPL, NLL e divergência de logits no WikiText-2.
"""

import sys
import os
import glob
import time
import math
import json
import torch
import torch.nn.functional as F
import safetensors.torch as st
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5RMSNorm
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]

def dequant(w, s):
    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.float() * s_exp.float())

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("       GANHO DE LIPSCHITZ NAS 64 CAMADAS E BENCHMARK DE RANK ADAPTATIVO POR PROFUNDIDADE")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Corpus WikiText-2: 8 sequências x 64 tokens = 512 tokens (504 avaliados)
    print("\n[1/5] Extraindo 512 tokens reais do WikiText-2 (8 seqs x 64 tokens)...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 8
    seq_len = 64
    total_tokens = num_seqs * seq_len
    input_ids = tokens_all[:total_tokens].view(num_seqs, seq_len).to(device)

    # 2. Embeddings e RoPE
    print("\n[2/5] Carregando embeddings, RMSNorm e lm_head...")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    with st.safe_open(outside_path, framework="pt") as f:
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
        norm_w = f.get_tensor("model.language_model.norm.weight").to(device=device, dtype=torch.bfloat16)
        x_0 = F.embedding(input_ids.cpu(), embed_w).to(device=device, dtype=torch.bfloat16)
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)

    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
    pos_emb = rotary(x_0, pos_ids)

    norm_mod = Qwen3_5RMSNorm(cfg.text_config.hidden_size, eps=cfg.text_config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm_mod.weight.data.copy_(norm_w)

    def step_fwd(mod, x_in):
        out = mod(x_in, position_embeddings=pos_emb)
        return out[0] if isinstance(out, tuple) else out

    # Políticas avaliadas
    policies = ["orig", "uniform_1536", "uniform_2048", "adapt_boost", "adapt_dense_island"]
    h_states = {p: x_0.clone() for p in policies}

    layer_metrics = []

    print("\n[3/5] Executando streaming pelas 64 camadas e computando ganhos locais...")
    t_start = time.time()

    for chart_idx in range(16):
        t_chart = time.time()
        start_l = chart_idx * 4
        end_l = start_l + 4

        # Carregar pesos
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

        # Gram matrices
        G_mix = torch.zeros(5120, 5120, device=device, dtype=torch.float32)
        G_down = torch.zeros(5120, 5120, device=device, dtype=torch.float32)
        G_joint = torch.zeros(5120, 5120, device=device, dtype=torch.float32)

        for l in range(start_l, end_l):
            W_m = chart_weights[l]["mix"].to(device)
            G_mix += torch.matmul(W_m, W_m.t())
            del W_m
            W_g = chart_weights[l]["gate"].to(device)
            W_u = chart_weights[l]["up"].to(device)
            G_joint += torch.matmul(W_g.t(), W_g) + torch.matmul(W_u.t(), W_u)
            del W_g, W_u
            W_d = chart_weights[l]["down"].to(device)
            G_down += torch.matmul(W_d, W_d.t())
            del W_d

        _, U_mix_full = torch.linalg.eigh(G_mix)
        _, U_down_full = torch.linalg.eigh(G_down)
        _, V_joint_full = torch.linalg.eigh(G_joint)
        del G_mix, G_down, G_joint
        torch.cuda.empty_cache()

        for l in range(start_l, end_l):
            layer_mod = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=l).to(device=device, dtype=torch.bfloat16)
            lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(lp, framework="pt") as f:
                layer_mod.input_layernorm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.input_layernorm.weight").to(device=device, dtype=torch.bfloat16))
                layer_mod.post_attention_layernorm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.post_attention_layernorm.weight").to(device=device, dtype=torch.bfloat16))
                if chart_weights[l]["is_attn"]:
                    layer_mod.self_attn.q_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                    layer_mod.self_attn.k_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                    layer_mod.self_attn.v_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                    layer_mod.self_attn.q_norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_norm.weight").to(device=device, dtype=torch.bfloat16))
                    layer_mod.self_attn.k_norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_norm.weight").to(device=device, dtype=torch.bfloat16))
                else:
                    layer_mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.in_proj_z.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.in_proj_a.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_a.weight").to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.in_proj_b.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_b.weight").to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.conv1d.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.conv1d.weight").to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.dt_bias.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.dt_bias").to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.A_log.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.A_log").to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.norm.weight").to(device=device, dtype=torch.bfloat16))

            W_mix_orig = chart_weights[l]["mix"].to(device=device, dtype=torch.bfloat16)
            W_gate_orig = chart_weights[l]["gate"].to(device=device, dtype=torch.bfloat16)
            W_up_orig = chart_weights[l]["up"].to(device=device, dtype=torch.bfloat16)
            W_down_orig = chart_weights[l]["down"].to(device=device, dtype=torch.bfloat16)

            # Setup modelo oficial
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(W_mix_orig)
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(W_mix_orig)
            layer_mod.mlp.gate_proj.weight.data.copy_(W_gate_orig)
            layer_mod.mlp.up_proj.weight.data.copy_(W_up_orig)
            layer_mod.mlp.down_proj.weight.data.copy_(W_down_orig)

            h_prev_orig = h_states["orig"].clone()

            # 1. Forward Oficial
            with torch.no_grad():
                h_next_orig = step_fwd(layer_mod, h_prev_orig)
            h_states["orig"] = h_next_orig

            norm_h_prev = torch.norm(h_prev_orig.float()).item()
            norm_h_next = torch.norm(h_next_orig.float()).item()

            # 2. Estimativa Rigorosa de Ganho de Lipschitz no Modelo Original
            # Perturbação isotrópica aleatória (5% da norma de h)
            torch.manual_seed(42 + l)
            u_rand = torch.randn_like(h_prev_orig)
            eps_rand = 0.05 * (u_rand / torch.norm(u_rand.float())) * norm_h_prev
            with torch.no_grad():
                h_next_pert = step_fwd(layer_mod, h_prev_orig + eps_rand)

            norm_eps = torch.norm(eps_rand.float()).item()
            diff_step = torch.norm((h_next_pert - h_next_orig).float()).item()
            G_step_rand = diff_step / norm_eps  # Ganho da camada inteira T_l(h) = h + F_l(h)

            # Ganho do sub-operador residual F_l(h) isolado
            # F_l(h + eps) - F_l(h) = (T_l(h+eps) - (h+eps)) - (T_l(h) - h) = (T_l(h+eps) - T_l(h)) - eps
            diff_F = torch.norm(((h_next_pert - (h_prev_orig + eps_rand)) - (h_next_orig - h_prev_orig)).float()).item()
            L_sub_rand = diff_F / norm_eps

            # Ganho direcional ao longo da deriva real de h_states["uniform_2048"]
            drift_vec = h_states["uniform_2048"] - h_prev_orig
            norm_drift = torch.norm(drift_vec.float()).item()
            if norm_drift > 1e-6:
                with torch.no_grad():
                    h_next_drift = step_fwd(layer_mod, h_states["uniform_2048"])
                G_step_drift = (torch.norm((h_next_drift - h_next_orig).float()).item()) / norm_drift
            else:
                G_step_drift = 1.0

            # 3. Propagação das Políticas Fatoradas / Adaptativas
            def apply_factorization(r_val):
                U_m = U_mix_full[:, -r_val:].to(dtype=torch.bfloat16)
                U_d = U_down_full[:, -r_val:].to(dtype=torch.bfloat16)
                V_j = V_joint_full[:, -r_val:].to(dtype=torch.bfloat16)
                W_mix_r = torch.matmul(U_m, torch.matmul(U_m.t(), W_mix_orig))
                W_gate_r = torch.matmul(torch.matmul(W_gate_orig, V_j), V_j.t())
                W_up_r = torch.matmul(torch.matmul(W_up_orig, V_j), V_j.t())
                W_down_r = torch.matmul(U_d, torch.matmul(U_d.t(), W_down_orig))
                if chart_weights[l]["is_attn"]:
                    layer_mod.self_attn.o_proj.weight.data.copy_(W_mix_r)
                else:
                    layer_mod.linear_attn.out_proj.weight.data.copy_(W_mix_r)
                layer_mod.mlp.gate_proj.weight.data.copy_(W_gate_r)
                layer_mod.mlp.up_proj.weight.data.copy_(W_up_r)
                layer_mod.mlp.down_proj.weight.data.copy_(W_down_r)

            def restore_orig_weights():
                if chart_weights[l]["is_attn"]:
                    layer_mod.self_attn.o_proj.weight.data.copy_(W_mix_orig)
                else:
                    layer_mod.linear_attn.out_proj.weight.data.copy_(W_mix_orig)
                layer_mod.mlp.gate_proj.weight.data.copy_(W_gate_orig)
                layer_mod.mlp.up_proj.weight.data.copy_(W_up_orig)
                layer_mod.mlp.down_proj.weight.data.copy_(W_down_orig)

            # A. Uniforme r=1536
            apply_factorization(1536)
            with torch.no_grad():
                h_states["uniform_1536"] = step_fwd(layer_mod, h_states["uniform_1536"])

            # B. Uniforme r=2048
            apply_factorization(2048)
            with torch.no_grad():
                h_states["uniform_2048"] = step_fwd(layer_mod, h_states["uniform_2048"])

            # C. Adaptativo com Reforço
            # Região crítica R = [48..55] usa r=3072; camadas finais [56..63] usam r=1536; demais usam r=1024
            if 48 <= l <= 55:
                r_adapt = 3072
            elif l >= 56:
                r_adapt = 1536
            else:
                r_adapt = 1024
            apply_factorization(r_adapt)
            with torch.no_grad():
                h_states["adapt_boost"] = step_fwd(layer_mod, h_states["adapt_boost"])

            # D. Adaptativo com Ilha Densa
            # Camadas [48..55] usam pesos ORIGINAIS DENSOS; demais usam r=1024
            if 48 <= l <= 55:
                restore_orig_weights()
            else:
                apply_factorization(1024)
            with torch.no_grad():
                h_states["adapt_dense_island"] = step_fwd(layer_mod, h_states["adapt_dense_island"])

            # Métricas da camada l
            e_u1536 = (torch.norm((h_states["uniform_1536"] - h_next_orig).float()) / norm_h_next).item()
            e_u2048 = (torch.norm((h_states["uniform_2048"] - h_next_orig).float()) / norm_h_next).item()
            e_boost = (torch.norm((h_states["adapt_boost"] - h_next_orig).float()) / norm_h_next).item()
            e_dense = (torch.norm((h_states["adapt_dense_island"] - h_next_orig).float()) / norm_h_next).item()

            cos_u2048 = F.cosine_similarity(h_states["uniform_2048"].float().flatten(), h_next_orig.float().flatten(), dim=0).item()
            cos_boost = F.cosine_similarity(h_states["adapt_boost"].float().flatten(), h_next_orig.float().flatten(), dim=0).item()
            cos_dense = F.cosine_similarity(h_states["adapt_dense_island"].float().flatten(), h_next_orig.float().flatten(), dim=0).item()

            layer_info = {
                "layer": l,
                "is_attn": chart_weights[l]["is_attn"],
                "norm_h": norm_h_next,
                "G_step_rand": G_step_rand,
                "G_step_drift": G_step_drift,
                "L_sub_rand": L_sub_rand,
                "e_u1536": e_u1536,
                "e_u2048": e_u2048,
                "e_boost": e_boost,
                "e_dense": e_dense,
                "cos_u2048": cos_u2048,
                "cos_boost": cos_boost,
                "cos_dense": cos_dense
            }
            layer_metrics.append(layer_info)

            is_crit = " [CRIT]" if 48 <= l <= 55 else ""
            print(f"  L{l:02d}{is_crit} | G_rand={G_step_rand:.3f} | G_drift={G_step_drift:.3f} | L_sub={L_sub_rand:.3f} | e_2048={e_u2048*100:5.1f}% | e_boost={e_boost*100:5.1f}% | e_dense={e_dense*100:5.1f}% | cos_dense={cos_dense:.3f}")

            del layer_mod, W_mix_orig, W_gate_orig, W_up_orig, W_down_orig
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} concluida em {time.time() - t_chart:.2f}s")

    t_total = time.time() - t_start
    print(f"\n[OK] Streaming das 64 camadas concluido em {t_total:.2f}s ({t_total/64:.2f}s/camada)!")

    # 4. Avaliação Final de Perplexidade e Divergência de Logits
    print("\n[4/5] Projetando logits via lm_head e calculando PPL/NLL...")
    targets = input_ids[:, 1:].contiguous()  # [8, 63] -> 504 tokens

    final_results = {}
    for p in policies:
        h_last = norm_mod(h_states[p])[:, :-1, :].contiguous()  # [8, 63, 5120]
        logits = F.linear(h_last, lm_head_w).float()  # [8, 63, 248320]

        nll = F.cross_entropy(logits.view(-1, 248320), targets.view(-1)).item()
        ppl = math.exp(min(nll, 20.0))

        if p == "orig":
            logits_orig = logits
            final_results[p] = {
                "policy": p,
                "params_b": 26.74,
                "nll": nll,
                "ppl": ppl,
                "delta_nll": 0.0,
                "delta_ppl": 0.0,
                "cos_z": 1.0,
                "top1_pct": 100.0
            }
        else:
            cos_z = F.cosine_similarity(logits.view(-1), logits_orig.view(-1), dim=0).item()
            top1_orig = logits_orig.argmax(dim=-1)
            top1_p = logits.argmax(dim=-1)
            top1_pct = (top1_p == top1_orig).float().mean().item() * 100.0

            # Contagem precisa de parâmetros
            if p == "uniform_1536":
                p_b = 12.29
            elif p == "uniform_2048":
                p_b = 14.29
            elif p == "adapt_boost":
                p_b = 11.45  # 48x1024 + 8x3072 + 8x1536
            elif p == "adapt_dense_island":
                p_b = 11.92  # 56x1024 + 8 densas
            else:
                p_b = 0.0

            final_results[p] = {
                "policy": p,
                "params_b": p_b,
                "nll": nll,
                "ppl": ppl,
                "delta_nll": nll - final_results["orig"]["nll"],
                "delta_ppl": ppl - final_results["orig"]["ppl"],
                "cos_z": cos_z,
                "top1_pct": top1_pct,
                "e_final": layer_metrics[-1][f"e_{'u1536' if '1536' in p else 'u2048' if '2048' in p else 'boost' if 'boost' in p else 'dense'}"]
            }

    print("\n" + "=" * 115)
    print("                 TABELA COMPARATIVA FINAL: UNIFORME vs ADAPTATIVO")
    print("=" * 115)
    print(f"{'Politica':<22} | {'Params':<8} | {'PPL':<11} | {'d_PPL':<11} | {'d_NLL':<8} | {'cos(z)':<8} | {'Top-1%':<7} | {'e_final':<8}")
    print("-" * 115)
    for p in policies:
        r = final_results[p]
        e_str = f"{r.get('e_final', 0.0)*100:6.2f}%" if p != "orig" else "  0.00%"
        print(f"{r['policy']:<22} | {r['params_b']:5.2f}B  | {r['ppl']:11.2f} | {r['delta_ppl']:+11.2f} | {r['delta_nll']:+7.4f} | {r['cos_z']:8.4f} | {r['top1_pct']:6.2f}% | {e_str}")
    print("=" * 115)

    # 5. Salvar e Gerar Gráficos
    print("\n[5/5] Salvando dados e gerando graficos...")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "lipschitz_and_adaptive_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"layer_metrics": layer_metrics, "final_results": final_results}, f, indent=2)

    # Gráfico 1: Perfil de Lipschitz
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    layers = [m["layer"] for m in layer_metrics]
    ax1.plot(layers, [m["G_step_rand"] for m in layer_metrics], color="#2980b9", linewidth=2.2, label="G_step (Perturbação Randômica Isotrópica)")
    ax1.plot(layers, [m["G_step_drift"] for m in layer_metrics], color="#e74c3c", linewidth=2.2, label="G_step (Ao longo da Deriva Real de Trajetória)")
    ax1.plot(layers, [m["L_sub_rand"] for m in layer_metrics], color="#8e44ad", linestyle="--", linewidth=1.8, label="L_sub (Sub-operador Residual F_l)")
    ax1.axhline(y=1.0, color="#34495e", linestyle=":", linewidth=2.0, label="G = 1.0 (Limiar de Estabilidade)")
    ax1.axvspan(48, 55, color="#f39c12", alpha=0.2, label="Região Crítica L48-55 (Cartas 12 e 13)")
    ax1.set_ylabel("Ganho / Fator de Escala", fontsize=12, fontweight="bold")
    ax1.set_title("Medição Rigorosa do Ganho de Lipschitz nas 64 Camadas Oficiais do Qwen 3.8 27B", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gráfico 2: Divergência e_l comparativa
    ax2.plot(layers, [m["e_u1536"] * 100 for m in layer_metrics], color="#95a5a6", linewidth=2.0, label="Uniforme r=1536 (12.29B)")
    ax2.plot(layers, [m["e_u2048"] * 100 for m in layer_metrics], color="#e67e22", linewidth=2.2, label="Uniforme r=2048 (14.29B)")
    ax2.plot(layers, [m["e_boost"] * 100 for m in layer_metrics], color="#27ae60", linewidth=2.5, label="Adaptativo Reforço r=3072 em L48-55 (11.45B)")
    ax2.plot(layers, [m["e_dense"] * 100 for m in layer_metrics], color="#2980b9", linewidth=2.5, linestyle="-.", label="Adaptativo Ilha Densa em L48-55 (11.92B)")
    ax2.axvspan(48, 55, color="#f39c12", alpha=0.2)
    ax2.set_xlabel("Índice da Camada (0 .. 63)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Erro Cumulativo e_l (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Comparação da Trajetória de Erro: Uniforme vs Estratégias Adaptativas por Profundidade", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "lipschitz_and_adaptive_profiles.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    root_dir = os.path.dirname(out_dir)
    import shutil
    shutil.copy2(plot_path, os.path.join(root_dir, "lipschitz_and_adaptive_profiles.png"))
    print("Grafico salvo com sucesso!")

if __name__ == "__main__":
    run_experiment()
