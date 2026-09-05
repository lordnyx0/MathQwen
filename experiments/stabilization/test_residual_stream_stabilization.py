# -*- coding: utf-8 -*-
"""Teste de Estabilização Matemática do Residual Stream:
Avalia se intervenções lineares diretamente no espaço de ativação h_l (5120)
conseguem conter a amplificação do erro através das 64 camadas:

Intervenções testadas no fluxo residual h_l:
1. raw_r2048:      h' = h (baseline sem correção)
2. scalar_r2048:   h' = alpha_l * h (1 escalar por camada = 64 floats)
3. diag_r2048:     h' = s_l * h (escala diagonal por canal, 5120 parâmetros por camada)
4. lowrank_r32:    h' = h + A_l (B_l^T h) com r_corr = 32 (~21M params no total)
5. lowrank_r64:    h' = h + A_l (B_l^T h) com r_corr = 64 (~42M params no total)
6. lowrank_r128:   h' = h + A_l (B_l^T h) com r_corr = 128 (~84M params no total)

Protocolo Científico:
- Batch A (256 tokens do WikiText-2): Calibração dos parâmetros alpha_l, s_l, C_l
- Batch B (256 tokens NÃO VISTOS do WikiText-2): Teste rigoroso de generalização
Mede: erro cumulativo e_l camada-a-camada e PPL final em tokens não vistos!
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

def fit_low_rank_corrector(X_calib, Y_calib, r_corr, lambda_reg=1e-3):
    # X_calib: [N, d] (ativação fatorada não corrigida)
    # Y_calib: [N, d] (ativação de referência oficial)
    E = Y_calib - X_calib  # [N, d]
    N, d = X_calib.shape

    U, S, Vh = torch.linalg.svd(X_calib.float(), full_matrices=False)  # U: [N, N], S: [N], Vh: [N, d]
    V = Vh.t()  # [d, N]

    reg = lambda_reg * (S**2).mean()
    # M = E^T * U * diag(S / (S^2 + reg))  -> shape: [d, N]
    weights = S / (S**2 + reg)
    M = torch.matmul(E.float().t(), U * weights)  # [d, N]

    Um, Sm, Vhm = torch.linalg.svd(M, full_matrices=False)  # Um: [d, N], Sm: [N], Vhm: [N, N]
    Wr = torch.matmul(V, Vhm.t())  # [d, N]

    k = min(r_corr, N)
    A = (Um[:, :k] * torch.sqrt(Sm[:k])).to(dtype=X_calib.dtype)  # [d, k]
    B = (Wr[:, :k] * torch.sqrt(Sm[:k])).to(dtype=X_calib.dtype)  # [d, k]

    return A, B

def run_stabilization_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("       TESTE DE ESTABILIZAÇÃO DO RESIDUAL STREAM: ESCALA, DIAGONAL E LOW-RANK")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Corpus: 8 seqs de 64 tokens = 512 tokens
    # Seq 0..3: Calibração (Batch A, 256 tokens)
    # Seq 4..7: Teste/Generalização (Batch B, 256 tokens)
    print("\n[1/5] Preparando 512 tokens do WikiText-2 (256 calibração + 256 teste)...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 8
    seq_len = 64
    total_tokens = num_seqs * seq_len
    input_ids = tokens_all[:total_tokens].view(num_seqs, seq_len).to(device)

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

    # Métodos avaliados (todos usam a mesma base de pesos fatorados r=2048)
    methods = ["orig", "raw_r2048", "scalar_r2048", "diag_r2048", "lowrank_r32", "lowrank_r64", "lowrank_r128"]
    h_states = {m: x_0.clone() for m in methods}

    layer_metrics = []

    print("\n[2/5] Executando streaming e calibrando estabilizadores camada por camada...")
    t_start = time.time()

    r_base = 2048

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

        # Gram matrices para r=2048
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

        _, U_mix = torch.linalg.eigh(G_mix)
        _, U_down = torch.linalg.eigh(G_down)
        _, V_joint = torch.linalg.eigh(G_joint)
        del G_mix, G_down, G_joint
        torch.cuda.empty_cache()

        U_m = U_mix[:, -r_base:].to(dtype=torch.bfloat16)
        U_d = U_down[:, -r_base:].to(dtype=torch.bfloat16)
        V_j = V_joint[:, -r_base:].to(dtype=torch.bfloat16)
        del U_mix, U_down, V_joint

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

            # 1. Forward Modelo Oficial
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(W_mix_orig)
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(W_mix_orig)
            layer_mod.mlp.gate_proj.weight.data.copy_(W_gate_orig)
            layer_mod.mlp.up_proj.weight.data.copy_(W_up_orig)
            layer_mod.mlp.down_proj.weight.data.copy_(W_down_orig)

            with torch.no_grad():
                h_states["orig"] = step_fwd(layer_mod, h_states["orig"])

            # 2. Configurar Pesos Fatorados r=2048
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

            # Separação Calibração (Batch A, seqs 0..3) e Teste (Batch B, seqs 4..7)
            y_orig = h_states["orig"]
            y_orig_calib = y_orig[:4].reshape(-1, 5120)  # [256, 5120]
            y_orig_test = y_orig[4:].reshape(-1, 5120)   # [256, 5120]

            # 3. Forwards e Calibrações dos Estabilizadores
            # A. raw_r2048 (sem correção)
            with torch.no_grad():
                h_states["raw_r2048"] = step_fwd(layer_mod, h_states["raw_r2048"])

            # B. scalar_r2048 (alpha_l)
            with torch.no_grad():
                h_raw_scalar = step_fwd(layer_mod, h_states["scalar_r2048"])
                x_calib = h_raw_scalar[:4].reshape(-1, 5120)
                alpha_l = (torch.sum(y_orig_calib * x_calib) / (torch.sum(x_calib * x_calib) + 1e-8)).item()
                h_states["scalar_r2048"] = alpha_l * h_raw_scalar

            # C. diag_r2048 (s_l por canal)
            with torch.no_grad():
                h_raw_diag = step_fwd(layer_mod, h_states["diag_r2048"])
                x_calib = h_raw_diag[:4].reshape(-1, 5120)
                # s_l por dimensão: dot(y[:, c], x[:, c]) / dot(x[:, c], x[:, c])
                s_l = torch.sum(y_orig_calib * x_calib, dim=0) / (torch.sum(x_calib * x_calib, dim=0) + 1e-8)
                h_states["diag_r2048"] = h_raw_diag * s_l.view(1, 1, 5120)

            # D. lowrank_r32
            with torch.no_grad():
                h_raw_r32 = step_fwd(layer_mod, h_states["lowrank_r32"])
                x_calib = h_raw_r32[:4].reshape(-1, 5120)
                A32, B32 = fit_low_rank_corrector(x_calib, y_orig_calib, r_corr=32)
                # Aplicar em todas as sequências: h' = h + (h @ B) @ A^T
                corr32 = torch.matmul(torch.matmul(h_raw_r32, B32), A32.t())
                h_states["lowrank_r32"] = h_raw_r32 + corr32

            # E. lowrank_r64
            with torch.no_grad():
                h_raw_r64 = step_fwd(layer_mod, h_states["lowrank_r64"])
                x_calib = h_raw_r64[:4].reshape(-1, 5120)
                A64, B64 = fit_low_rank_corrector(x_calib, y_orig_calib, r_corr=64)
                corr64 = torch.matmul(torch.matmul(h_raw_r64, B64), A64.t())
                h_states["lowrank_r64"] = h_raw_r64 + corr64

            # F. lowrank_r128
            with torch.no_grad():
                h_raw_r128 = step_fwd(layer_mod, h_states["lowrank_r128"])
                x_calib = h_raw_r128[:4].reshape(-1, 5120)
                A128, B128 = fit_low_rank_corrector(x_calib, y_orig_calib, r_corr=128)
                corr128 = torch.matmul(torch.matmul(h_raw_r128, B128), A128.t())
                h_states["lowrank_r128"] = h_raw_r128 + corr128

            # Métricas no conjunto de TESTE (tokens não vistos!)
            norm_test = torch.norm(y_orig_test.float()).item()
            layer_info = {"layer": l, "is_attn": chart_weights[l]["is_attn"], "alpha_l": alpha_l}

            for m in methods:
                if m != "orig":
                    h_test = h_states[m][4:].reshape(-1, 5120)
                    e_test = (torch.norm((h_test - y_orig_test).float()) / norm_test).item()
                    cos_test = F.cosine_similarity(h_test.float().flatten(), y_orig_test.float().flatten(), dim=0).item()
                    layer_info[f"e_{m}"] = e_test
                    layer_info[f"cos_{m}"] = cos_test

            layer_metrics.append(layer_info)

            print(f"  L{l:02d} | alpha={alpha_l:.4f} | e_raw={layer_info['e_raw_r2048']*100:5.1f}% | e_scalar={layer_info['e_scalar_r2048']*100:5.1f}% | e_diag={layer_info['e_diag_r2048']*100:5.1f}% | e_r32={layer_info['e_lowrank_r32']*100:5.1f}% | e_r64={layer_info['e_lowrank_r64']*100:5.1f}% | e_r128={layer_info['e_lowrank_r128']*100:5.1f}%")

            del layer_mod, W_mix_orig, W_gate_orig, W_up_orig, W_down_orig, W_mix_r, W_gate_r, W_up_r, W_down_r
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} concluida em {time.time() - t_chart:.2f}s")

    t_total = time.time() - t_start
    print(f"\n[OK] Streaming e calibração concluídos em {t_total:.2f}s ({t_total/64:.2f}s/camada)!")

    # 4. Avaliação de Perplexidade no Conjunto de Teste (Tokens Não Vistos!)
    print("\n[3/5] Calculando PPL e Alinhamento de Logits no conjunto de TESTE (tokens não vistos)...")
    test_targets = input_ids[4:, 1:].contiguous()  # [4, 63] -> 252 tokens

    final_results = {}
    for m in methods:
        h_last = norm_mod(h_states[m][4:])[:, :-1, :].contiguous()  # [4, 63, 5120]
        logits = F.linear(h_last, lm_head_w).float()

        nll = F.cross_entropy(logits.view(-1, 248320), test_targets.view(-1)).item()
        ppl = math.exp(min(nll, 20.0))

        if m == "orig":
            logits_orig = logits
            final_results[m] = {
                "method": m,
                "extra_params_m": 0.0,
                "nll": nll,
                "ppl": ppl,
                "delta_nll": 0.0,
                "delta_ppl": 0.0,
                "cos_z": 1.0,
                "top1_pct": 100.0,
                "e_final": 0.0
            }
        else:
            cos_z = F.cosine_similarity(logits.view(-1), logits_orig.view(-1), dim=0).item()
            top1_orig = logits_orig.argmax(dim=-1)
            top1_m = logits.argmax(dim=-1)
            top1_pct = (top1_m == top1_orig).float().mean().item() * 100.0

            if m == "raw_r2048":
                extra_p = 0.0
            elif m == "scalar_r2048":
                extra_p = 64 / 1e6
            elif m == "diag_r2048":
                extra_p = (64 * 5120) / 1e6
            elif m == "lowrank_r32":
                extra_p = (64 * 2 * 5120 * 32) / 1e6  # 20.97M
            elif m == "lowrank_r64":
                extra_p = (64 * 2 * 5120 * 64) / 1e6  # 41.94M
            elif m == "lowrank_r128":
                extra_p = (64 * 2 * 5120 * 128) / 1e6  # 83.89M

            final_results[m] = {
                "method": m,
                "extra_params_m": extra_p,
                "nll": nll,
                "ppl": ppl,
                "delta_nll": nll - final_results["orig"]["nll"],
                "delta_ppl": ppl - final_results["orig"]["ppl"],
                "cos_z": cos_z,
                "top1_pct": top1_pct,
                "e_final": layer_metrics[-1][f"e_{m}"]
            }

    print("\n" + "=" * 115)
    print("        RESULTADO DO TESTE DE ESTABILIZAÇÃO NO CONJUNTO NÃO VISTO (TEST SET)")
    print("=" * 115)
    print(f"{'Metodo':<18} | {'Extra Params':<13} | {'PPL (Test)':<12} | {'d_PPL':<12} | {'d_NLL':<8} | {'cos(z)':<8} | {'Top-1%':<7} | {'e_64':<8}")
    print("-" * 115)
    for m in methods:
        r = final_results[m]
        p_str = f"+{r['extra_params_m']:6.2f}M" if r['extra_params_m'] > 0 else "    0.00M"
        e_str = f"{r['e_final']*100:6.2f}%" if m != "orig" else "  0.00%"
        print(f"{r['method']:<18} | {p_str:<13} | {r['ppl']:12.2f} | {r['delta_ppl']:+12.2f} | {r['delta_nll']:+7.4f} | {r['cos_z']:8.4f} | {r['top1_pct']:6.2f}% | {e_str}")
    print("=" * 115)

    # 5. Salvar e Gerar Gráficos
    print("\n[5/5] Salvando dados e gerando graficos...")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "residual_stream_stabilization_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"layer_metrics": layer_metrics, "final_results": final_results}, f, indent=2)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    layers = [m["layer"] for m in layer_metrics]
    ax1.plot(layers, [m["e_raw_r2048"] * 100 for m in layer_metrics], color="#e74c3c", linewidth=2.5, label="raw_r2048 (Sem Correção)")
    ax1.plot(layers, [m["e_scalar_r2048"] * 100 for m in layer_metrics], color="#f39c12", linewidth=2.0, linestyle="--", label="scalar_r2048 (Escala alpha_l)")
    ax1.plot(layers, [m["e_diag_r2048"] * 100 for m in layer_metrics], color="#8e44ad", linewidth=2.0, linestyle="-.", label="diag_r2048 (Diagonal s_l)")
    ax1.plot(layers, [m["e_lowrank_r32"] * 100 for m in layer_metrics], color="#3498db", linewidth=2.2, label="lowrank_r32 (+21M params)")
    ax1.plot(layers, [m["e_lowrank_r64"] * 100 for m in layer_metrics], color="#2ecc71", linewidth=2.4, label="lowrank_r64 (+42M params)")
    ax1.plot(layers, [m["e_lowrank_r128"] * 100 for m in layer_metrics], color="#16a085", linewidth=2.6, label="lowrank_r128 (+84M params)")

    ax1.axhline(y=100.0, color="#7f8c8d", linestyle=":", label="100% de Erro")
    ax1.set_ylabel("Erro de Ativação e_l (%) em Teste", fontsize=12, fontweight="bold")
    ax1.set_title("Efeito das Intervenções no Residual Stream ao Longo das 64 Camadas (Conjunto de Teste)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gráfico 2: Fator de Escala alpha_l por camada
    ax2.plot(layers, [m["alpha_l"] for m in layer_metrics], color="#d35400", linewidth=2.2, label="Escala Ótima alpha_l")
    ax2.axhline(y=1.0, color="#2c3e50", linestyle=":", label="alpha = 1.0 (Sem Amortecimento)")
    ax2.set_xlabel("Índice da Camada (0 .. 63)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Valor de alpha_l", fontsize=12, fontweight="bold")
    ax2.set_title("Evolução do Fator de Escala Ótimo alpha_l ao Longo da Profundidade", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "residual_stream_stabilization.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    root_dir = os.path.dirname(out_dir)
    import shutil
    shutil.copy2(plot_path, os.path.join(root_dir, "residual_stream_stabilization.png"))
    print("Grafico salvo com sucesso!")

if __name__ == "__main__":
    run_stabilization_benchmark()
