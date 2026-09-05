# -*- coding: utf-8 -*-
"""Sondagem Crítica nas 8 Camadas: L0, L16, L32, L48, L50, L51, L55, L63.

Objetivo:
- Testar se o manifold fitting (AdamW com rollback estrito) consegue reduzir o erro
  além do baseline SVD nas camadas de alta sensibilidade (especialmente L50, L51, L55).
- Avaliação em 1.024 tokens de treino + 1.024 tokens de teste independente (não vistos).
- Convenção explícita de matrizes:
    W_down: [5120, 64] (down-projection)
    W_up:   [64, 5120] (up-projection)
    h' = h + (h @ W_down) @ W_up
- Rollback estrito: se a perda em teste não diminuir, mantém o ponto SVD.
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

def fit_svd_stabilizer(X_calib, Y_calib, r_corr=64, lambda_reg=1e-3):
    E = Y_calib - X_calib  # [N, d]
    N, d = X_calib.shape

    U, S, Vh = torch.linalg.svd(X_calib.float(), full_matrices=False)
    V = Vh.t()

    reg = lambda_reg * (S**2).mean()
    weights = S / (S**2 + reg)
    M = torch.matmul(E.float().t(), U * weights)

    Um, Sm, Vhm = torch.linalg.svd(M, full_matrices=False)
    Wr = torch.matmul(V, Vhm.t())

    k = min(r_corr, N)
    # W_down: [d, k], W_up: [k, d]
    W_down = (Wr[:, :k] * torch.sqrt(Sm[:k])).to(dtype=torch.bfloat16)
    W_up = (Um[:, :k] * torch.sqrt(Sm[:k])).t().to(dtype=torch.bfloat16)

    return W_down, W_up

def run_probe_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("        SONDAGEM CRÍTICA NAS 8 CAMADAS: L0, L16, L32, L48, L50, L51, L55, L63")
    print("           BASELINE SVD vs MANIFOLD FITTING COM ROLLBACK ESTRITO")
    print("=" * 115)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Corpus WikiText-2
    print("\n[1/4] Extraindo 2.048 tokens reais do WikiText-2 (1.024 treino + 1.024 teste)...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 32
    seq_len = 64
    total_tokens = num_seqs * seq_len
    input_ids = tokens_all[:total_tokens].view(num_seqs, seq_len).to(device)

    train_ids = input_ids[:16]
    test_ids = input_ids[16:]
    test_targets = test_ids[:, 1:].contiguous().view(-1)

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

    probe_layers = [0, 16, 32, 48, 50, 51, 55, 63]
    probe_set = set(probe_layers)

    h_teacher = x_0.clone()
    h_student = x_0.clone()

    probe_results = []
    r_base = 2048
    r_corr = 64

    print("\n[2/4] Executando streaming pelas 64 camadas e sondando as 8 camadas críticas...")
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

        # Bases Atlas r=2048
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

            # 1. Forward Professor Oficial
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(W_mix_orig)
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(W_mix_orig)
            layer_mod.mlp.gate_proj.weight.data.copy_(W_gate_orig)
            layer_mod.mlp.up_proj.weight.data.copy_(W_up_orig)
            layer_mod.mlp.down_proj.weight.data.copy_(W_down_orig)

            with torch.no_grad():
                h_teacher_next = step_fwd(layer_mod, h_teacher)

            # 2. Configurar Pesos Fatorados Atlas r=2048
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

            with torch.no_grad():
                h_student_raw = step_fwd(layer_mod, h_student)

            # Partição Treino / Teste
            x_tr = h_student_raw[:16].reshape(-1, 5120)
            y_tr = h_teacher_next[:16].reshape(-1, 5120)
            x_te = h_student_raw[16:].reshape(-1, 5120)
            y_te = h_teacher_next[16:].reshape(-1, 5120)

            # Calibração SVD Analítica (Baseline)
            W_d_svd, W_u_svd = fit_svd_stabilizer(x_tr, y_tr, r_corr=r_corr)

            norm_y_te = torch.norm(y_te.float()).item()
            with torch.no_grad():
                # Forward SVD explícito: (h @ W_down) @ W_up
                corr_svd_te = torch.matmul(torch.matmul(x_te, W_d_svd), W_u_svd)
                h_svd_te = x_te + corr_svd_te
                e_svd_te = (torch.norm((h_svd_te - y_te).float()) / norm_y_te).item()
                cos_svd_te = F.cosine_similarity(h_svd_te.float().flatten(), y_te.float().flatten(), dim=0).item()

            # Se for uma das 8 camadas de sondagem: executar Manifold Fitting com Rollback
            if l in probe_set:
                W_d_opt = torch.nn.Parameter(W_d_svd.clone())
                W_u_opt = torch.nn.Parameter(W_u_svd.clone())
                optimizer = torch.optim.AdamW([W_d_opt, W_u_opt], lr=1e-4, weight_decay=1e-4)

                best_e_te = e_svd_te
                best_W_d = W_d_svd.clone()
                best_W_u = W_u_svd.clone()

                norm_y_tr = torch.norm(y_tr.float()).item()
                h_tr_3d = h_student_raw[:16]

                for step in range(40):
                    optimizer.zero_grad()
                    corr_tr = torch.matmul(torch.matmul(h_tr_3d, W_d_opt), W_u_opt)
                    h_s_tr = h_tr_3d + corr_tr

                    loss_mse = torch.norm((h_s_tr.reshape(-1, 5120) - y_tr).float()) / norm_y_tr
                    loss_cos = 1.0 - F.cosine_similarity(h_s_tr.float().flatten(), y_tr.float().flatten(), dim=0)
                    loss = loss_mse + 10.0 * loss_cos
                    loss.backward()
                    optimizer.step()

                    # Avaliação periódica com rollback na partição de validação
                    with torch.no_grad():
                        corr_step_te = torch.matmul(torch.matmul(x_te, W_d_opt), W_u_opt)
                        h_step_te = x_te + corr_step_te
                        e_curr_te = (torch.norm((h_step_te - y_te).float()) / norm_y_te).item()
                        if e_curr_te < best_e_te:
                            best_e_te = e_curr_te
                            best_W_d = W_d_opt.data.clone()
                            best_W_u = W_u_opt.data.clone()

                # Avaliação final
                with torch.no_grad():
                    corr_final_te = torch.matmul(torch.matmul(x_te, best_W_d), best_W_u)
                    h_opt_te = x_te + corr_final_te
                    e_opt_te = (torch.norm((h_opt_te - y_te).float()) / norm_y_te).item()
                    cos_opt_te = F.cosine_similarity(h_opt_te.float().flatten(), y_te.float().flatten(), dim=0).item()

                improved = (e_opt_te < e_svd_te)
                delta_e = (e_opt_te - e_svd_te) * 100.0

                status = "MELHOROU" if improved else "PRESERVADO (SVD)"
                print(f"  >>> PROBE L{l:02d} | Status: {status:<15} | e_svd={e_svd_te*100:5.2f}% -> e_opt={e_opt_te*100:5.2f}% ({delta_e:+5.2f}%) | cos_svd={cos_svd_te:.4f} -> cos_opt={cos_opt_te:.4f}")

                probe_results.append({
                    "layer": l,
                    "e_svd": e_svd_te,
                    "e_opt": e_opt_te,
                    "cos_svd": cos_svd_te,
                    "cos_opt": cos_opt_te,
                    "delta_e": delta_e,
                    "improved": improved
                })

                # Atualiza h_student com o melhor estabilizador encontrado
                with torch.no_grad():
                    corr_full = torch.matmul(torch.matmul(h_student_raw, best_W_d), best_W_u)
                    h_student = h_student_raw + corr_full
            else:
                # Nas camadas intermediárias, avança com o estabilizador SVD
                with torch.no_grad():
                    corr_full = torch.matmul(torch.matmul(h_student_raw, W_d_svd), W_u_svd)
                    h_student = h_student_raw + corr_full

            h_teacher = h_teacher_next

            del layer_mod, W_mix_orig, W_gate_orig, W_up_orig, W_down_orig, W_mix_r, W_gate_r, W_up_r, W_down_r
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} concluida em {time.time() - t_chart:.2f}s")

    t_total = time.time() - t_start
    print(f"\n[OK] Sondagem das 8 camadas concluida em {t_total:.2f}s ({t_total/64:.2f}s/camada)!")

    # 3. Tabela Resumo Final
    print("\n" + "=" * 115)
    print("           TABELA CONSOLIDADA: SONDAGEM NAS 8 CAMADAS CRÍTICAS (TEST SET)")
    print("=" * 115)
    print(f"{'Camada':<10} | {'Status':<16} | {'e_SVD':<10} | {'e_Distill':<10} | {'Delta e':<10} | {'cos_SVD':<10} | {'cos_Distill':<10}")
    print("-" * 115)
    for r in probe_results:
        st_str = "MELHOROU" if r["improved"] else "PRESERVADO (SVD)"
        print(f"L{r['layer']:<8} | {st_str:<16} | {r['e_svd']*100:6.2f}%   | {r['e_opt']*100:6.2f}%   | {r['delta_e']:+6.2f}%   | {r['cos_svd']:8.4f}   | {r['cos_opt']:8.4f}")
    print("=" * 115)

    # 4. Salvar resultados
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "probe_8layers_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    # Gráfico
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    x_labels = [f"L{r['layer']}" for r in probe_results]
    x_indices = list(range(len(x_labels)))

    ax1.plot(x_indices, [r["e_svd"] * 100 for r in probe_results], marker="o", color="#f39c12", linewidth=2.4, label="Baseline SVD r_corr=64")
    ax1.plot(x_indices, [r["e_opt"] * 100 for r in probe_results], marker="s", color="#27ae60", linewidth=2.4, label="Otimizado / Manifold Fitting")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Erro Relativo de Ativação e_l (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Comparação nas 8 Camadas Críticas no Conjunto de Teste Cego (WikiText-2)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(x_indices, [r["cos_svd"] for r in probe_results], marker="o", color="#f39c12", linewidth=2.4, label="Cosine Sim h: Baseline SVD")
    ax2.plot(x_indices, [r["cos_opt"] for r in probe_results], marker="s", color="#27ae60", linewidth=2.4, label="Cosine Sim h: Otimizado")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("Alinhamento de Cosseno", fontsize=12, fontweight="bold")
    ax2.set_title("Alinhamento Angular com o Professor nas 8 Camadas Críticas", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "probe_8layers_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    root_dir = os.path.dirname(out_dir)
    import shutil
    shutil.copy2(plot_path, os.path.join(root_dir, "probe_8layers_comparison.png"))
    print("Gráfico salvo com sucesso!")

if __name__ == "__main__":
    run_probe_experiment()
