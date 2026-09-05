# -*- coding: utf-8 -*-
"""Diagnóstico de Divergência Profunda: Orig vs Joint (r=1024, 1536, 2048) Camada por Camada.

Mede:
1. Erro cumulativo de ativação: e_l = ||h_l^{joint} - h_l^{orig}|| / ||h_l^{orig}|| para cada camada l in [0..63]
2. Erro local de um único passo: delta_l = ||F_l^{joint}(h_{l-1}^{orig}) - F_l^{orig}(h_{l-1}^{orig})|| / ||h_{l-1}^{orig}||
3. Alinhamento angular: cos(h_l^{joint}, h_l^{orig})
4. Logit Lens em cada camada: cos(z_l^{joint}, z_l^{orig}) e Top-1 Agreement de logits intermediários
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

def run_layer_divergence_tracking():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print("     RASTREAMENTO CAMADA-A-CAMADA DA DIVERGÊNCIA: ORIG vs JOINT (r=1024, 1536, 2048)")
    print("=" * 110)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Preparar Corpus: 4 sequências x 64 tokens = 256 tokens reais
    print("\n[1/4] Extraindo 256 tokens reais do WikiText-2 (4 seqs x 64 tokens)...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    num_seqs = 4
    seq_len = 64
    total_tokens = num_seqs * seq_len
    input_ids = tokens_all[:total_tokens].view(num_seqs, seq_len).to(device)

    # 2. Embeddings, RMSNorm e lm_head
    print("\n[2/4] Carregando embeddings, RoPE, RMSNorm e lm_head na GPU...")
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

    ranks = [1024, 1536, 2048]
    model_keys = ["orig"] + [f"joint_r{r}" for r in ranks]

    h_states = {k: x_0.clone() for k in model_keys}

    def step_fwd(mod, x_in):
        out = mod(x_in, position_embeddings=pos_emb)
        return out[0] if isinstance(out, tuple) else out

    def get_logits(h):
        h_n = norm_mod(h)
        return F.linear(h_n, lm_head_w)

    history = []

    print("\n[3/4] Executando streaming e registrando divergência em cada uma das 64 camadas...")
    t_start = time.time()

    for chart_idx in range(16):
        t_chart = time.time()
        start_l = chart_idx * 4
        end_l = start_l + 4

        # Carregar pesos dos 4 operadores
        chart_weights = {}
        for l in range(start_l, end_l):
            lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(lp, framework="pt") as f:
                is_attn = (l % 4 == 3)
                if is_attn:
                    mix_w = dequant(
                        f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight"),
                        f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight_scale_inv")
                    )
                else:
                    mix_w = dequant(
                        f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight"),
                        f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight_scale_inv")
                    )
                gate_w = dequant(
                    f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"),
                    f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")
                )
                up_w = dequant(
                    f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"),
                    f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")
                )
                down_w = dequant(
                    f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"),
                    f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")
                )
                chart_weights[l] = {
                    "mix": mix_w,
                    "gate": gate_w,
                    "up": up_w,
                    "down": down_w,
                    "is_attn": is_attn
                }

        # Matrizes de Gram para bases conjuntas e unilaterais
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

        # Propagar camadas da carta
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

            # Pesos originais
            W_mix_orig = chart_weights[l]["mix"].to(device=device, dtype=torch.bfloat16)
            W_gate_orig = chart_weights[l]["gate"].to(device=device, dtype=torch.bfloat16)
            W_up_orig = chart_weights[l]["up"].to(device=device, dtype=torch.bfloat16)
            W_down_orig = chart_weights[l]["down"].to(device=device, dtype=torch.bfloat16)

            # 1. Forward Original
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(W_mix_orig)
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(W_mix_orig)
            layer_mod.mlp.gate_proj.weight.data.copy_(W_gate_orig)
            layer_mod.mlp.up_proj.weight.data.copy_(W_up_orig)
            layer_mod.mlp.down_proj.weight.data.copy_(W_down_orig)

            h_orig_prev = h_states["orig"].clone()
            with torch.no_grad():
                h_states["orig"] = step_fwd(layer_mod, h_orig_prev)

            norm_orig = torch.norm(h_states["orig"].float()).item()
            layer_entry = {
                "layer": l,
                "is_attn": chart_weights[l]["is_attn"],
                "norm_orig": norm_orig
            }

            # 2. Forwards Fatorados para r in [1024, 1536, 2048]
            for r in ranks:
                U_m = U_mix_full[:, -r:].to(dtype=torch.bfloat16)
                U_d = U_down_full[:, -r:].to(dtype=torch.bfloat16)
                V_j = V_joint_full[:, -r:].to(dtype=torch.bfloat16)

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

                # A. Erro Local (sem deriva acumulada, alimentando h_orig_prev)
                with torch.no_grad():
                    h_local_r = step_fwd(layer_mod, h_orig_prev)
                delta_l = (torch.norm((h_local_r - h_states["orig"]).float()) / norm_orig).item()
                layer_entry[f"delta_l_r{r}"] = delta_l

                # B. Erro Cumulativo (na trajetória derivada real h_states[f"joint_r{r}"])
                with torch.no_grad():
                    h_states[f"joint_r{r}"] = step_fwd(layer_mod, h_states[f"joint_r{r}"])

                e_l = (torch.norm((h_states[f"joint_r{r}"] - h_states["orig"]).float()) / norm_orig).item()
                layer_entry[f"e_l_r{r}"] = e_l

                # Similaridade angular no hidden state
                cos_h = F.cosine_similarity(
                    h_states[f"joint_r{r}"].float().flatten(),
                    h_states["orig"].float().flatten(),
                    dim=0
                ).item()
                layer_entry[f"cos_h_r{r}"] = cos_h

            # C. Logit Lens na Camada Atual (para orig e joint_r2048)
            with torch.no_grad():
                z_orig = get_logits(h_states["orig"])
                z_2048 = get_logits(h_states["joint_r2048"])

                norm_z_orig = torch.norm(z_orig.float()).item()
                e_logit_2048 = (torch.norm((z_2048 - z_orig).float()) / norm_z_orig).item()
                cos_z_2048 = F.cosine_similarity(z_2048.float().flatten(), z_orig.float().flatten(), dim=0).item()

                top1_orig = z_orig.argmax(dim=-1)
                top1_2048 = z_2048.argmax(dim=-1)
                top1_agree_2048 = (top1_orig == top1_2048).float().mean().item() * 100.0

            layer_entry["e_logit_r2048"] = e_logit_2048
            layer_entry["cos_z_r2048"] = cos_z_2048
            layer_entry["top1_z_r2048"] = top1_agree_2048

            history.append(layer_entry)

            op_type = "GQA " if chart_weights[l]["is_attn"] else "Delta"
            print(f"  L{l:02d} ({op_type}) | delta_loc(2048)={layer_entry['delta_l_r2048']*100:5.2f}% | e_l(1024)={layer_entry['e_l_r1024']*100:6.2f}% | e_l(1536)={layer_entry['e_l_r1536']*100:6.2f}% | e_l(2048)={layer_entry['e_l_r2048']*100:6.2f}% | cos_h(2048)={cos_h:.4f} | cos_z={cos_z_2048:.4f} | top1_z={top1_agree_2048:5.1f}%")

            del layer_mod, W_mix_orig, W_gate_orig, W_up_orig, W_down_orig, h_local_r, z_orig, z_2048
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} concluida em {time.time() - t_chart:.2f}s")

    t_total = time.time() - t_start
    print(f"\n[OK] Rastreamento das 64 camadas concluido em {t_total:.2f}s ({t_total/64:.2f}s/camada)!")

    # 4. Salvar dados
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "layer_divergence_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Resultados salvos em: {json_path}")

    # Gerar Gráficos Diagnósticos
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    layers = [h["layer"] for h in history]

    # Gráfico 1: Erro Cumulativo vs Erro Local (Passo Único)
    ax1.plot(layers, [h["e_l_r1024"] * 100 for h in history], color="#e74c3c", linewidth=2.2, label="e_l (Cumulativo) r=1024")
    ax1.plot(layers, [h["e_l_r1536"] * 100 for h in history], color="#f39c12", linewidth=2.2, label="e_l (Cumulativo) r=1536")
    ax1.plot(layers, [h["e_l_r2048"] * 100 for h in history], color="#27ae60", linewidth=2.5, label="e_l (Cumulativo) r=2048")
    ax1.plot(layers, [h["delta_l_r2048"] * 100 for h in history], color="#2980b9", linestyle="--", linewidth=2.0, label="delta_l (Local Passo Único) r=2048")

    # Marcar camadas de Atenção GQA (l % 4 == 3)
    attn_layers = [h["layer"] for h in history if h["is_attn"]]
    for al in attn_layers:
        ax1.axvline(x=al, color="#bdc3c7", linestyle=":", alpha=0.6)

    ax1.set_ylabel("Erro Relativo (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Diagnóstico de Divergência: Erro Cumulativo (e_l) vs Erro Local Passo Único (delta_l) nas 64 Camadas", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gráfico 2: Similaridade de Cosseno de Hidden States e Logits (Logit Lens)
    ax2.plot(layers, [h["cos_h_r2048"] for h in history], color="#27ae60", linewidth=2.5, label="Cosine Sim h_l (r=2048)")
    ax2.plot(layers, [h["cos_z_r2048"] for h in history], color="#8e44ad", linewidth=2.2, linestyle="-.", label="Cosine Sim Logits z_l (Logit Lens r=2048)")
    ax2.plot(layers, [h["top1_z_r2048"] / 100.0 for h in history], color="#d35400", linewidth=2.0, linestyle=":", label="Top-1 Agreement Logits (r=2048)")

    for al in attn_layers:
        ax2.axvline(x=al, color="#bdc3c7", linestyle=":", alpha=0.6)

    ax2.set_xlabel("Índice da Camada (0 .. 63)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Alinhamento / Concordância", fontsize=12, fontweight="bold")
    ax2.set_title("Alinhamento Semântico ao Longo da Profundidade: Hidden States vs Projeção de Logits (Logit Lens)", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "layer_divergence_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")

    # Copiar também para o diretório raiz de artefatos
    artifact_root = os.path.dirname(out_dir)
    root_plot = os.path.join(artifact_root, "layer_divergence_curve.png")
    import shutil
    shutil.copy2(plot_path, root_plot)
    print(f"Gráfico copiado para: {root_plot}")

if __name__ == "__main__":
    run_layer_divergence_tracking()
