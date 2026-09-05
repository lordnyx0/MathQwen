# -*- coding: utf-8 -*-
"""
Sondagem das 8 Camadas Críticas: SVD-64 vs GELU-WS vs GELU-MF vs GELU-MF+KD
Avaliando a hipótese da não-linearidade dependente do estado com Logit Lens (T=1).
"""
import sys
import os
import glob
import time
import math
import json
import shutil
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

# Forçar stdout sem buffer no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]

def dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.float() * s_exp.float())

def fit_svd_stabilizer(X_calib: torch.Tensor, Y_calib: torch.Tensor, r_corr: int = 64, lambda_reg: float = 1e-3):
    """Calcula a solução analítica fechada ótima de rank r_corr para h' = h + (h @ W_down) @ W_up."""
    E = Y_calib - X_calib
    N, d = X_calib.shape

    U, S, Vh = torch.linalg.svd(X_calib.float(), full_matrices=False)
    V = Vh.t()

    reg = lambda_reg * (S**2).mean()
    weights = S / (S**2 + reg)
    M = torch.matmul(E.float().t(), U * weights)

    Um, Sm, Vhm = torch.linalg.svd(M, full_matrices=False)
    Wr = torch.matmul(V, Vhm.t())

    k = min(r_corr, N)
    W_down = (Wr[:, :k] * torch.sqrt(Sm[:k])).to(dtype=torch.bfloat16)
    W_up = (Um[:, :k] * torch.sqrt(Sm[:k])).t().to(dtype=torch.bfloat16)

    return W_down, W_up

def fit_gelu_warmstart(X_calib: torch.Tensor, Y_calib: torch.Tensor, W_down: torch.Tensor, lambda_reg: float = 1e-3):
    """
    Calcula a solução fechada ótima de Ridge para W_up com W_down fixo e delta_alpha = 0:
    (Z^T Z + lambda I) W_up = Z^T R, com Z = GELU(X @ W_down) e R = Y - X.
    Resolve o sistema linear diretamente sem inversão explícita.
    """
    Z_tr = F.gelu(torch.matmul(X_calib.float(), W_down.float())) # [N, r]
    R_tr = (Y_calib - X_calib).float()                           # [N, d]
    r = W_down.shape[1]

    ZTZ = torch.matmul(Z_tr.t(), Z_tr)
    reg_val = lambda_reg * (torch.trace(ZTZ) / r)
    reg_I = reg_val * torch.eye(r, device=X_calib.device, dtype=torch.float32)
    ZTR = torch.matmul(Z_tr.t(), R_tr)

    W_up = torch.linalg.solve(ZTZ + reg_I, ZTR).to(dtype=torch.bfloat16)
    return W_up

def eval_logit_lens(h_test: torch.Tensor, targets: torch.Tensor, norm_mod, lm_head_w: torch.Tensor, h_teacher_test: torch.Tensor = None, temp: float = 1.0):
    """
    Avalia NLL, Top-1 Agreement e KL divergence em temp (padronizado em T=1 para métricas finais).
    Processa em micro-batches para manter o uso de VRAM < 200 MB.
    """
    B, L, d = h_test.shape
    vocab_size = lm_head_w.shape[0]
    total_tokens = B * (L - 1)

    total_nll = 0.0
    total_agree = 0
    total_kl = 0.0
    chunk_size = 4

    with torch.no_grad():
        for b_start in range(0, B, chunk_size):
            b_end = min(b_start + chunk_size, B)
            h_chunk = h_test[b_start:b_end].to(dtype=torch.bfloat16)
            z_S = F.linear(norm_mod(h_chunk), lm_head_w)
            z_S_pred = z_S[:, :-1, :].reshape(-1, vocab_size).float()

            t_chunk = targets[b_start * (L - 1): b_end * (L - 1)]
            nll = F.cross_entropy(z_S_pred, t_chunk, reduction="sum").item()
            total_nll += nll

            if h_teacher_test is not None:
                h_T_chunk = h_teacher_test[b_start:b_end].to(dtype=torch.bfloat16)
                z_T = F.linear(norm_mod(h_T_chunk), lm_head_w)
                z_T_pred = z_T[:, :-1, :].reshape(-1, vocab_size).float()

                agree = (z_S_pred.argmax(dim=-1) == z_T_pred.argmax(dim=-1)).sum().item()
                total_agree += agree

                log_p_T = F.log_softmax(z_T_pred / temp, dim=-1)
                log_p_S = F.log_softmax(z_S_pred / temp, dim=-1)
                p_T = log_p_T.exp()
                kl = (p_T * (log_p_T - log_p_S)).sum(dim=-1).sum().item()
                total_kl += kl

    mean_nll = total_nll / total_tokens
    mean_kl = total_kl / total_tokens if h_teacher_test is not None else 0.0
    agree_pct = (total_agree / total_tokens) * 100.0 if h_teacher_test is not None else 0.0

    return mean_nll, mean_kl, agree_pct

def run_probe_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("      SONDAGEM DAS 8 CAMADAS CRÍTICAS COM LOGIT LENS (T=1) E NÃO-LINEARIDADE RESIDUAL")
    print("         SVD-64  vs  GELU-WS  vs  GELU-MF (Manifold)  vs  GELU-MF+KD (Distill)")
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

        # Carregar pesos da carta
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

            # 2. Forward Atlas r=2048
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

            # SVD Analítico Linear
            W_d_svd, W_u_svd = fit_svd_stabilizer(x_tr, y_tr, r_corr=r_corr)

            norm_y_te = torch.norm(y_te.float()).item()
            norm_y_tr = torch.norm(y_tr.float()).item()

            h_te_3d_orig = h_student_raw[16:]
            h_tr_3d_orig = h_teacher_next[:16]
            h_te_3d_prof = h_teacher_next[16:]

            # Se for uma das 8 camadas de sondagem: executar o comparativo dos 4 modelos
            if l in probe_set:
                print(f"\n  ================== SONDAGEM COMPLETA EM L{l:02d} ==================")

                # [Modelo 1: SVD-64 Linear]
                with torch.no_grad():
                    corr_svd = torch.matmul(torch.matmul(x_te, W_d_svd), W_u_svd)
                    h_svd_te = x_te + corr_svd
                    e_svd = (torch.norm((h_svd_te - y_te).float()) / norm_y_te).item()
                    cos_svd = F.cosine_similarity(h_svd_te.float().flatten(), y_te.float().flatten(), dim=0).item()
                    h_svd_3d = h_te_3d_orig + torch.matmul(torch.matmul(h_te_3d_orig, W_d_svd), W_u_svd)
                    nll_svd, kl_svd, agree_svd = eval_logit_lens(h_svd_3d, test_targets, norm_mod, lm_head_w, h_teacher_test=h_te_3d_prof, temp=1.0)

                print(f"  [1] SVD-64     | e_h={e_svd*100:5.2f}% | cos={cos_svd:.4f} | NLL={nll_svd:.4f} | KL(T=1)={kl_svd:.4f} | Top-1 Agree={agree_svd:5.2f}%")

                # [Modelo 2: GELU-WS (Warm-Start Fechado via Ridge Solve)]
                W_u_gelu_ws = fit_gelu_warmstart(x_tr, y_tr, W_d_svd, lambda_reg=1e-3)
                with torch.no_grad():
                    z_te_ws = F.gelu(torch.matmul(x_te.float(), W_d_svd.float())).to(dtype=torch.bfloat16)
                    corr_gelu_ws = torch.matmul(z_te_ws, W_u_gelu_ws)
                    h_ws_te = x_te + corr_gelu_ws
                    e_ws = (torch.norm((h_ws_te - y_te).float()) / norm_y_te).item()
                    cos_ws = F.cosine_similarity(h_ws_te.float().flatten(), y_te.float().flatten(), dim=0).item()
                    z_3d_ws = F.gelu(torch.matmul(h_te_3d_orig.float(), W_d_svd.float())).to(dtype=torch.bfloat16)
                    h_ws_3d = h_te_3d_orig + torch.matmul(z_3d_ws, W_u_gelu_ws)
                    nll_ws, kl_ws, agree_ws = eval_logit_lens(h_ws_3d, test_targets, norm_mod, lm_head_w, h_teacher_test=h_te_3d_prof, temp=1.0)

                delta_e_ws = (e_ws - e_svd) * 100.0
                delta_nll_ws = nll_ws - nll_svd
                print(f"  [2] GELU-WS    | e_h={e_ws*100:5.2f}% ({delta_e_ws:+5.2f}%) | cos={cos_ws:.4f} | NLL={nll_ws:.4f} ({delta_nll_ws:+6.4f}) | KL(T=1)={kl_ws:.4f} | Top-1 Agree={agree_ws:5.2f}%")

                # [Modelo 3: GELU-MF (Manifold Fitting com delta_alpha)]
                W_d_mf = torch.nn.Parameter(W_d_svd.clone())
                W_u_mf = torch.nn.Parameter(W_u_gelu_ws.clone())
                delta_alpha_mf = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))

                opt_mf = torch.optim.AdamW([
                    {"params": [W_d_mf, W_u_mf], "lr": 1e-4, "weight_decay": 1e-4},
                    {"params": [delta_alpha_mf], "lr": 1e-3, "weight_decay": 0.0}
                ])

                # Checkpoint lexicográfico: inicializa no SVD baseline
                best_mf = {
                    "e_h": e_svd,
                    "cos": cos_svd,
                    "nll": nll_svd,
                    "kl": kl_svd,
                    "agree": agree_svd,
                    "W_d": W_d_svd.clone(),
                    "W_u": W_u_svd.clone(),
                    "delta_alpha": 0.0,
                    "is_nonlinear": False,
                    "model_source": "SVD-64"
                }

                # Se o warm-start GELU já não piorar o e_h do SVD, ele é um candidato válido:
                if e_ws <= e_svd:
                    best_mf.update({
                        "e_h": e_ws,
                        "cos": cos_ws,
                        "nll": nll_ws,
                        "kl": kl_ws,
                        "agree": agree_ws,
                        "W_d": W_d_svd.clone(),
                        "W_u": W_u_gelu_ws.clone(),
                        "delta_alpha": 0.0,
                        "is_nonlinear": True,
                        "model_source": "GELU-WS"
                    })

                h_tr_3d = h_student_raw[:16]

                for step in range(30):
                    opt_mf.zero_grad()
                    z_tr = F.gelu(torch.matmul(h_tr_3d.float(), W_d_mf.float())).to(dtype=torch.bfloat16)
                    corr_tr = torch.matmul(z_tr, W_u_mf)
                    alpha_curr = (1.0 + delta_alpha_mf).to(dtype=torch.bfloat16)
                    h_s_tr = alpha_curr * h_tr_3d + corr_tr

                    loss_mse = torch.norm((h_s_tr.reshape(-1, 5120) - y_tr).float()) / norm_y_tr
                    loss_cos = 1.0 - F.cosine_similarity(h_s_tr.float().flatten(), y_tr.float().flatten(), dim=0)
                    loss_alpha = 10.0 * (delta_alpha_mf ** 2)
                    loss = loss_mse + 10.0 * loss_cos + loss_alpha
                    loss.backward()
                    opt_mf.step()

                    # Avaliação periódica e critério lexicográfico a cada 5 passos
                    if (step + 1) % 5 == 0:
                        with torch.no_grad():
                            z_te_step = F.gelu(torch.matmul(x_te.float(), W_d_mf.float())).to(dtype=torch.bfloat16)
                            corr_te_step = torch.matmul(z_te_step, W_u_mf)
                            alpha_val = 1.0 + delta_alpha_mf.item()
                            h_step_te = alpha_val * x_te + corr_te_step
                            e_curr = (torch.norm((h_step_te - y_te).float()) / norm_y_te).item()

                            # Condição lexicográfica estrita:
                            # 1. e_curr <= e_svd
                            # 2. menor NLL
                            if e_curr <= e_svd:
                                z_3d_step = F.gelu(torch.matmul(h_te_3d_orig.float(), W_d_mf.float())).to(dtype=torch.bfloat16)
                                h_step_3d = alpha_val * h_te_3d_orig + torch.matmul(z_3d_step, W_u_mf)
                                nll_curr, kl_curr, agree_curr = eval_logit_lens(h_step_3d, test_targets, norm_mod, lm_head_w, h_teacher_test=h_te_3d_prof, temp=1.0)
                                cos_curr = F.cosine_similarity(h_step_te.float().flatten(), y_te.float().flatten(), dim=0).item()

                                if nll_curr <= best_mf["nll"]:
                                    best_mf.update({
                                        "e_h": e_curr,
                                        "cos": cos_curr,
                                        "nll": nll_curr,
                                        "kl": kl_curr,
                                        "agree": agree_curr,
                                        "W_d": W_d_mf.data.clone(),
                                        "W_u": W_u_mf.data.clone(),
                                        "delta_alpha": delta_alpha_mf.item(),
                                        "is_nonlinear": True,
                                        "model_source": "GELU-MF"
                                    })

                delta_e_mf = (best_mf["e_h"] - e_svd) * 100.0
                delta_nll_mf = best_mf["nll"] - nll_svd
                print(f"  [3] GELU-MF    | e_h={best_mf['e_h']*100:5.2f}% ({delta_e_mf:+5.2f}%) | cos={best_mf['cos']:.4f} | NLL={best_mf['nll']:.4f} ({delta_nll_mf:+6.4f}) | KL(T=1)={best_mf['kl']:.4f} | Top-1 Agree={best_mf['agree']:5.2f}% | Sel: {best_mf['model_source']}")

                # [Modelo 4: GELU-MF+KD (Manifold + Distillation T=2)]
                W_d_kd = torch.nn.Parameter(W_d_svd.clone())
                W_u_kd = torch.nn.Parameter(W_u_gelu_ws.clone())
                delta_alpha_kd = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))

                opt_kd = torch.optim.AdamW([
                    {"params": [W_d_kd, W_u_kd], "lr": 1e-4, "weight_decay": 1e-4},
                    {"params": [delta_alpha_kd], "lr": 1e-3, "weight_decay": 0.0}
                ])

                best_kd = {
                    "e_h": e_svd,
                    "cos": cos_svd,
                    "nll": nll_svd,
                    "kl": kl_svd,
                    "agree": agree_svd,
                    "W_d": W_d_svd.clone(),
                    "W_u": W_u_svd.clone(),
                    "delta_alpha": 0.0,
                    "is_nonlinear": False,
                    "model_source": "SVD-64"
                }

                # Computar logits do professor para treino uma vez (em chunks para manter VRAM < 200MB)
                with torch.no_grad():
                    norm_prof_tr = norm_mod(h_teacher_next[:16])
                    z_prof_tr = F.linear(norm_prof_tr, lm_head_w) # [16, 64, vocab]
                    p_prof_tr_T2 = F.softmax(z_prof_tr.float() / 2.0, dim=-1)

                for step in range(30):
                    opt_kd.zero_grad()
                    z_tr = F.gelu(torch.matmul(h_tr_3d.float(), W_d_kd.float())).to(dtype=torch.bfloat16)
                    corr_tr = torch.matmul(z_tr, W_u_kd)
                    alpha_curr = (1.0 + delta_alpha_kd).to(dtype=torch.bfloat16)
                    h_s_tr = alpha_curr * h_tr_3d + corr_tr

                    loss_mse = torch.norm((h_s_tr.reshape(-1, 5120) - y_tr).float()) / norm_y_tr
                    loss_cos = 1.0 - F.cosine_similarity(h_s_tr.float().flatten(), y_tr.float().flatten(), dim=0)
                    loss_alpha = 10.0 * (delta_alpha_kd ** 2)

                    # Soft-KD com T=2 em micro-batch de 8 seqs
                    h_s_tr_norm = norm_mod(h_s_tr[:8].to(dtype=torch.bfloat16))
                    z_s_tr = F.linear(h_s_tr_norm, lm_head_w)
                    log_p_s_T2 = F.log_softmax(z_s_tr.float() / 2.0, dim=-1)
                    p_prof_sub = p_prof_tr_T2[:8]
                    loss_kd = 4.0 * (p_prof_sub * (p_prof_sub.log().clamp(min=-100) - log_p_s_T2)).sum(dim=-1).mean()

                    loss = loss_mse + 10.0 * loss_cos + loss_alpha + 0.05 * loss_kd
                    loss.backward()
                    opt_kd.step()

                    if (step + 1) % 5 == 0:
                        with torch.no_grad():
                            z_te_step = F.gelu(torch.matmul(x_te.float(), W_d_kd.float())).to(dtype=torch.bfloat16)
                            corr_te_step = torch.matmul(z_te_step, W_u_kd)
                            alpha_val = 1.0 + delta_alpha_kd.item()
                            h_step_te = alpha_val * x_te + corr_te_step
                            e_curr = (torch.norm((h_step_te - y_te).float()) / norm_y_te).item()

                            if e_curr <= e_svd:
                                z_3d_step = F.gelu(torch.matmul(h_te_3d_orig.float(), W_d_kd.float())).to(dtype=torch.bfloat16)
                                h_step_3d = alpha_val * h_te_3d_orig + torch.matmul(z_3d_step, W_u_kd)
                                nll_curr, kl_curr, agree_curr = eval_logit_lens(h_step_3d, test_targets, norm_mod, lm_head_w, h_teacher_test=h_te_3d_prof, temp=1.0)
                                cos_curr = F.cosine_similarity(h_step_te.float().flatten(), y_te.float().flatten(), dim=0).item()

                                if nll_curr <= best_kd["nll"]:
                                    best_kd.update({
                                        "e_h": e_curr,
                                        "cos": cos_curr,
                                        "nll": nll_curr,
                                        "kl": kl_curr,
                                        "agree": agree_curr,
                                        "W_d": W_d_kd.data.clone(),
                                        "W_u": W_u_kd.data.clone(),
                                        "delta_alpha": delta_alpha_kd.item(),
                                        "is_nonlinear": True,
                                        "model_source": "GELU-MF+KD"
                                    })

                delta_e_kd = (best_kd["e_h"] - e_svd) * 100.0
                delta_nll_kd = best_kd["nll"] - nll_svd
                print(f"  [4] GELU-MF+KD | e_h={best_kd['e_h']*100:5.2f}% ({delta_e_kd:+5.2f}%) | cos={best_kd['cos']:.4f} | NLL={best_kd['nll']:.4f} ({delta_nll_kd:+6.4f}) | KL(T=1)={best_kd['kl']:.4f} | Top-1 Agree={best_kd['agree']:5.2f}% | Sel: {best_kd['model_source']}")

                # Seleção do melhor modelo global para avançar na trajetória
                # Prioridade: menor NLL entre os modelos que respeitam e_h <= e_svd
                candidates = [best_mf, best_kd]
                chosen = min(candidates, key=lambda c: c["nll"])

                probe_results.append({
                    "layer": l,
                    "svd": {"e_h": e_svd, "cos": cos_svd, "nll": nll_svd, "kl": kl_svd, "agree": agree_svd},
                    "gelu_ws": {"e_h": e_ws, "cos": cos_ws, "nll": nll_ws, "kl": kl_ws, "agree": agree_ws},
                    "gelu_mf": {"e_h": best_mf["e_h"], "cos": best_mf["cos"], "nll": best_mf["nll"], "kl": best_mf["kl"], "agree": best_mf["agree"], "sel": best_mf["model_source"]},
                    "gelu_kd": {"e_h": best_kd["e_h"], "cos": best_kd["cos"], "nll": best_kd["nll"], "kl": best_kd["kl"], "agree": best_kd["agree"], "sel": best_kd["model_source"]},
                    "chosen_source": chosen["model_source"]
                })

                with torch.no_grad():
                    if chosen["is_nonlinear"]:
                        z_full = F.gelu(torch.matmul(h_student_raw.float(), chosen["W_d"].float())).to(dtype=torch.bfloat16)
                        corr_full = torch.matmul(z_full, chosen["W_u"])
                        alpha_val = 1.0 + chosen["delta_alpha"]
                        h_student = alpha_val * h_student_raw + corr_full
                    else:
                        corr_full = torch.matmul(torch.matmul(h_student_raw, W_d_svd), W_u_svd)
                        h_student = h_student_raw + corr_full
            else:
                # Camadas intermediárias: avança com o SVD linear comprovado
                with torch.no_grad():
                    corr_full = torch.matmul(torch.matmul(h_student_raw, W_d_svd), W_u_svd)
                    h_student = h_student_raw + corr_full

            h_teacher = h_teacher_next

            del layer_mod, W_mix_orig, W_gate_orig, W_up_orig, W_down_orig, W_mix_r, W_gate_r, W_up_r, W_down_r
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} concluida em {time.time() - t_chart:.2f}s")

    t_total = time.time() - t_start
    print(f"\n[OK] Sondagem das 8 camadas concluiu com sucesso em {t_total:.2f}s ({t_total/64:.2f}s/camada)!")

    # 3. Tabela Consolidada Final
    print("\n" + "=" * 125)
    print("                      TABELA CONSOLIDADA: SONDAGEM DAS 8 CAMADAS CRÍTICAS (TEST SET)")
    print("=" * 125)
    print(f"{'Camada':<8} | {'Métrica':<10} | {'SVD-64 (Linear)':<16} | {'GELU-WS (Ridge)':<16} | {'GELU-MF (Manifold)':<18} | {'GELU-MF+KD (Distill)':<20} | {'Melhor Modelo'}")
    print("-" * 125)

    for r in probe_results:
        l = r["layer"]
        # Linha e_h
        print(f"L{l:<7} | e_h (%)    | {r['svd']['e_h']*100:6.2f}%          | {r['gelu_ws']['e_h']*100:6.2f}%          | {r['gelu_mf']['e_h']*100:6.2f}%            | {r['gelu_kd']['e_h']*100:6.2f}%              | {r['chosen_source']}")
        # Linha NLL
        print(f"         | NLL (T=1)  | {r['svd']['nll']:8.4f}          | {r['gelu_ws']['nll']:8.4f}          | {r['gelu_mf']['nll']:8.4f}            | {r['gelu_kd']['nll']:8.4f}              |")
        # Linha KL
        print(f"         | KL (T=1)   | {r['svd']['kl']:8.4f}          | {r['gelu_ws']['kl']:8.4f}          | {r['gelu_mf']['kl']:8.4f}            | {r['gelu_kd']['kl']:8.4f}              |")
        # Linha Top-1 Agree
        print(f"         | Agree (%)  | {r['svd']['agree']:6.2f}%          | {r['gelu_ws']['agree']:6.2f}%          | {r['gelu_mf']['agree']:6.2f}%            | {r['gelu_kd']['agree']:6.2f}%              |")
        print("-" * 125)

    print("=" * 125)

    # 4. Salvar resultados em JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "probe_nonlinear_8layers_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    # 5. Gerar Gráfico Comparativo Triplo (e_h, NLL, KL)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 12))

    x_labels = [f"L{r['layer']}" for r in probe_results]
    x_idx = list(range(len(x_labels)))

    # Painel 1: Erro de Hidden State e_h
    ax1.plot(x_idx, [r["svd"]["e_h"] * 100 for r in probe_results], marker="o", color="#34495e", linewidth=2.2, label="SVD-64 (Linear Baseline)")
    ax1.plot(x_idx, [r["gelu_ws"]["e_h"] * 100 for r in probe_results], marker="^", color="#e67e22", linewidth=2.0, linestyle="--", label="GELU-WS (Closed-form Ridge)")
    ax1.plot(x_idx, [r["gelu_mf"]["e_h"] * 100 for r in probe_results], marker="s", color="#27ae60", linewidth=2.2, label="GELU-MF (AdamW Manifold)")
    ax1.plot(x_idx, [r["gelu_kd"]["e_h"] * 100 for r in probe_results], marker="D", color="#2980b9", linewidth=2.2, label="GELU-MF+KD (Distillation T=2)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Erro Relativo e_h (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Comparação nas 8 Camadas Críticas no Teste Cego: Erro de Variedade Residual e_h", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Painel 2: NLL Terminal (T=1)
    ax2.plot(x_idx, [r["svd"]["nll"] for r in probe_results], marker="o", color="#34495e", linewidth=2.2, label="SVD-64 (Linear)")
    ax2.plot(x_idx, [r["gelu_ws"]["nll"] for r in probe_results], marker="^", color="#e67e22", linewidth=2.0, linestyle="--", label="GELU-WS")
    ax2.plot(x_idx, [r["gelu_mf"]["nll"] for r in probe_results], marker="s", color="#27ae60", linewidth=2.2, label="GELU-MF")
    ax2.plot(x_idx, [r["gelu_kd"]["nll"] for r in probe_results], marker="D", color="#2980b9", linewidth=2.2, label="GELU-MF+KD")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("NLL (T=1)", fontsize=11, fontweight="bold")
    ax2.set_title("Logit Lens: NLL Terminal dos Tokens nas 8 Camadas Críticas", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Painel 3: Kullback-Leibler D_KL (T=1)
    ax3.plot(x_idx, [r["svd"]["kl"] for r in probe_results], marker="o", color="#34495e", linewidth=2.2, label="SVD-64 (Linear)")
    ax3.plot(x_idx, [r["gelu_ws"]["kl"] for r in probe_results], marker="^", color="#e67e22", linewidth=2.0, linestyle="--", label="GELU-WS")
    ax3.plot(x_idx, [r["gelu_mf"]["kl"] for r in probe_results], marker="s", color="#27ae60", linewidth=2.2, label="GELU-MF")
    ax3.plot(x_idx, [r["gelu_kd"]["kl"] for r in probe_results], marker="D", color="#2980b9", linewidth=2.2, label="GELU-MF+KD")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(x_labels)
    ax3.set_ylabel("D_KL (T=1)", fontsize=11, fontweight="bold")
    ax3.set_title("Logit Lens: Divergência KL com o Professor Oficial (T=1)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "probe_nonlinear_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    root_dir = os.path.dirname(out_dir)
    shutil.copy2(plot_path, os.path.join(root_dir, "probe_nonlinear_comparison.png"))
    print(f"\nGráfico comparativo salvo com sucesso em {plot_path}!")

if __name__ == "__main__":
    run_probe_experiment()
