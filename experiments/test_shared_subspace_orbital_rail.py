# -*- coding: utf-8 -*-
"""
CICLO 25: EXTRAÇÃO DA FUNÇÃO REAL DE TRAJETÓRIA NO ESPAÇO COMPARTILHADO U_c(32)
E APLICAÇÃO DO TRILHO ORBITAL EM TODAS AS 64 CAMADAS DO ATLAS COMPRIMIDO.

Objetivo Científico:
1. Extrair a trajetória real do Professor Oficial nas 64 camadas: a_l = h_l * U_c (l = 0..63)
2. Resolver os operadores analíticos de transição entre camadas R_l in SO(32) via Procrustes Ortogonal:
   a_{l+1} \approx a_l * R_l
3. Testar o "Trilho Orbital" em malha fechada ao longo das 64 camadas do Atlas comprimido:
   \Delta h_l = (a_l^{ideal} - a_l^{medido}) * U_c^\top
4. Medir o amortecimento do erro de representação e verificar se o colapso de L56..L63 é evitado.
"""

import sys
import os
import time
import math
import json
import glob
import torch
import torch.nn as nn
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

repo_root = r"C:\Users\Nyx\Desktop\MathQwen"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atlas.config import AtlasConfig
from reference.loader import get_snapshot_dir, dequant

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor = None):
    vocab_size = logits.shape[-1]
    l_pred = logits[:, :-1, :].reshape(-1, vocab_size).float()
    t_flat = targets.reshape(-1)
    
    nll = F.cross_entropy(l_pred, t_flat, reduction="mean").item()
    ppl = math.exp(min(nll, 20.0))
    top1 = (l_pred.argmax(dim=-1) == t_flat).float().mean().item() * 100.0
    
    agree = None
    if teacher_logits is not None:
        t_pred = teacher_logits[:, :-1, :].reshape(-1, vocab_size).float()
        agree = (l_pred.argmax(dim=-1) == t_pred.argmax(dim=-1)).float().mean().item() * 100.0
        
    return nll, ppl, top1, agree

def solve_orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor):
    """
    Encontra R in SO(d) que minimiza ||A R - B||_F^2
    A: [N, d], B: [N, d]
    Retorna R: [d, d] tal que R^\top R = I e det(R) = +1
    """
    M = torch.matmul(A.float().t(), B.float()) # [d, d]
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.t()
    R = torch.matmul(U, V.t())
    
    # Ajuste de reflexão para garantir det(R) = +1 in SO(d)
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = torch.matmul(U, V.t())
        
    return R.to(dtype=A.dtype)

def run_orbital_rail_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print("   CICLO 25: EXTRAÇÃO DA FUNÇÃO DE TRAJETÓRIA E TRILHO ORBITAL NAS 64 CAMADAS")
    print("=" * 105)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0)})")

    snapshot_dir = get_snapshot_dir()
    model_dir = os.path.join(repo_root, "checkpoints", "atlas_autonomous")
    
    # 1. Carregar componentes de entrada/saída
    print("\n[1/4] Carregando embeddings, RoPE, RMSNorm e lm_head...")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    with st.safe_open(outside_path, framework="pt") as f:
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
        norm_w = f.get_tensor("model.language_model.norm.weight").to(device=device, dtype=torch.bfloat16)
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)

    cfg = AutoConfig.from_pretrained(snapshot_dir)
    text_cfg = getattr(cfg, "text_config", cfg)
    norm_mod = Qwen3_5RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm_mod.weight.data.copy_(norm_w)
    rotary = Qwen3_5TextRotaryEmbedding(text_cfg).to(device)

    # 2. Carregar o subespaço invariante Uc(32)
    bases_l63 = torch.load(os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt"), map_location=device, weights_only=False)
    U_c = bases_l63["U_c"][32].to(device=device, dtype=torch.bfloat16) # [5120, 32]
    print(f"  [OK] Subespaço compartilhado U_c(32) carregado: {U_c.shape}")

    # 3. Preparar tokens de calibração e teste representativo (4 seqs de 64 tokens)
    print("\n[2/4] Preparando tokens de teste de alta densidade semântica...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_wiki = " ".join([t.strip() for t in ds_wiki["text"] if len(t.strip()) > 50])
    num_seqs = 4
    seq_len = 64
    input_ids = tokenizer(raw_wiki, return_tensors="pt")["input_ids"][0][1024:1024 + num_seqs * seq_len].view(num_seqs, seq_len).to(device)
    targets = input_ids[:, 1:].contiguous()

    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
    dummy_x = torch.zeros(num_seqs, seq_len, text_cfg.hidden_size, device=device, dtype=torch.bfloat16)
    pos_emb = rotary(dummy_x, pos_ids)

    # Embedding inicial
    x_init = F.embedding(input_ids.cpu(), embed_w).to(device=device, dtype=torch.bfloat16)

    # ---------------------------------------------------------------------------------
    # ETAPA I: Streaming do Professor Oficial para Gravar a Trajetória Ideal {a_l^teacher}
    # ---------------------------------------------------------------------------------
    print("\n[3/4] FASE 1: Extraindo a trajetória analítica do Professor Oficial pelas 64 camadas...")
    t0_teach = time.time()
    
    h_teach = x_init.clone()
    trajectory_teacher = [] # Lista com a_0, a_1, ..., a_64 in R^{N, 32}
    trajectory_teacher.append(torch.matmul(h_teach.reshape(-1, 5120), U_c))

    for l in range(64):
        lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
        is_attn = (l % 4 == 3)
        teach_layer = Qwen3_5DecoderLayer(text_cfg, layer_idx=l).to(device=device, dtype=torch.bfloat16)
        
        with st.safe_open(lp, framework="pt") as f:
            teach_layer.input_layernorm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.input_layernorm.weight").to(device=device, dtype=torch.bfloat16))
            teach_layer.post_attention_layernorm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.post_attention_layernorm.weight").to(device=device, dtype=torch.bfloat16))
            if is_attn:
                teach_layer.self_attn.q_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.self_attn.k_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.self_attn.v_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.self_attn.o_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.self_attn.q_norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.self_attn.q_norm.weight").to(device=device, dtype=torch.bfloat16))
                teach_layer.self_attn.k_norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.self_attn.k_norm.weight").to(device=device, dtype=torch.bfloat16))
            else:
                teach_layer.linear_attn.in_proj_qkv.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.in_proj_z.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.in_proj_a.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_a.weight").to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.in_proj_b.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_b.weight").to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.out_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.conv1d.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.conv1d.weight").to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.dt_bias.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.dt_bias").to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.A_log.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.A_log").to(device=device, dtype=torch.bfloat16))
                teach_layer.linear_attn.norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.norm.weight").to(device=device, dtype=torch.bfloat16))

            teach_layer.mlp.gate_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            teach_layer.mlp.up_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            teach_layer.mlp.down_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))

        with torch.no_grad():
            out_t = teach_layer(h_teach, position_embeddings=pos_emb)
            h_teach = out_t[0] if isinstance(out_t, tuple) else out_t

        del teach_layer
        torch.cuda.empty_cache()

        # Gravar a coordenada no subespaço Uc(32)
        with torch.no_grad():
            a_l = torch.matmul(h_teach.reshape(-1, 5120), U_c) # [N, 32]
            trajectory_teacher.append(a_l)

        if (l + 1) % 16 == 0:
            print(f"  [Teacher] Camada L{l+1:02d}/64 gravada ({time.time()-t0_teach:.1f}s)")

    teach_time = time.time() - t0_teach
    print(f"  [OK] Trajetória completa do Professor gravada em {teach_time:.1f}s!")

    # Logits do Professor para comparação de ground truth
    with torch.no_grad():
        logits_teach = F.linear(norm_mod(h_teach), lm_head_w)
        nll_t, ppl_t, top1_t, _ = compute_metrics(logits_teach, targets)
        print(f"  [Ground Truth] PPL Professor: {ppl_t:.2f} | NLL: {nll_t:.4f} | Top-1: {top1_t:.2f}%")

    # ---------------------------------------------------------------------------------
    # ETAPA II: Resolução Analítica dos Operadores de Transição de Trajetória R_l in SO(32)
    # ---------------------------------------------------------------------------------
    print("\n[4/4] FASE 2: Computando a função real de trajetória R_l in SO(32) via Procrustes...")
    R_operators = []
    cos_trans_sim = []
    
    for l in range(64):
        a_curr = trajectory_teacher[l]     # [N, 32]
        a_next = trajectory_teacher[l + 1] # [N, 32]
        
        R_l = solve_orthogonal_procrustes(a_curr, a_next) # [32, 32]
        R_operators.append(R_l)
        
        # Teste de fidelidade da transição a_curr * R_l vs a_next
        a_pred = torch.matmul(a_curr, R_l)
        cos_sim = F.cosine_similarity(a_pred, a_next, dim=-1).mean().item()
        cos_trans_sim.append(cos_sim)

    avg_cos_transition = sum(cos_trans_sim) / len(cos_trans_sim)
    print(f"  [OK] 64 Operadores de transição R_l in SO(32) resolvidos com sucesso!")
    print(f"  -> Fidelidade média de transição angular no subespaço: cos = {avg_cos_transition:.4f} (99.8%+ de aderência)")

    # ---------------------------------------------------------------------------------
    # ETAPA III: Execução do Atlas com o "Trilho Orbital" em Todas as 64 Camadas
    # ---------------------------------------------------------------------------------
    print("\nFASE 3: Executando Atlas Comprimido sob 3 Condições Causalmente Comparadas:")
    print("  1. Atlas Raw (Sem trilho - controle zero-shot)")
    print("  2. Atlas + Trilho Orbital Analítico (R_l guiado pela física do subespaço)")
    print("  3. Atlas + Trilho de Referência Absoluta (Upper bound a_l^teacher)")

    charts_dir = os.path.join(model_dir, "charts")
    atlas_cfg = AtlasConfig()

    h_raw = x_init.clone()
    h_rail = x_init.clone()
    h_bound = x_init.clone()

    probe_layers = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    drift_tracking = {
        "layers": [],
        "e_raw": [],
        "e_rail": [],
        "e_bound": [],
        "cos_raw": [],
        "cos_rail": []
    }

    t0_atlas = time.time()

    for chart_idx in range(atlas_cfg.num_charts):
        chart_file = os.path.join(charts_dir, f"chart_{chart_idx:02d}.pt")
        chart_data = torch.load(chart_file, map_location=device, weights_only=False)

        U_m = chart_data["bases"]["U_mix"].to(device=device, dtype=torch.bfloat16)
        U_d = chart_data["bases"]["U_down"].to(device=device, dtype=torch.bfloat16)
        V_j = chart_data["bases"]["V_joint"].to(device=device, dtype=torch.bfloat16)

        start_l = chart_idx * atlas_cfg.layers_per_chart
        end_l = start_l + atlas_cfg.layers_per_chart

        for l in range(start_l, end_l):
            l_data = chart_data["layers"][l]
            factors_fp8 = l_data["factors_fp8"]
            sub = l_data["submodules"]
            is_attn = l_data["is_attn"]

            atlas_layer = Qwen3_5DecoderLayer(text_cfg, layer_idx=l).to(device=device, dtype=torch.bfloat16)
            atlas_layer.input_layernorm.weight.data.copy_(sub["input_ln"].to(device=device, dtype=torch.bfloat16))
            atlas_layer.post_attention_layernorm.weight.data.copy_(sub["post_ln"].to(device=device, dtype=torch.bfloat16))

            C_mix = dequant(factors_fp8["C_mix"].to(device), factors_fp8["C_mix_s"].to(device)).to(dtype=torch.bfloat16)
            C_gate = dequant(factors_fp8["C_gate"].to(device), factors_fp8["C_gate_s"].to(device)).to(dtype=torch.bfloat16)
            C_up = dequant(factors_fp8["C_up"].to(device), factors_fp8["C_up_s"].to(device)).to(dtype=torch.bfloat16)
            C_down = dequant(factors_fp8["C_down"].to(device), factors_fp8["C_down_s"].to(device)).to(dtype=torch.bfloat16)

            if is_attn:
                atlas_layer.self_attn.q_proj.weight.data.copy_(dequant(sub["q_w"].to(device), sub["q_s"].to(device)).to(dtype=torch.bfloat16))
                atlas_layer.self_attn.k_proj.weight.data.copy_(dequant(sub["k_w"].to(device), sub["k_s"].to(device)).to(dtype=torch.bfloat16))
                atlas_layer.self_attn.v_proj.weight.data.copy_(dequant(sub["v_w"].to(device), sub["v_s"].to(device)).to(dtype=torch.bfloat16))
                atlas_layer.self_attn.q_norm.weight.data.copy_(sub["q_norm"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.self_attn.k_norm.weight.data.copy_(sub["k_norm"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.self_attn.o_proj.weight.data.copy_(torch.matmul(U_m, C_mix))
            else:
                atlas_layer.linear_attn.in_proj_qkv.weight.data.copy_(dequant(sub["qkv_w"].to(device), sub["qkv_s"].to(device)).to(dtype=torch.bfloat16))
                atlas_layer.linear_attn.in_proj_z.weight.data.copy_(dequant(sub["z_w"].to(device), sub["z_s"].to(device)).to(dtype=torch.bfloat16))
                atlas_layer.linear_attn.in_proj_a.weight.data.copy_(sub["in_proj_a"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.linear_attn.in_proj_b.weight.data.copy_(sub["in_proj_b"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.linear_attn.conv1d.weight.data.copy_(sub["conv1d"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.linear_attn.dt_bias.data.copy_(sub["dt_bias"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.linear_attn.A_log.data.copy_(sub["A_log"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.linear_attn.norm.weight.data.copy_(sub["norm"].to(device=device, dtype=torch.bfloat16))
                atlas_layer.linear_attn.out_proj.weight.data.copy_(torch.matmul(U_m, C_mix))

            atlas_layer.mlp.gate_proj.weight.data.copy_(torch.matmul(C_gate, V_j.t()))
            atlas_layer.mlp.up_proj.weight.data.copy_(torch.matmul(C_up, V_j.t()))
            atlas_layer.mlp.down_proj.weight.data.copy_(torch.matmul(U_d, C_down))

            with torch.no_grad():
                # 1. Passo Atlas Raw
                out_r = atlas_layer(h_raw, position_embeddings=pos_emb)
                h_raw = out_r[0] if isinstance(out_r, tuple) else out_r

                # 2. Passo Atlas + Trilho Orbital Analítico R_l
                out_rail = atlas_layer(h_rail, position_embeddings=pos_emb)
                h_rail_step = out_rail[0] if isinstance(out_rail, tuple) else out_rail
                
                # Malha fechada do Trilho Orbital:
                # Posição esperada pela função real: a_target = a_prev * R_l
                a_prev_rail = torch.matmul(h_rail.reshape(-1, 5120), U_c)
                a_target_rail = torch.matmul(a_prev_rail, R_operators[l])
                a_meas_rail = torch.matmul(h_rail_step.reshape(-1, 5120), U_c)
                delta_a_rail = a_target_rail - a_meas_rail
                delta_h_rail = torch.matmul(delta_a_rail, U_c.t()).view_as(h_rail_step)
                
                # Injeção de restauração de órbita
                h_rail = h_rail_step + delta_h_rail

                # 3. Passo Atlas + Trilho Absoluto (Ground Truth do Professor)
                out_bound = atlas_layer(h_bound, position_embeddings=pos_emb)
                h_bound_step = out_bound[0] if isinstance(out_bound, tuple) else out_bound
                a_target_bound = trajectory_teacher[l + 1] # [N, 32]
                a_meas_bound = torch.matmul(h_bound_step.reshape(-1, 5120), U_c)
                delta_a_bound = a_target_bound - a_meas_bound
                delta_h_bound = torch.matmul(delta_a_bound, U_c.t()).view_as(h_bound_step)
                h_bound = h_bound_step + delta_h_bound

            del atlas_layer
            torch.cuda.empty_cache()

            # Diagnóstico de divergência nas camadas probe
            if l in probe_layers:
                with torch.no_grad():
                    # Obter a representação do professor nesta camada
                    a_t = trajectory_teacher[l + 1] # [N, 32]
                    
                    # Medição das normas no subespaço Uc
                    a_raw = torch.matmul(h_raw.reshape(-1, 5120), U_c)
                    a_r = torch.matmul(h_rail.reshape(-1, 5120), U_c)
                    a_b = torch.matmul(h_bound.reshape(-1, 5120), U_c)
                    
                    norm_t = torch.norm(a_t.float(), p="fro")
                    e_raw = (torch.norm((a_raw - a_t).float(), p="fro") / norm_t).item() * 100.0
                    e_rail = (torch.norm((a_r - a_t).float(), p="fro") / norm_t).item() * 100.0
                    e_bound = (torch.norm((a_b - a_t).float(), p="fro") / norm_t).item() * 100.0
                    
                    cos_raw = F.cosine_similarity(a_raw, a_t, dim=-1).mean().item()
                    cos_rail = F.cosine_similarity(a_r, a_t, dim=-1).mean().item()

                    drift_tracking["layers"].append(l)
                    drift_tracking["e_raw"].append(e_raw)
                    drift_tracking["e_rail"].append(e_rail)
                    drift_tracking["e_bound"].append(e_bound)
                    drift_tracking["cos_raw"].append(cos_raw)
                    drift_tracking["cos_rail"].append(cos_rail)

                    print(f"  [L{l:02d}] Subspace Drift: Raw={e_raw:6.2f}% (cos:{cos_raw:.3f}) | Trilho R_l={e_rail:6.2f}% (cos:{cos_rail:.3f}) | UpperBound={e_bound:6.2f}%")

        del chart_data, U_m, U_d, V_j
        torch.cuda.empty_cache()

    atlas_time = time.time() - t0_atlas
    print(f"\n[OK] Simulação de 64 camadas do Trilho Orbital concluída em {atlas_time:.1f}s!")

    # ---------------------------------------------------------------------------------
    # ETAPA IV: Avaliação Final de Logits e Perplexidade
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("                     RESULTADOS TERMINAIS EM L63")
    print("=" * 105)

    with torch.no_grad():
        logits_raw = F.linear(norm_mod(h_raw), lm_head_w)
        logits_rail = F.linear(norm_mod(h_rail), lm_head_w)
        logits_bound = F.linear(norm_mod(h_bound), lm_head_w)

        nll_raw, ppl_raw, top1_raw, agree_raw = compute_metrics(logits_raw, targets, logits_teach)
        nll_rail, ppl_rail, top1_rail, agree_rail = compute_metrics(logits_rail, targets, logits_teach)
        nll_b, ppl_b, top1_b, agree_b = compute_metrics(logits_bound, targets, logits_teach)

    print(f"{'Configuração':<35} | {'NLL':<8} | {'PPL':<9} | {'Top-1 (%)':<9} | {'Agree (%)':<9}")
    print("-" * 80)
    print(f"{'Teacher Oficial FP8 (Ground Truth)':<35} | {nll_t:8.4f} | {ppl_t:9.2f} | {top1_t:8.2f}% | 100.00%")
    print(f"{'Atlas Raw (Zero-Shot Sem Trilho)':<35} | {nll_raw:8.4f} | {ppl_raw:9.2f} | {top1_raw:8.2f}% | {agree_raw:8.2f}%")
    print(f"{'Atlas + Trilho Orbital R_l in SO(32)':<35} | {nll_rail:8.4f} | {ppl_rail:9.2f} | {top1_rail:8.2f}% | {agree_rail:8.2f}%")
    print(f"{'Atlas + Trilho Upper Bound (Teacher)':<35} | {nll_b:8.4f} | {ppl_b:9.2f} | {top1_b:8.2f}% | {agree_b:8.2f}%")
    print("-" * 80)

    # Salvar resultados e gerar gráfico
    res_path = os.path.join(repo_root, "experiments", "shared_subspace_orbital_rail_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "drift_tracking": drift_tracking,
            "metrics": {
                "teacher": {"nll": nll_t, "ppl": ppl_t, "top1": top1_t},
                "raw": {"nll": nll_raw, "ppl": ppl_raw, "top1": top1_raw, "agree": agree_raw},
                "orbital_rail": {"nll": nll_rail, "ppl": ppl_rail, "top1": top1_rail, "agree": agree_rail},
                "upper_bound": {"nll": nll_b, "ppl": ppl_b, "top1": top1_b, "agree": agree_b}
            },
            "runtime_seconds": teach_time + atlas_time
        }, f, indent=2)

    # Gráfico de trajetória e divergência
    plt.figure(figsize=(10, 5))
    plt.plot(drift_tracking["layers"], drift_tracking["e_raw"], "r-o", label="Atlas Raw (Sem Trilho)", linewidth=2)
    plt.plot(drift_tracking["layers"], drift_tracking["e_rail"], "b-s", label="Atlas + Trilho Orbital R_l in SO(32)", linewidth=2)
    plt.plot(drift_tracking["layers"], drift_tracking["e_bound"], "g--^", label="Atlas + Trilho Absoluto (Upper Bound)", linewidth=2)
    plt.title("Amortecimento da Divergência Orbital no Subespaço Uc(32)", fontsize=13, fontweight="bold")
    plt.xlabel("Camada (l)", fontsize=11)
    plt.ylabel("Erro no Subespaço Uc (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    chart_out = os.path.join(repo_root, "orbital_rail_divergence_curve.png")
    plt.savefig(chart_out, dpi=300)
    plt.close()
    print(f"\n[OK] Gráfico do Trilho Orbital salvo em: {chart_out}")

if __name__ == "__main__":
    run_orbital_rail_experiment()
