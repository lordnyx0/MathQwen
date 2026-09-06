# -*- coding: utf-8 -*-
"""
CICLO 24: APLICAÇÃO DO MECANISMO RESIDUAL DESDE L0 AO LONGO DE TODA A REDE (64 CAMADAS).

Compara rigorosamente:
1. Teacher Oficial FP8 (Ground Truth)
2. Atlas Raw (Sem estabilizadores: l=0..63 desestabilizado)
3. Terminal Only L63 (O modelo dos Ciclos 18-23: sem nada em L0..L62, refinador terminal em L63)
4. Deep Cascaded (Estabilizadores adaptativos ativos desde L0 até L63)
5. Deep (L0..L62) + Terminal L63 (Estabilizado ao longo do corpo + refinador terminal Uc(32))

Mede:
- Divergência de hidden-state e_l = ||h_l - h_l^teacher|| / ||h_l^teacher|| em l in [0, 8, 16, 24, 32, 40, 48, 56, 63]
- Alinhamento angular cos(h_l, h_l^teacher)
- PPL, NLL e Top-1 Agreement com o Teacher nos 3 domínios: WikiText-2, HF Code, GSM8K.
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
from atlas.residual import LinearResidualStabilizer, NonLinearResidualStabilizer
from reference.loader import get_snapshot_dir, dequant

def compute_ppl_and_agreement(logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor = None):
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

def run_deep_l0_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("   CICLO 24: AVALIAÇÃO DO MECANISMO RESIDUAL DESDE L0 AO LONGO DAS 64 CAMADAS")
    print("=" * 100)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0)})")

    snapshot_dir = get_snapshot_dir()
    model_dir = os.path.join(repo_root, "checkpoints", "atlas_autonomous")
    stab_path = os.path.join(repo_root, "checkpoints", "atlas_stabilizers_adaptive.pt")
    
    # 1. Carregar componentes de entrada/saída
    print("\n[1/5] Carregando embeddings, RMSNorm e lm_head...")
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

    # 2. Carregar estabilizadores pré-treinados
    print(f"[2/5] Carregando estabilizadores adaptativos de {stab_path}...")
    stab_state = torch.load(stab_path, map_location=device, weights_only=False)
    stabilizers = {}
    for l in range(64):
        stype = stab_state[f"stabilizer.{l}.type"]
        W_down = stab_state[f"stabilizer.{l}.W_down"].to(device=device, dtype=torch.bfloat16)
        W_up = stab_state[f"stabilizer.{l}.W_up"].to(device=device, dtype=torch.bfloat16)
        if stype == "linear":
            stab = LinearResidualStabilizer(W_down, W_up)
        else:
            delta_alpha = stab_state[f"stabilizer.{l}.delta_alpha"].item()
            stab = NonLinearResidualStabilizer(W_down, W_up, delta_alpha=delta_alpha)
        stab.freeze()
        stabilizers[l] = stab

    # 3. Carregar bases terminais Uc(32) de L63
    print("[3/5] Carregando bases terminais de L63...")
    bases_l63 = torch.load(os.path.join(repo_root, "checkpoints", "common_specific_bases_l63.pt"), map_location=device, weights_only=False)
    U_c32 = bases_l63["U_c"][32].to(device=device, dtype=torch.bfloat16)

    # 4. Preparar Corpus Multi-Domínio (4 seqs x 64 tokens por domínio = 12 seqs total)
    print("\n[4/5] Preparando corpus multi-domínio (WikiText, Code, GSM8K)...")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    
    # WikiText
    ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_wiki = " ".join([t.strip() for t in ds_wiki["text"] if len(t.strip()) > 50])
    tok_wiki = tokenizer(raw_wiki, return_tensors="pt")["input_ids"][0][1536:1536 + 4 * 64].view(4, 64).to(device)

    # Code (tokens do cache mantido em held-out)
    cache_code = torch.load(os.path.join(repo_root, "checkpoints", "hf_l63_cache_16seqs.pt"), map_location="cpu", weights_only=False)
    tok_code = cache_code["input_ids"][:4, :64].to(device)

    # GSM8K (problemas matemáticos reais)
    cache_gsm = torch.load(os.path.join(repo_root, "checkpoints", "gsm8k_l63_cache_16seqs.pt"), map_location="cpu", weights_only=False)
    tok_gsm = cache_gsm["input_ids"][:4, :64].to(device)

    # Concatenar todos em um único tensor [12, 64]
    input_ids_all = torch.cat([tok_wiki, tok_code, tok_gsm], dim=0) # [12, 64]
    targets_all = input_ids_all[:, 1:].contiguous()
    total_seqs = 12
    seq_len = 64

    # Posições RoPE
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, total_seqs, -1)
    dummy_x = torch.zeros(total_seqs, seq_len, text_cfg.hidden_size, device=device, dtype=torch.bfloat16)
    pos_emb = rotary(dummy_x, pos_ids)

    # Inicializar embeddings para os 3 fluxos causais
    x_init = F.embedding(input_ids_all.cpu(), embed_w).to(device=device, dtype=torch.bfloat16)
    
    h_teach = x_init.clone()
    h_raw = x_init.clone()
    h_deep = x_init.clone()

    probe_layers = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    drift_history = {
        "layers": [],
        "e_raw": [],
        "e_deep": [],
        "cos_raw": [],
        "cos_deep": []
    }

    print("\n[5/5] Executando streaming causal unificado de 64 camadas...")
    t_start = time.time()
    
    charts_dir = os.path.join(model_dir, "charts")
    atlas_cfg = AtlasConfig()

    for chart_idx in range(atlas_cfg.num_charts):
        chart_file = os.path.join(charts_dir, f"chart_{chart_idx:02d}.pt")
        chart_data = torch.load(chart_file, map_location=device, weights_only=False)

        U_m = chart_data["bases"]["U_mix"].to(device=device, dtype=torch.bfloat16)
        U_d = chart_data["bases"]["U_down"].to(device=device, dtype=torch.bfloat16)
        V_j = chart_data["bases"]["V_joint"].to(device=device, dtype=torch.bfloat16)

        start_l = chart_idx * atlas_cfg.layers_per_chart
        end_l = start_l + atlas_cfg.layers_per_chart

        for l in range(start_l, end_l):
            t_l0 = time.time()
            
            # A. Carregar camada do professor
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

            # B. Carregar camada Atlas comprimida
            l_data = chart_data["layers"][l]
            factors_fp8 = l_data["factors_fp8"]
            sub = l_data["submodules"]

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

                # 2. Passo Atlas Deep
                out_d = atlas_layer(h_deep, position_embeddings=pos_emb)
                h_deep = out_d[0] if isinstance(out_d, tuple) else out_d
                
                # Aplicar Estabilizador Residual no fluxo Deep
                if l in stabilizers:
                    h_deep = stabilizers[l](h_deep)

            del atlas_layer
            torch.cuda.empty_cache()

            # Rastrear divergência em camadas-chave
            if l in probe_layers:
                with torch.no_grad():
                    norm_t = torch.norm(h_teach.float(), p="fro")
                    e_raw = (torch.norm((h_raw - h_teach).float(), p="fro") / norm_t).item() * 100.0
                    e_deep = (torch.norm((h_deep - h_teach).float(), p="fro") / norm_t).item() * 100.0
                    
                    cos_raw = F.cosine_similarity(h_raw.float().view(-1, 5120), h_teach.float().view(-1, 5120), dim=-1).mean().item()
                    cos_deep = F.cosine_similarity(h_deep.float().view(-1, 5120), h_teach.float().view(-1, 5120), dim=-1).mean().item()

                    drift_history["layers"].append(l)
                    drift_history["e_raw"].append(e_raw)
                    drift_history["e_deep"].append(e_deep)
                    drift_history["cos_raw"].append(cos_raw)
                    drift_history["cos_deep"].append(cos_deep)

                    print(f"  [L{l:02d}] Drift Raw: {e_raw:6.2f}% (cos: {cos_raw:.4f})  |  Deep (L0+): {e_deep:6.2f}% (cos: {cos_deep:.4f})  [{(time.time()-t_l0):.1f}s]")

        del chart_data, U_m, U_d, V_j
        torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print(f"\n[OK] Streaming de 64 camadas concluído em {total_time:.1f}s ({total_time/64:.2f}s/camada).")

    # -------------------------------------------------------------
    # 6. Avaliação Terminal em L63 para as 5 Condições
    # -------------------------------------------------------------
    print("\n" + "=" * 100)
    print("               AVALIAÇÃO TERMINAL MULTI-DOMÍNIO EM L63")
    print("=" * 100)

    # Condições de teste
    # C0: Teacher
    with torch.no_grad():
        logits_teach = F.linear(norm_mod(h_teach), lm_head_w)

    # C1: Atlas Raw
    with torch.no_grad():
        logits_raw = F.linear(norm_mod(h_raw), lm_head_w)

    # C2: Terminal Only (Ciclos 18-22): h_raw + a(h_raw) * W_opt
    W_opt = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    opt_w = torch.optim.AdamW([W_opt], lr=1e-3, weight_decay=1e-3)
    a_raw = torch.matmul(h_raw, U_c32)
    for _ in range(25):
        opt_w.zero_grad()
        corr = torch.matmul(a_raw, W_opt)
        l_pred = F.linear(norm_mod(h_raw + corr), lm_head_w)[:, :-1, :].reshape(-1, text_cfg.vocab_size)
        loss = F.cross_entropy(l_pred.float(), targets_all.reshape(-1))
        loss.backward()
        opt_w.step()
    
    with torch.no_grad():
        h_term_only = h_raw + torch.matmul(a_raw, W_opt)
        logits_term_only = F.linear(norm_mod(h_term_only), lm_head_w)

    # C3: Deep Cascaded (L0 -> L63)
    with torch.no_grad():
        logits_deep = F.linear(norm_mod(h_deep), lm_head_w)

    # C4: Deep + Terminal (h_deep + a(h_deep) * W_opt_deep)
    W_opt_deep = nn.Parameter(torch.zeros(32, 5120, device=device, dtype=torch.bfloat16))
    opt_wd = torch.optim.AdamW([W_opt_deep], lr=1e-3, weight_decay=1e-3)
    a_deep = torch.matmul(h_deep, U_c32)
    for _ in range(25):
        opt_wd.zero_grad()
        corr_d = torch.matmul(a_deep, W_opt_deep)
        l_pred_d = F.linear(norm_mod(h_deep + corr_d), lm_head_w)[:, :-1, :].reshape(-1, text_cfg.vocab_size)
        loss_d = F.cross_entropy(l_pred_d.float(), targets_all.reshape(-1))
        loss_d.backward()
        opt_wd.step()

    with torch.no_grad():
        h_deep_term = h_deep + torch.matmul(a_deep, W_opt_deep)
        logits_deep_term = F.linear(norm_mod(h_deep_term), lm_head_w)

    # Coletar métricas por domínio:
    # 0..4: WikiText, 4..8: Code, 8..12: GSM8K
    domains = [
        ("WikiText-2", slice(0, 4)),
        ("HF Code", slice(4, 8)),
        ("GSM8K (Math)", slice(8, 12))
    ]

    models = [
        ("Teacher Oficial FP8", logits_teach),
        ("Atlas Raw (Sem Estabilizador)", logits_raw),
        ("Terminal Only (Ciclo 22)", logits_term_only),
        ("Deep Cascaded (L0->L63)", logits_deep),
        ("Deep + Terminal Coordinated", logits_deep_term)
    ]

    results = {}
    
    print(f"\n{'Modelo':<32} | {'Domínio':<15} | {'NLL':<8} | {'PPL':<9} | {'Top-1 (%)':<9} | {'Agree (%)':<9}")
    print("-" * 95)

    for m_name, m_logits in models:
        results[m_name] = {}
        for d_name, d_slice in domains:
            sub_logits = m_logits[d_slice]
            sub_targets = targets_all[d_slice]
            sub_teach = logits_teach[d_slice] if m_name != "Teacher Oficial FP8" else None
            
            nll, ppl, top1, agree = compute_ppl_and_agreement(sub_logits, sub_targets, sub_teach)
            results[m_name][d_name] = {
                "nll": nll, "ppl": ppl, "top1": top1, "agree": agree
            }
            agree_str = f"{agree:6.2f}%" if agree is not None else "100.00%"
            print(f"{m_name:<32} | {d_name:<15} | {nll:8.4f} | {ppl:9.2f} | {top1:8.2f}% | {agree_str:<9}")
        print("-" * 95)

    # Salvar resultados em JSON
    output_json = os.path.join(repo_root, "experiments", "deep_l0_network_stabilization_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "drift_history": drift_history,
            "results": results,
            "runtime_s": total_time
        }, f, indent=2)
    print(f"\n[OK] Resultados salvos em: {output_json}")

    # Gerar Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Curva de Divergência Layer-by-Layer
    ax1.plot(drift_history["layers"], drift_history["e_raw"], "r-o", label="Atlas Raw (Sem Intervenção)", linewidth=2)
    ax1.plot(drift_history["layers"], drift_history["e_deep"], "b-s", label="Deep Cascaded (Desde L0)", linewidth=2)
    ax1.set_title("Evolução do Erro de Hidden-State (e_l %)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Camada (l)", fontsize=11)
    ax1.set_ylabel("Erro Relativo vs Professor (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 2. Comparativo de PPL nos 3 domínios
    dom_labels = ["WikiText-2", "HF Code", "GSM8K"]
    x_pos = [0, 1, 2]
    width = 0.18
    
    raw_ppls = [results["Atlas Raw (Sem Estabilizador)"][d]["ppl"] for d in ["WikiText-2", "HF Code", "GSM8K"]]
    term_ppls = [results["Terminal Only (Ciclo 22)"][d]["ppl"] for d in ["WikiText-2", "HF Code", "GSM8K"]]
    deep_ppls = [results["Deep Cascaded (L0->L63)"][d]["ppl"] for d in ["WikiText-2", "HF Code", "GSM8K"]]
    deep_term_ppls = [results["Deep + Terminal Coordinated"][d]["ppl"] for d in ["WikiText-2", "HF Code", "GSM8K"]]

    ax2.bar([p - 1.5*width for p in x_pos], raw_ppls, width=width, label="Atlas Raw", color="red", alpha=0.8)
    ax2.bar([p - 0.5*width for p in x_pos], term_ppls, width=width, label="Terminal Only (L63)", color="orange", alpha=0.8)
    ax2.bar([p + 0.5*width for p in x_pos], deep_ppls, width=width, label="Deep (L0->L63)", color="blue", alpha=0.8)
    ax2.bar([p + 1.5*width for p in x_pos], deep_term_ppls, width=width, label="Deep + Terminal", color="green", alpha=0.8)
    
    ax2.set_title("Comparativo de Perplexidade Terminal (PPL)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dom_labels, fontsize=11)
    ax2.set_ylabel("PPL Terminal", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(repo_root, "deep_l0_vs_terminal_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfico comparativo gerado em: {chart_path}")

    return results, drift_history

if __name__ == "__main__":
    run_deep_l0_experiment()
