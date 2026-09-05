# -*- coding: utf-8 -*-
"""Módulo canônico de calibração offline dos estabilizadores residuais e persistência em disco."""
import os
import sys
import time
import torch
import torch.nn.functional as F
import safetensors.torch as st
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import AtlasConfig
from .projection import compute_chart_atlas_bases, project_layer_weights
from .residual import (
    LinearResidualStabilizer,
    NonLinearResidualStabilizer,
    fit_svd_stabilizer,
    fit_gelu_warmstart,
    freeze_backbone_and_isolate_stabilizer
)
from reference.loader import load_qwen_reference_components, load_layer_module, dequant, get_snapshot_dir

def calibrate_offline_stabilizers(
    policy: str = "adaptive",
    output_path: str = "checkpoints/atlas_stabilizers_adaptive.pt",
    r_base: int = 2048,
    r_corr: int = 64,
    num_train_seqs: int = 16,
    seq_len: int = 64,
    device = None
):
    """
    Executa a calibração offline das 64 camadas e salva os estabilizadores em disco.
    
    Políticas suportadas:
      - 'adaptive': GELU nas camadas receptivas (L0, L32, L48, L63) e SVD-64 linear nas demais.
      - 'linear': SVD-64 linear uniforme em todas as 64 camadas.
      - 'nonlinear': GELU-64 não-linear uniforme em todas as 64 camadas.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"CALIBRAÇÃO OFFLINE DE ESTABILIZADORES ATLAS — POLÍTICA: {policy.upper()}")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"Saída: {output_path}")

    snapshot_dir = get_snapshot_dir()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t.strip() for t in ds["text"] if len(t.strip()) > 50])
    tokens_all = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    train_ids = tokens_all[:num_train_seqs * seq_len].view(num_train_seqs, seq_len).to(device)

    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=num_train_seqs)
    cfg = comp["cfg"]

    x_0 = F.embedding(train_ids.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_train_seqs, -1)
    pos_emb = comp["rotary"](x_0, pos_ids)

    h_prof = x_0.clone()
    h_stud = x_0.clone()

    stabilizers = {}
    gelu_layers = {0, 32, 48, 63} if policy == "adaptive" else (set(range(64)) if policy == "nonlinear" else set())

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
                out_p = layer_mod(h_prof, position_embeddings=pos_emb)
                h_prof_next = out_p[0] if isinstance(out_p, tuple) else out_p

            # 2. Configurar pesos Atlas no Aluno
            p_w = project_layer_weights(chart_weights[l], bases)
            if chart_weights[l]["is_attn"]:
                layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
            else:
                layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
            layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
            layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
            layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

            # 3. Forward Aluno Atlas
            with torch.no_grad():
                out_s = layer_mod(h_stud, position_embeddings=pos_emb)
                h_stud_in = out_s[0] if isinstance(out_s, tuple) else out_s

            # 4. Ajuste do Estabilizador conforme a política
            x_tr = h_stud_in.reshape(-1, 5120)
            y_tr = h_prof_next.reshape(-1, 5120)

            lambda_l = 5e-3 if (48 <= l <= 55) else 1e-3

            if l in gelu_layers:
                W_down, _ = fit_svd_stabilizer(x_tr, y_tr, r_corr=r_corr, lambda_reg=lambda_l)
                W_up = fit_gelu_warmstart(x_tr, y_tr, W_down, lambda_reg=lambda_l)
                stab = NonLinearResidualStabilizer(W_down, W_up)
            else:
                W_down, W_up = fit_svd_stabilizer(x_tr, y_tr, r_corr=r_corr, lambda_reg=lambda_l)
                stab = LinearResidualStabilizer(W_down, W_up)

            stab.freeze()
            stabilizers[l] = stab

            # 5. Aplicação da correção ao estado do aluno
            with torch.no_grad():
                h_stud_next = stab(h_stud_in)

            h_prof = h_prof_next
            h_stud = h_stud_next

            del layer_mod
            torch.cuda.empty_cache()

        print(f"  -> Carta {chart_idx:02d} calibrada e congelada.")

    t_total = time.time() - t_start
    print(f"\n[OK] Calibração de 64 camadas concluída em {t_total:.2f}s!")

    # 6. Salvar estabilizadores em disco
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    state = {}
    for l, stab in stabilizers.items():
        if isinstance(stab, LinearResidualStabilizer):
            state[f"stabilizer.{l}.type"] = "linear"
            state[f"stabilizer.{l}.W_down"] = stab.W_down.data.cpu()
            state[f"stabilizer.{l}.W_up"] = stab.W_up.data.cpu()
        elif isinstance(stab, NonLinearResidualStabilizer):
            state[f"stabilizer.{l}.type"] = "nonlinear"
            state[f"stabilizer.{l}.W_down"] = stab.W_down.data.cpu()
            state[f"stabilizer.{l}.W_up"] = stab.W_up.data.cpu()
            state[f"stabilizer.{l}.delta_alpha"] = stab.delta_alpha.data.cpu()

    torch.save(state, output_path)
    print(f"[OK] {len(stabilizers)} estabilizadores persistidos em: {output_path}")

    return stabilizers
