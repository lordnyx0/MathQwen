# -*- coding: utf-8 -*-
"""Carregamento oficial dos pesos do checkpoint Qwen/Qwen3.8-27B-FP8."""
import os
import glob
import torch
import torch.nn.functional as F
import safetensors.torch as st
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5RMSNorm
)

def get_snapshot_dir():
    paths = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))
    if not paths:
        raise FileNotFoundError("Checkpoint Qwen/Qwen3.8-27B-FP8 não encontrado no cache do HuggingFace.")
    return paths[0]

def dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.float() * s_exp.float())

def load_qwen_reference_components(device=None, seq_len=64, num_seqs=16):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_dir = get_snapshot_dir()
    cfg = AutoConfig.from_pretrained(snapshot_dir)

    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    with st.safe_open(outside_path, framework="pt") as f:
        embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
        norm_w = f.get_tensor("model.language_model.norm.weight").to(device=device, dtype=torch.bfloat16)
        lm_head_w = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)

    norm_mod = Qwen3_5RMSNorm(cfg.text_config.hidden_size, eps=cfg.text_config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm_mod.weight.data.copy_(norm_w)

    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    dummy_x = torch.zeros(num_seqs, seq_len, cfg.text_config.hidden_size, device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
    pos_emb = rotary(dummy_x, pos_ids)

    return {
        "cfg": cfg,
        "device": device,
        "snapshot_dir": snapshot_dir,
        "embed_w": embed_w,
        "lm_head_w": lm_head_w,
        "norm_mod": norm_mod,
        "rotary": rotary,
        "pos_emb": pos_emb
    }

def load_layer_module(layer_idx: int, cfg, device, snapshot_dir=None):
    if snapshot_dir is None:
        snapshot_dir = get_snapshot_dir()
    layer_mod = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=layer_idx).to(device=device, dtype=torch.bfloat16)
    lp = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")
    is_attn = (layer_idx % 4 == 3)

    with st.safe_open(lp, framework="pt") as f:
        layer_mod.input_layernorm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.input_layernorm.weight").to(device=device, dtype=torch.bfloat16))
        layer_mod.post_attention_layernorm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight").to(device=device, dtype=torch.bfloat16))

        if is_attn:
            layer_mod.self_attn.q_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.self_attn.k_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.self_attn.v_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.self_attn.o_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.self_attn.q_norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight").to(device=device, dtype=torch.bfloat16))
            layer_mod.self_attn.k_norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight").to(device=device, dtype=torch.bfloat16))
        else:
            layer_mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.in_proj_z.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.in_proj_a.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a.weight").to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.in_proj_b.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b.weight").to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.out_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.conv1d.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.conv1d.weight").to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.dt_bias.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.dt_bias").to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.A_log.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.A_log").to(device=device, dtype=torch.bfloat16))
            layer_mod.linear_attn.norm.weight.data.copy_(f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.norm.weight").to(device=device, dtype=torch.bfloat16))

        layer_mod.mlp.gate_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
        layer_mod.mlp.up_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))
        layer_mod.mlp.down_proj.weight.data.copy_(dequant(f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16))

    return layer_mod
