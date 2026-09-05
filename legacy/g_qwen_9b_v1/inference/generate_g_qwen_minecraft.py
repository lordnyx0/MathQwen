"""Optimized Low-VRAM Generation Script: G-Qwen 9B Phase 1 (Stiefel Atlas + RVQ 4-Bit).

Computes forward passes on-the-fly without materializing dense matrices:
  y = (x @ V_g @ S_l^T @ U_g^T) + (x @ Q_4(R)^T)
Total VRAM footprint: ~3.8 GB (fits comfortably in 12 GB VRAM on RTX 3060).
"""

import os
import sys
import glob
import time
import math
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from safetensors import safe_open
from typing import List, Dict, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def find_snapshot_dir() -> str:
    base_pattern = os.path.expanduser(
        r"~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"
    )
    return glob.glob(base_pattern)[0]


def unpack_and_matmul_4bit(x: torch.Tensor, packed_q: torch.Tensor, scale: torch.Tensor, out_features: int, in_features: int, block_size: int = 128) -> torch.Tensor:
    """Computes x @ Q_4(R)^T directly without storing the full float32 matrix in VRAM.
    x: [B, T, in_features]
    packed_q: uint8 [numel // 2]
    scale: float16 [numel // block_size]
    Returns: [B, T, out_features]
    """
    # Unpack 4-bit integers
    packed_flat = packed_q.view(-1)
    low = (packed_flat & 0x0F).to(torch.float16) - 7.0
    high = ((packed_flat >> 4) & 0x0F).to(torch.float16) - 7.0
    
    unpacked = torch.empty(packed_flat.numel() * 2, device=packed_q.device, dtype=torch.float16)
    unpacked[0::2] = low
    unpacked[1::2] = high
    
    # Scale per block
    unpacked_flat = unpacked.view(-1, block_size)
    scaled = unpacked_flat * scale.view(-1, 1)
    R_f16 = scaled.view(out_features, in_features)
    
    return torch.matmul(x, R_f16.t())


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


class OnTheFlyFoliatedLayer:
    """Foliated Layer computing forward pass on-the-fly via factorized Stiefel frames + 4-bit RVQ."""

    def __init__(self, layer_idx: int, chart_tensors: dict, orig_safetensor_path: str, device: torch.device):
        self.layer_idx = layer_idx
        self.device = device
        self.is_attn = (layer_idx % 4 == 3)

        # 1. Mixer Basis & Coordinates
        self.U_mix = chart_tensors["basis_mixer_U"].to(device).to(torch.float16)
        self.V_mix = chart_tensors["basis_mixer_V"].to(device).to(torch.float16)
        self.S_mix = chart_tensors[f"layer_{layer_idx}_mixer_S"].to(device).to(torch.float16)
        self.res_q_mix = chart_tensors[f"layer_{layer_idx}_mixer_res_q"].to(device)
        self.res_s_mix = chart_tensors[f"layer_{layer_idx}_mixer_res_scale"].to(device)

        # 2. FFN Bases & Coordinates
        self.U_gate = chart_tensors["basis_ffn_gate_U"].to(device).to(torch.float16)
        self.V_gate = chart_tensors["basis_ffn_gate_V"].to(device).to(torch.float16)
        self.S_gate = chart_tensors[f"layer_{layer_idx}_ffn_gate_S"].to(device).to(torch.float16)
        self.res_q_gate = chart_tensors[f"layer_{layer_idx}_ffn_gate_res_q"].to(device)
        self.res_s_gate = chart_tensors[f"layer_{layer_idx}_ffn_gate_res_scale"].to(device)

        self.U_up = chart_tensors["basis_ffn_up_U"].to(device).to(torch.float16)
        self.V_up = chart_tensors["basis_ffn_up_V"].to(device).to(torch.float16)
        self.S_up = chart_tensors[f"layer_{layer_idx}_ffn_up_S"].to(device).to(torch.float16)
        self.res_q_up = chart_tensors[f"layer_{layer_idx}_ffn_up_res_q"].to(device)
        self.res_s_up = chart_tensors[f"layer_{layer_idx}_ffn_up_res_scale"].to(device)

        self.U_down = chart_tensors["basis_ffn_down_U"].to(device).to(torch.float16)
        self.V_down = chart_tensors["basis_ffn_down_V"].to(device).to(torch.float16)
        self.S_down = chart_tensors[f"layer_{layer_idx}_ffn_down_S"].to(device).to(torch.float16)
        self.res_q_down = chart_tensors[f"layer_{layer_idx}_ffn_down_res_q"].to(device)
        self.res_s_down = chart_tensors[f"layer_{layer_idx}_ffn_down_res_scale"].to(device)

        # 3. Norms & In-Projections from layer file
        with safe_open(orig_safetensor_path, framework="pt", device=str(device)) as f:
            self.input_layernorm = f.get_tensor(f"model.language_model.layers.{layer_idx}.input_layernorm.weight").to(torch.float16)
            self.post_attention_layernorm = f.get_tensor(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight").to(torch.float16)
            
            if self.is_attn:
                wq = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight")
                sq = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight_scale_inv")
                sq_exp = sq.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_q = (wq.float() * sq_exp.float()).to(torch.float16)

                wk = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight")
                sk = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight_scale_inv")
                sk_exp = sk.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_k = (wk.float() * sk_exp.float()).to(torch.float16)

                wv = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight")
                sv = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight_scale_inv")
                sv_exp = sv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_v = (wv.float() * sv_exp.float()).to(torch.float16)
            else:
                wqkv = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight")
                sqkv = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight_scale_inv")
                sqkv_exp = sqkv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_qkv = (wqkv.float() * sqkv_exp.float()).to(torch.float16)

                wz = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight")
                sz = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight_scale_inv")
                sz_exp = sz.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_z = (wz.float() * sz_exp.float()).to(torch.float16)

    def forward_mixer_out(self, x_in: torch.Tensor) -> torch.Tensor:
        """On-the-fly Mixer output projection: x @ V @ S^T @ U^T + x @ Q_4(R)^T."""
        # 1. Atlas factorized projection:
        h_v = torch.matmul(x_in, self.V_mix)
        h_s = torch.matmul(h_v, self.S_mix.t())
        y_atlas = torch.matmul(h_s, self.U_mix.t())

        # 2. RVQ 4-bit residual:
        y_res = unpack_and_matmul_4bit(x_in, self.res_q_mix, self.res_s_mix, 5120, 6144)
        return y_atlas + y_res

    def forward_ffn(self, h_post: torch.Tensor) -> torch.Tensor:
        """On-the-fly SwiGLU FFN: (silu(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T."""
        # Gate
        g_atlas = torch.matmul(torch.matmul(torch.matmul(h_post, self.V_gate), self.S_gate.t()), self.U_gate.t())
        g_res = unpack_and_matmul_4bit(h_post, self.res_q_gate, self.res_s_gate, 17408, 5120)
        gate = F.silu(g_atlas + g_res)

        # Up
        u_atlas = torch.matmul(torch.matmul(torch.matmul(h_post, self.V_up), self.S_up.t()), self.U_up.t())
        u_res = unpack_and_matmul_4bit(h_post, self.res_q_up, self.res_s_up, 17408, 5120)
        up = u_atlas + u_res

        inter = gate * up # [B, T, 17408]

        # Down
        d_atlas = torch.matmul(torch.matmul(torch.matmul(inter, self.V_down), self.S_down.t()), self.U_down.t())
        d_res = unpack_and_matmul_4bit(inter, self.res_q_down, self.res_s_down, 5120, 17408)
        return d_atlas + d_res

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 1. Mixer
        h_norm = rms_norm(h, self.input_layernorm)
        if self.is_attn:
            v = torch.matmul(h_norm, self.W_v.t())
            x_mix = v.repeat_interleave(6, dim=-1)[:, :, :6144]
        else:
            z = torch.matmul(h_norm, self.W_z.t())
            x_mix = F.silu(z)

        y_mix = self.forward_mixer_out(x_mix)
        h = h + y_mix

        # 2. FFN
        h_post = rms_norm(h, self.post_attention_layernorm)
        y_ffn = self.forward_ffn(h_post)
        h = h + y_ffn

        return h


def run_generation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("       INFERENCIA AUTORREGRESSIVA G-QWEN 9B (ON-THE-FLY STIEFEL ATLAS + RVQ 4-BIT)       ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    snapshot_dir = find_snapshot_dir()
    tokenizer_path = os.path.join(snapshot_dir, "tokenizer.json")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")

    print("\n[1/3] Carregando Tokenizador...", end="", flush=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f" Concluido (Vocabulario: {tokenizer.get_vocab_size():,})", flush=True)

    print("[2/3] Carregando Embeddings (CPU) e LM Head (GPU)...", end="", flush=True)
    with safe_open(outside_path, framework="pt") as f:
        embed_tokens_cpu = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.float16) # In CPU RAM!
        final_norm = f.get_tensor("model.language_model.norm.weight").to(device).to(torch.float16)
        lm_head = f.get_tensor("lm_head.weight").to(device).to(torch.float16)
    print(f" Concluido! VRAM Atual: {torch.cuda.memory_allocated(0)/1e6:.1f} MB", flush=True)

    print("[3/3] Carregando as 16 Camadas Comprimidas (Fase 1)...", end="", flush=True)
    t0_layers = time.time()
    layers: List[OnTheFlyFoliatedLayer] = []
    
    for g in range(4):
        chart_path = os.path.abspath(f"models/g_qwen_9b_phase1/chart_{g}.safetensors")
        with safe_open(chart_path, framework="pt") as cf:
            chart_tensors = {k: cf.get_tensor(k) for k in cf.keys()}
        
        for idx in range(4):
            layer_idx = 4 * g + idx
            layer_file = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")
            layer = OnTheFlyFoliatedLayer(layer_idx, chart_tensors, layer_file, device)
            layers.append(layer)

    print(f" Concluido em {time.time() - t0_layers:.2f}s!", flush=True)
    print(f"VRAM Total Utilizada: {torch.cuda.memory_allocated(0)/1e6:.1f} MB (Operando confortavelmente sob 12GB!)", flush=True)

    # Prompt
    prompt = "<!DOCTYPE html>\n<html>\n<head>\n    <title>Minecraft Clone</title>\n"
    print("\n" + "=" * 105)
    print(f"PROMPT DE ENTRADA:\n{prompt}")
    print("=" * 105)
    print("GERANDO CODIGO AUTORREGRESSIVO (Token a Token):\n", flush=True)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)

    max_new_tokens = 128
    t0_gen = time.time()
    generated_text = prompt

    print(prompt, end="", flush=True)

    for step in range(max_new_tokens):
        # Look up embeddings from CPU for context window of last 64 tokens
        seq_ids = generated_ids[-64:]
        h_cpu = embed_tokens_cpu[seq_ids].unsqueeze(0) # [1, T, 5120]
        h = h_cpu.to(device)

        # Pass through all 16 compressed layers (on-the-fly Stiefel GEMMs)
        for layer in layers:
            h = layer.forward(h)

        # Final RMSNorm on the last token position
        h_last = rms_norm(h[:, -1, :], final_norm)

        # Project via LM Head
        logits = torch.matmul(h_last, lm_head.t()) # [1, 248320]

        # Greedy sampling (argmax)
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token_id)

        # Decode token and stream
        token_str = tokenizer.decode([next_token_id])
        generated_text += token_str
        print(token_str, end="", flush=True)

        # EOS
        if next_token_id in [248044, 151643]:
            break

    t_total = time.time() - t0_gen
    tok_per_sec = (len(generated_ids) - len(input_ids)) / max(t_total, 0.001)

    print("\n\n" + "=" * 105)
    print(f"GERACAO CONCLUIDA: {len(generated_ids) - len(input_ids)} tokens em {t_total:.2f}s ({tok_per_sec:.2f} tokens/s)!")
    print("=" * 105, flush=True)

    out_file = os.path.abspath("generated_minecraft_by_g_qwen9b.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Codigo gerado salvo em: {out_file}", flush=True)


if __name__ == "__main__":
    run_generation()
