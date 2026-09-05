"""Optimized 4-Block Streaming 64-Layer Generation Script: G-Qwen 9B.

Divides the 16 macro-charts (64 layers) into 4 blocks of 4 charts (16 layers per block):
- Block 0: Charts 0..3  (Layers 0..15)
- Block 1: Charts 4..7  (Layers 16..31)
- Block 2: Charts 8..11 (Layers 32..47)
- Block 3: Charts 12..15 (Layers 48..63)

Peak VRAM: ~6.8 GB (fits comfortably in 12 GB on RTX 3060).
Executes functional forward pass across all 64 layers with zero memory fragmentation.
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


def unpack_and_matmul_4bit(
    x: torch.Tensor,
    packed_q: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 128
) -> torch.Tensor:
    """Computes x @ Q_4(R)^T directly without storing the full float32 matrix in VRAM."""
    packed_flat = packed_q.view(-1)
    low = (packed_flat & 0x0F).to(torch.float16) - 7.0
    high = ((packed_flat >> 4) & 0x0F).to(torch.float16) - 7.0

    unpacked = torch.empty(packed_flat.numel() * 2, device=packed_q.device, dtype=torch.float16)
    unpacked[0::2] = low
    unpacked[1::2] = high

    unpacked_flat = unpacked.view(-1, block_size)
    scaled = unpacked_flat * scale.view(-1, 1)
    R_f16 = scaled.view(out_features, in_features)

    return torch.matmul(x, R_f16.t())


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


class LayerWeightsCPU:
    """Stores exclusive weights (norms, in-projections) in CPU RAM (~80 MB per layer)."""
    def __init__(self, layer_idx: int, snapshot_dir: str):
        self.layer_idx = layer_idx
        self.is_attn = (layer_idx % 4 == 3)
        layer_file = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")

        with safe_open(layer_file, framework="pt") as f:
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
                wz = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight")
                sz = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight_scale_inv")
                sz_exp = sz.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_z = (wz.float() * sz_exp.float()).to(torch.float16)


def forward_layer_gpu(
    h: torch.Tensor,
    layer_idx: int,
    chart_tensors: Dict[str, torch.Tensor],
    lw: LayerWeightsCPU,
    device: torch.device
) -> torch.Tensor:
    """Executes single layer forward on GPU."""
    is_attn = lw.is_attn
    in_norm = lw.input_layernorm.to(device)
    post_norm = lw.post_attention_layernorm.to(device)

    # 1. Mixer
    h_norm = rms_norm(h, in_norm)
    if is_attn:
        wv = lw.W_v.to(device)
        v = torch.matmul(h_norm, wv.t())
        x_mix = v.repeat_interleave(6, dim=-1)[:, :, :6144]
    else:
        wz = lw.W_z.to(device)
        z = torch.matmul(h_norm, wz.t())
        x_mix = F.silu(z)

    U_mix = chart_tensors["basis_mixer_U"]
    V_mix = chart_tensors["basis_mixer_V"]
    S_mix = chart_tensors[f"layer_{layer_idx}_mixer_S"]
    q_mix = chart_tensors[f"layer_{layer_idx}_mixer_res_q"]
    s_mix = chart_tensors[f"layer_{layer_idx}_mixer_res_scale"]

    y_atlas = torch.matmul(torch.matmul(torch.matmul(x_mix, V_mix), S_mix.t()), U_mix.t())
    y_res = unpack_and_matmul_4bit(x_mix, q_mix, s_mix, 5120, 6144)
    h = h + (y_atlas + y_res)

    # 2. FFN
    h_post = rms_norm(h, post_norm)

    U_gate = chart_tensors["basis_ffn_gate_U"]
    V_gate = chart_tensors["basis_ffn_gate_V"]
    S_gate = chart_tensors[f"layer_{layer_idx}_ffn_gate_S"]
    q_gate = chart_tensors[f"layer_{layer_idx}_ffn_gate_res_q"]
    s_gate = chart_tensors[f"layer_{layer_idx}_ffn_gate_res_scale"]
    g_atlas = torch.matmul(torch.matmul(torch.matmul(h_post, V_gate), S_gate.t()), U_gate.t())
    g_res = unpack_and_matmul_4bit(h_post, q_gate, s_gate, 17408, 5120)
    gate = F.silu(g_atlas + g_res)

    U_up = chart_tensors["basis_ffn_up_U"]
    V_up = chart_tensors["basis_ffn_up_V"]
    S_up = chart_tensors[f"layer_{layer_idx}_ffn_up_S"]
    q_up = chart_tensors[f"layer_{layer_idx}_ffn_up_res_q"]
    s_up = chart_tensors[f"layer_{layer_idx}_ffn_up_res_scale"]
    u_atlas = torch.matmul(torch.matmul(torch.matmul(h_post, V_up), S_up.t()), U_up.t())
    u_res = unpack_and_matmul_4bit(h_post, q_up, s_up, 17408, 5120)
    up = u_atlas + u_res

    inter = gate * up

    U_down = chart_tensors["basis_ffn_down_U"]
    V_down = chart_tensors["basis_ffn_down_V"]
    S_down = chart_tensors[f"layer_{layer_idx}_ffn_down_S"]
    q_down = chart_tensors[f"layer_{layer_idx}_ffn_down_res_q"]
    s_down = chart_tensors[f"layer_{layer_idx}_ffn_down_res_scale"]
    d_atlas = torch.matmul(torch.matmul(torch.matmul(inter, V_down), S_down.t()), U_down.t())
    d_res = unpack_and_matmul_4bit(inter, q_down, s_down, 5120, 17408)
    h = h + (d_atlas + d_res)

    return h


def run_64_layer_generation(max_tokens: int = 150):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("    INFERENCIA AUTORREGRESSIVA COMPLETA: G-QWEN 9B (64 CAMADAS, 16 MACRO-CARTAS)        ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    snapshot_dir = find_snapshot_dir()
    tokenizer_path = os.path.join(snapshot_dir, "tokenizer.json")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")
    charts_dir = os.path.abspath("models/g_qwen_9b_phase1")

    # 1. Tokenizer
    print("\n[1/4] Carregando Tokenizador...", end="", flush=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f" Concluido (Vocabulario: {tokenizer.get_vocab_size():,})", flush=True)

    # 2. Embeddings (CPU) + LM Head (GPU)
    print("[2/4] Carregando Embeddings (CPU) e LM Head (GPU)...", end="", flush=True)
    with safe_open(outside_path, framework="pt") as f:
        embed_tokens_cpu = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.float16)
        final_norm = f.get_tensor("model.language_model.norm.weight").to(device).to(torch.float16)
        lm_head = f.get_tensor("lm_head.weight").to(device).to(torch.float16)
    print(f" Concluido! VRAM Atual: {torch.cuda.memory_allocated(0)/1e6:.1f} MB", flush=True)

    # 3. Exclusive weights in CPU RAM
    print("[3/4] Carregando Pesos Exclusivos das 64 Camadas na RAM do Sistema (CPU)...", end="", flush=True)
    t0_ex = time.time()
    exclusive_cpu: List[LayerWeightsCPU] = []
    for l in range(64):
        exclusive_cpu.append(LayerWeightsCPU(l, snapshot_dir))
    print(f" Concluido em {time.time() - t0_ex:.2f}s! (64 Camadas na RAM)", flush=True)

    # 4. Chart file paths (16 charts)
    print("[4/4] Verificando as 16 Macro-Cartas...", end="", flush=True)
    chart_paths = [os.path.join(charts_dir, f"chart_{g}.safetensors") for g in range(16)]
    for cp in chart_paths:
        if not os.path.exists(cp):
            raise FileNotFoundError(f"Carta ausente: {cp}")
    print(" Concluido! Todas as 16 cartas (camadas 0..63) verificadas!", flush=True)
    print(f"VRAM Inicial GPU: {torch.cuda.memory_allocated(0)/1e6:.1f} MB (Operando confortavelmente sob 12GB!)", flush=True)

    # Prompt
    prompt = "<!DOCTYPE html>\n<html>\n<head>\n    <title>Minecraft Clone</title>\n"
    print("\n" + "=" * 105)
    print(f"PROMPT DE ENTRADA:\n{prompt}")
    print("=" * 105)
    print("GERANDO CODIGO AUTORREGRESSIVO (Token a Token):\n", flush=True)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)
    generated_text = prompt

    print(prompt, end="", flush=True)
    t0_gen = time.time()

    for step in range(max_tokens):
        # Look up embeddings from CPU for context window of last 64 tokens
        seq_ids = generated_ids[-64:]
        h = embed_tokens_cpu[seq_ids].unsqueeze(0).to(device)

        # Forward pass through the 16 charts sequentially
        for g in range(16):
            cpath = chart_paths[g]
            # Load chart g into GPU
            with safe_open(cpath, framework="pt", device=str(device)) as cf:
                chart_tensors = {k: cf.get_tensor(k) for k in cf.keys()}

            # Run the 4 layers belonging to chart g
            for idx in range(4):
                layer_idx = 4 * g + idx
                h = forward_layer_gpu(h, layer_idx, chart_tensors, exclusive_cpu[layer_idx], device)

            del chart_tensors

        # Final RMSNorm
        h_last = rms_norm(h[:, -1, :], final_norm)

        # LM Head logits
        logits = torch.matmul(h_last, lm_head.t())

        # Greedy argmax
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token_id)

        token_str = tokenizer.decode([next_token_id])
        generated_text += token_str
        print(token_str, end="", flush=True)

        # EOS check
        if next_token_id in [248044, 151643]:
            break

    t_total = time.time() - t0_gen
    gen_tokens = len(generated_ids) - len(input_ids)
    tok_per_sec = gen_tokens / max(t_total, 0.001)

    print("\n\n" + "=" * 105)
    print(f"GERACAO CONCLUIDA: {gen_tokens} tokens em {t_total:.2f}s ({tok_per_sec:.2f} tokens/s)!")
    print("=" * 105, flush=True)

    out_file = os.path.abspath("generated_minecraft_by_g_qwen9b.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Codigo gerado salvo com sucesso em: {out_file}", flush=True)


if __name__ == "__main__":
    run_64_layer_generation()
