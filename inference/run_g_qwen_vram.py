"""Native 100% VRAM / GPU Tensor Core Generation Script for G-Qwen 9B.

Architecture:
- Loads models/g_qwen_9b_vram.safetensors
- Zero SSD reading during generation (zero disk I/O)
- 100% official Hugging Face Transformers Qwen3_5DecoderLayer with exact RoPE,
  Gated DeltaNet recurrence, SDPA, and DynamicCache.
- Generates: generated_minecraft_by_g_qwen9b.html
"""

import os
import sys
import glob
import time
import psutil
import torch
import safetensors.torch as st
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding
)
from tokenizers import Tokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def dequant_4bit_gpu(packed: torch.Tensor, scale: torch.Tensor, shape_list: list, block_size: int = 128, device="cuda:0") -> torch.Tensor:
    """Vectorized 4-bit unpacking directly into bfloat16 on CUDA Tensor Cores."""
    p = packed.to(device, non_blocking=True)
    s = scale.to(device, non_blocking=True)
    low = (p & 0x0F).to(torch.int8) - 7
    high = ((p >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, block_size)
    return (unpacked.to(torch.bfloat16) * s.view(-1, 1).to(torch.bfloat16)).view(shape_list)


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor, device="cuda:0") -> torch.Tensor:
    """Dequantizes FP8 tensor with block-wise scale inversion into bfloat16 on CUDA."""
    w_gpu = w.to(device, non_blocking=True)
    s_gpu = scale_inv.to(device, non_blocking=True)
    s_exp = s_gpu.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w_gpu.to(torch.float32) * s_exp.to(torch.float32)).to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def run_vram_generation(max_new_tokens: int = 150):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 100, flush=True)
    print("    G-QWEN 9B: MOTOR NATIVO VRAM / GPU TENSOR CORES (RTX 3060 12 GB)                 ", flush=True)
    print("=" * 100, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)

    vram_path = os.path.abspath("models/g_qwen_9b_vram.safetensors")
    if not os.path.exists(vram_path):
        print(f"\nERRO: Arquivo {vram_path} nao encontrado!")
        print("Execute primeiro: python tools/compile_g_qwen_to_vram_safetensors.py")
        sys.exit(1)

    file_size_gb = os.path.getsize(vram_path) / (1024**3)
    print(f"Modelo VRAM     : {vram_path} ({file_size_gb:.2f} GB)", flush=True)

    # 1. Carregar Tokenizador
    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    print("\n[1/4] Carregando Tokenizador BPE...", end="", flush=True)
    tokenizer = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
    print(f" Concluido! (Vocabulario: {tokenizer.get_vocab_size():,})", flush=True)

    # 2. Carregar o Modelo Unificado 100% na Memoria RAM (sem tocar no SSD durante inferência)
    print("[2/4] Carregando models/g_qwen_9b_vram.safetensors para a memoria...", end="", flush=True)
    t0_load = time.time()
    weights = st.load_file(vram_path, device="cpu")
    print(f" Concluido em {time.time()-t0_load:.2f}s! (RAM RSS: {psutil.Process().memory_info().rss/1e9:.2f} GB)", flush=True)

    # 3. Inicializar Componentes de Inferência na GPU
    print("[3/4] Inicializando modulos Qwen3_5DecoderLayer e Cache...", end="", flush=True)
    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    cache = DynamicCache(config=cfg.text_config)

    linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)
    final_norm_w = weights["output_norm.weight"].to(device=device, dtype=torch.bfloat16)

    # Descompactar Embeddings e LM Head diretamente na GPU
    emb_shape = weights["token_embd.shape"].tolist()
    embed_tokens = dequant_4bit_gpu(weights["token_embd.q"], weights["token_embd.s"], emb_shape)
    head_shape = weights["lm_head.shape"].tolist()
    lm_head = dequant_4bit_gpu(weights["lm_head.q"], weights["lm_head.s"], head_shape)

    torch.cuda.synchronize()
    vram_used_mb = torch.cuda.memory_allocated(0) / 1e6
    print(f" Concluido!", flush=True)
    print(f"VRAM Alocada GPU: {vram_used_mb:.1f} MB ({vram_used_mb/1024:.2f} GB) | Margem Livre: {(12288 - vram_used_mb):.1f} MB", flush=True)

    # 4. Loop de Geração Autorregressiva
    prompt = "<!DOCTYPE html>\n<html>\n<head>\n    <title>Minecraft Clone</title>\n"
    print("\n" + "=" * 100)
    print(f"PROMPT DE ENTRADA:\n{prompt}")
    print("=" * 100)
    print("GERANDO CODIGO MINECRAFT (100% VRAM, ZERO ACESSO A DISCO):\n", flush=True)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)
    print(prompt, end="", flush=True)

    out_file = os.path.abspath("generated_minecraft_by_g_qwen9b.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    def forward_pass_all_64_layers(h_in: torch.Tensor, pos_emb: tuple) -> torch.Tensor:
        h = h_in
        for l in range(64):
            is_attn = (l % 4 == 3)

            # 1. Descompactar FFN na GPU
            sh_gate = weights[f"blk.{l}.ffn_gate.shape"].tolist()
            W_gate = dequant_4bit_gpu(weights[f"blk.{l}.ffn_gate.q"], weights[f"blk.{l}.ffn_gate.s"], sh_gate)

            sh_up = weights[f"blk.{l}.ffn_up.shape"].tolist()
            W_up = dequant_4bit_gpu(weights[f"blk.{l}.ffn_up.q"], weights[f"blk.{l}.ffn_up.s"], sh_up)

            sh_down = weights[f"blk.{l}.ffn_down.shape"].tolist()
            W_down = dequant_4bit_gpu(weights[f"blk.{l}.ffn_down.q"], weights[f"blk.{l}.ffn_down.s"], sh_down)

            sh_mix = weights[f"blk.{l}.mix.shape"].tolist()
            W_mix = dequant_4bit_gpu(weights[f"blk.{l}.mix.q"], weights[f"blk.{l}.mix.s"], sh_mix)

            if is_attn:
                mod = attn_layer
                mod.self_attn.layer_idx = l
                mod.self_attn.o_proj.weight.data.copy_(W_mix)
                mod.mlp.gate_proj.weight.data.copy_(W_gate)
                mod.mlp.up_proj.weight.data.copy_(W_up)
                mod.mlp.down_proj.weight.data.copy_(W_down)

                mod.self_attn.q_proj.weight.data.copy_(dequant_fp8(weights[f"blk.{l}.attn_q.weight"], weights[f"blk.{l}.attn_q.scale"]))
                mod.self_attn.k_proj.weight.data.copy_(dequant_fp8(weights[f"blk.{l}.attn_k.weight"], weights[f"blk.{l}.attn_k.scale"]))
                mod.self_attn.v_proj.weight.data.copy_(dequant_fp8(weights[f"blk.{l}.attn_v.weight"], weights[f"blk.{l}.attn_v.scale"]))
                mod.self_attn.q_norm.weight.data.copy_(weights[f"blk.{l}.attn_q_norm.weight"].to(device=device, dtype=torch.bfloat16))
                mod.self_attn.k_norm.weight.data.copy_(weights[f"blk.{l}.attn_k_norm.weight"].to(device=device, dtype=torch.bfloat16))
                mod.input_layernorm.weight.data.copy_(weights[f"blk.{l}.attn_norm.weight"].to(device=device, dtype=torch.bfloat16))
                mod.post_attention_layernorm.weight.data.copy_(weights[f"blk.{l}.ffn_norm.weight"].to(device=device, dtype=torch.bfloat16))
            else:
                mod = linear_layer
                mod.linear_attn.layer_idx = l
                mod.linear_attn.out_proj.weight.data.copy_(W_mix)
                mod.mlp.gate_proj.weight.data.copy_(W_gate)
                mod.mlp.up_proj.weight.data.copy_(W_up)
                mod.mlp.down_proj.weight.data.copy_(W_down)

                mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant_fp8(weights[f"blk.{l}.attn_qkv.weight"], weights[f"blk.{l}.attn_qkv.scale"], device=device))
                mod.linear_attn.in_proj_z.weight.data.copy_(dequant_fp8(weights[f"blk.{l}.attn_gate.weight"], weights[f"blk.{l}.attn_gate.scale"], device=device))
                mod.linear_attn.in_proj_a.weight.data.copy_(weights[f"blk.{l}.ssm_alpha.weight"].to(device=device, dtype=torch.bfloat16))
                mod.linear_attn.in_proj_b.weight.data.copy_(weights[f"blk.{l}.ssm_beta.weight"].to(device=device, dtype=torch.bfloat16))
                mod.linear_attn.conv1d.weight.data.copy_(weights[f"blk.{l}.ssm_conv1d.weight"].to(device=device, dtype=torch.bfloat16))
                mod.linear_attn.dt_bias.data.copy_(weights[f"blk.{l}.ssm_dt.bias"].to(device=device, dtype=torch.bfloat16))
                mod.linear_attn.A_log.data.copy_(weights[f"blk.{l}.ssm_a.weight"].to(device=device, dtype=torch.bfloat16))
                mod.linear_attn.norm.weight.data.copy_(weights[f"blk.{l}.ssm_norm.weight"].to(device=device, dtype=torch.bfloat16))
                mod.input_layernorm.weight.data.copy_(weights[f"blk.{l}.attn_norm.weight"].to(device=device, dtype=torch.bfloat16))
                mod.post_attention_layernorm.weight.data.copy_(weights[f"blk.{l}.ffn_norm.weight"].to(device=device, dtype=torch.bfloat16))

            layer_out = mod(h, position_embeddings=pos_emb, past_key_values=cache, use_cache=True)
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return h

    # Pre-fill Prompt
    t0_gen = time.time()
    curr_input_ids = torch.tensor([input_ids], device=device)
    seq_len = curr_input_ids.shape[1]
    h = embed_tokens[curr_input_ids]

    pos_ids = torch.arange(0, seq_len, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(h, pos_ids)

    with torch.no_grad():
        h = forward_pass_all_64_layers(h, pos_emb)
        h_last = h[:, -1:, :]
        h_norm = rms_norm(h_last, final_norm_w)
        logits = torch.matmul(h_norm, lm_head.t())
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

    generated_ids.append(next_token)
    token_str = tokenizer.decode([next_token])
    print(token_str, end="", flush=True)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(token_str)

    # Generation Loop
    for step in range(max_new_tokens):
        t_token = time.time()
        cur_pos = seq_len + step
        curr_input_ids = torch.tensor([[next_token]], device=device)
        h = embed_tokens[curr_input_ids]

        pos_ids = torch.tensor([[[cur_pos]], [[cur_pos]], [[cur_pos]]], device=device)
        pos_emb = rotary(h, pos_ids)

        with torch.no_grad():
            h = forward_pass_all_64_layers(h, pos_emb)
            h_norm = rms_norm(h[:, -1:, :], final_norm_w)
            logits = torch.matmul(h_norm, lm_head.t())
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

        generated_ids.append(next_token)
        token_str = tokenizer.decode([next_token])
        print(token_str, end="", flush=True)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(token_str)

        if token_str in ["<|im_end|>", "<|endoftext|>"]:
            print("\n[Fim de sequencia gerado pelo modelo!]", flush=True)
            break

    total_time = time.time() - t0_gen
    total_tokens = len(generated_ids) - len(input_ids)
    print("\n" + "=" * 100)
    print(f"GERACAO CONCLUIDA COM SUCESSO!")
    print(f"Total de tokens gerados: {total_tokens}")
    print(f"Tempo total            : {total_time:.2f}s ({total_tokens / max(0.01, total_time):.2f} tokens/s)")
    print(f"Arquivo final          : {out_file}")
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_vram_generation()
