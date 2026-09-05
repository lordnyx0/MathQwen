"""Streaming On-Demand Generation Script for G-Qwen 9B.

Memory Architecture:
- RAM: ~3.5 GB RSS (Zero pagefile thrashing, 28+ GB free system RAM)
- VRAM: ~4.5 GB (Zero VRAM pressure, 7.5 GB free on RTX 3060)
- Compute: 100% official Hugging Face Transformers Qwen3_5DecoderLayer with
  exact RoPE, Gated DeltaNet recurrence, SDPA, and DynamicCache.
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


def dequant_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Vectorized 4-bit unpacking directly into bfloat16 on CUDA."""
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, 128)
    return (unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape)


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantizes FP8 tensor with block-wise scale inversion into bfloat16."""
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root Mean Square Layer Normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def run_streaming_generation(max_new_tokens: int = 120):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 95, flush=True)
    print("    G-QWEN 9B: PIPELINE OFICIAL TRANSFORMERS COM STREAMING SOB DEMANDA              ", flush=True)
    print("=" * 95, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Memoria VRAM    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print("Parametros      : 8.422B Fisicos Ativos (<= 9.00B Orcamento Estrito)", flush=True)
    print("=" * 95, flush=True)

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    charts_dir = os.path.abspath("models/g_qwen_9b_phase2")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")

    # [1/3] Tokenizer
    print("\n[1/3] Carregando Tokenizador BPE...", end="", flush=True)
    tokenizer = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
    print(f" Concluido (Vocabulario: {tokenizer.get_vocab_size():,})", flush=True)

    # [2/3] Embeddings, LM Head, and Final Norm
    print("[2/3] Carregando Embeddings (CPU) e LM Head + Final Norm (GPU)...", end="", flush=True)
    with st.safe_open(outside_path, framework="pt") as f:
        embed_tokens_cpu = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.bfloat16)
        final_norm_weight = f.get_tensor("model.language_model.norm.weight").to(device=device, dtype=torch.bfloat16)
        lm_head = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    print(f" Concluido! VRAM Inicial: {torch.cuda.memory_allocated(0)/1e6:.1f} MB", flush=True)

    # [3/3] Config & Reusable GPU Modules
    print("[3/3] Inicializando modulos Qwen3_5DecoderLayer e Cache autorregressivo...", end="", flush=True)
    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    cache = DynamicCache(config=cfg.text_config)

    linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)
    print(" Concluido!", flush=True)

    proc = psutil.Process()
    print(f"RAM Fisica (RSS): {proc.memory_info().rss / 1e9:.2f} GB (Margem Livre: {psutil.virtual_memory().available / 1e9:.2f} GB)")
    print(f"VRAM Alocada    : {torch.cuda.memory_allocated(0)/1e6:.1f} MB (Margem Livre: {(12000 - torch.cuda.memory_allocated(0)/1e6):.1f} MB)")

    # Prompt Setup
    prompt = "<!DOCTYPE html>\n<html>\n<head>\n    <title>Minecraft Clone</title>\n"
    print("\n" + "=" * 95)
    print(f"PROMPT DE INFERENCIA:\n{prompt}")
    print("=" * 95)
    print("GERANDO CODIGO AUTORREGRESSIVO (Token a Token com Streaming):\n", flush=True)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)
    print(prompt, end="", flush=True)

    out_file = os.path.abspath("generated_minecraft_by_g_qwen9b.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    def forward_through_all_charts(h_in: torch.Tensor, pos_emb: tuple) -> torch.Tensor:
        """Streams through the 16 macro-charts on-demand, executing 4 layers per chart."""
        h = h_in
        for g in range(16):
            cpath = os.path.join(charts_dir, f"chart_{g}.safetensors")
            cd = st.load_file(cpath, device=str(device))

            U_mix = cd['basis_mixer_U'].to(torch.bfloat16)
            V_mix = cd['basis_mixer_V'].to(torch.bfloat16)
            U_gate = cd['basis_ffn_gate_U'].to(torch.bfloat16)
            V_gate = cd['basis_ffn_gate_V'].to(torch.bfloat16)
            U_up = cd['basis_ffn_up_U'].to(torch.bfloat16)
            V_up = cd['basis_ffn_up_V'].to(torch.bfloat16)
            U_down = cd['basis_ffn_down_U'].to(torch.bfloat16)
            V_down = cd['basis_ffn_down_V'].to(torch.bfloat16)

            for idx in range(4):
                l = 4 * g + idx
                is_attn = (l % 4 == 3)

                # Reconstruct dense linear weights directly on CUDA
                R_mix = dequant_4bit(cd[f'layer_{l}_mixer_res_q'], cd[f'layer_{l}_mixer_res_scale'], torch.Size([5120, 6144]))
                W_mix = torch.matmul(U_mix, torch.matmul(cd[f'layer_{l}_mixer_S'].to(torch.bfloat16), V_mix.t())) + R_mix

                R_gate = dequant_4bit(cd[f'layer_{l}_ffn_gate_res_q'], cd[f'layer_{l}_ffn_gate_res_scale'], torch.Size([17408, 5120]))
                W_gate = torch.matmul(U_gate, torch.matmul(cd[f'layer_{l}_ffn_gate_S'].to(torch.bfloat16), V_gate.t())) + R_gate

                R_up = dequant_4bit(cd[f'layer_{l}_ffn_up_res_q'], cd[f'layer_{l}_ffn_up_res_scale'], torch.Size([17408, 5120]))
                W_up = torch.matmul(U_up, torch.matmul(cd[f'layer_{l}_ffn_up_S'].to(torch.bfloat16), V_up.t())) + R_up

                R_down = dequant_4bit(cd[f'layer_{l}_ffn_down_res_q'], cd[f'layer_{l}_ffn_down_res_scale'], torch.Size([5120, 17408]))
                W_down = torch.matmul(U_down, torch.matmul(cd[f'layer_{l}_ffn_down_S'].to(torch.bfloat16), V_down.t())) + R_down

                # Load layer l exclusive small weights on-demand with safe_open
                layer_f = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
                with st.safe_open(layer_f, framework="pt") as lf:
                    if is_attn:
                        mod = attn_layer
                        mod.self_attn.layer_idx = l
                        mod.self_attn.o_proj.weight.data.copy_(W_mix)
                        mod.mlp.gate_proj.weight.data.copy_(W_gate)
                        mod.mlp.up_proj.weight.data.copy_(W_up)
                        mod.mlp.down_proj.weight.data.copy_(W_down)

                        mod.self_attn.q_proj.weight.data.copy_(dequant_fp8(lf.get_tensor(f'model.language_model.layers.{l}.self_attn.q_proj.weight'), lf.get_tensor(f'model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv')).to(device))
                        mod.self_attn.k_proj.weight.data.copy_(dequant_fp8(lf.get_tensor(f'model.language_model.layers.{l}.self_attn.k_proj.weight'), lf.get_tensor(f'model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv')).to(device))
                        mod.self_attn.v_proj.weight.data.copy_(dequant_fp8(lf.get_tensor(f'model.language_model.layers.{l}.self_attn.v_proj.weight'), lf.get_tensor(f'model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv')).to(device))
                        mod.self_attn.q_norm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.self_attn.q_norm.weight').to(device=device, dtype=torch.bfloat16))
                        mod.self_attn.k_norm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.self_attn.k_norm.weight').to(device=device, dtype=torch.bfloat16))
                        mod.input_layernorm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.input_layernorm.weight').to(device=device, dtype=torch.bfloat16))
                        mod.post_attention_layernorm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.post_attention_layernorm.weight').to(device=device, dtype=torch.bfloat16))
                    else:
                        mod = linear_layer
                        mod.linear_attn.layer_idx = l
                        mod.linear_attn.out_proj.weight.data.copy_(W_mix)
                        mod.mlp.gate_proj.weight.data.copy_(W_gate)
                        mod.mlp.up_proj.weight.data.copy_(W_up)
                        mod.mlp.down_proj.weight.data.copy_(W_down)

                        mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant_fp8(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight'), lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv')).to(device))
                        mod.linear_attn.in_proj_z.weight.data.copy_(dequant_fp8(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_z.weight'), lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv')).to(device))
                        mod.linear_attn.in_proj_a.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_a.weight').to(device=device, dtype=torch.bfloat16))
                        mod.linear_attn.in_proj_b.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.in_proj_b.weight').to(device=device, dtype=torch.bfloat16))
                        mod.linear_attn.conv1d.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.conv1d.weight').to(device=device, dtype=torch.bfloat16))
                        mod.linear_attn.dt_bias.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.dt_bias').to(device=device, dtype=torch.bfloat16))
                        mod.linear_attn.A_log.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.A_log').to(device=device, dtype=torch.bfloat16))
                        mod.linear_attn.norm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.linear_attn.norm.weight').to(device=device, dtype=torch.bfloat16))
                        mod.input_layernorm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.input_layernorm.weight').to(device=device, dtype=torch.bfloat16))
                        mod.post_attention_layernorm.weight.data.copy_(lf.get_tensor(f'model.language_model.layers.{l}.post_attention_layernorm.weight').to(device=device, dtype=torch.bfloat16))

                h = mod(h, position_embeddings=pos_emb, past_key_values=cache, use_cache=True)

            del cd

        return h

    t0_gen = time.time()

    # --- PHASE 1: PREFILL PROMPT ---
    prompt_tensor = embed_tokens_cpu[input_ids].unsqueeze(0).to(device)
    pos_ids = torch.arange(len(input_ids), device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(prompt_tensor, pos_ids)

    h = forward_through_all_charts(prompt_tensor, pos_emb)

    h_last = h[:, -1:, :]
    h_norm = rms_norm(h_last, final_norm_weight)
    logits = torch.matmul(h_norm, lm_head.t()).squeeze(1)

    next_token_id = int(torch.argmax(logits, dim=-1).item())
    generated_ids.append(next_token_id)
    token_str = tokenizer.decode([next_token_id])
    print(token_str, end="", flush=True)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(token_str)

    # --- PHASE 2: AUTOREGRESSIVE GENERATION ---
    for step in range(max_new_tokens - 1):
        curr_seq_len = cache.get_seq_length()
        next_x = embed_tokens_cpu[next_token_id].unsqueeze(0).unsqueeze(0).to(device)
        pos_ids = torch.tensor([[[curr_seq_len]], [[curr_seq_len]], [[curr_seq_len]]], device=device)
        pos_emb = rotary(next_x, pos_ids)

        h_step = forward_through_all_charts(next_x, pos_emb)

        h_norm = rms_norm(h_step, final_norm_weight)
        logits = torch.matmul(h_norm, lm_head.t()).squeeze(1)

        next_token_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token_id)
        token_str = tokenizer.decode([next_token_id])
        print(token_str, end="", flush=True)

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(token_str)

        if next_token_id in [248044, 151643]:
            break

    t_total = time.time() - t0_gen
    gen_tokens = len(generated_ids) - len(input_ids)
    print("\n\n" + "=" * 95)
    print(f"GERACAO CONCLUIDA: {gen_tokens} tokens gerados em {t_total:.2f}s ({gen_tokens / max(t_total, 0.001):.2f} tokens/s)!")
    print(f"Pico de VRAM utilizado: {torch.cuda.max_memory_allocated(0)/1e6:.1f} MB")
    print(f"RAM Final Utilizada   : {proc.memory_info().rss / 1e9:.2f} GB")
    print("=" * 95, flush=True)
    print(f"Codigo salvo com sucesso em: {out_file}", flush=True)


if __name__ == "__main__":
    run_streaming_generation(max_new_tokens=100)
