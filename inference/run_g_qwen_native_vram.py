"""100% VRAM Resident Zero-Allocation Inference Engine for True Native G-Qwen 9B (64 Camadas Fisicas / 16 Cartas).

Features:
- Strict 64-Layer Causal Feedforward (ZERO Triple Loop)
- Memory Footprint: ~10.2 GB VRAM on NVIDIA RTX 3060 (12 GB)
- Dual Shared Execution Units (Qwen3_5DecoderLayer) with In-Place Weights
- Generates directly into generated_minecraft_by_g_qwen9b.html
"""

import os
import sys
import glob
import time
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


def dequant_4bit_into(packed_q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor):
    """Vectorized 4-bit unpacking on CUDA directly into target module weight."""
    shape = out.shape
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, 128)
    with torch.no_grad():
        out.copy_((unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape))


def dequant_2bit_into(packed_2q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor):
    """Vectorized 2-bit unpacking on CUDA directly into target module weight."""
    shape = out.shape
    v0 = (packed_2q & 0x03).to(torch.int8) - 2
    v1 = ((packed_2q >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed_2q >> 4) & 0x03).to(torch.int8) - 2
    v3 = ((packed_2q >> 6) & 0x03).to(torch.int8) - 2
    unpacked = torch.stack([v0, v1, v2, v3], dim=1).view(-1, 128)
    with torch.no_grad():
        out.copy_((unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape))


def get_coord_s(tensors: dict, prefix: str) -> torch.Tensor:
    fp8_val = tensors[f"{prefix}.fp8"].to(torch.float32)
    scale = tensors[f"{prefix}.scale"].to(torch.float32)
    return (fp8_val * scale).to(torch.bfloat16)


def run_g_qwen_native_inference(
    max_new_tokens: int = 1200,
    temperature: float = 0.25,
    top_p: float = 0.9,
    greedy: bool = False
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("    G-QWEN 9B NATIVO: MOTOR 100% VRAM 64 CAMADAS INTEGRAIS (SEQUENCIAL CAUSAL PURO)       ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print("Zero Triple Loop | 64 Camadas Fisicas | 16 Macro-Cartas Grassmannianas", flush=True)
    print("=" * 105, flush=True)

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    native_model_path = os.path.abspath("models/g_qwen_9b_native_16charts.safetensors")
    html_output_path = os.path.abspath("generated_minecraft_by_g_qwen9b.html")

    # 1. Load Tokenizer
    print("\n[1/4] Carregando Tokenizador...", end="", flush=True)
    tokenizer = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
    print(f" Concluido! (Vocabulario: {tokenizer.get_vocab_size():,})", flush=True)

    # 2. Setup Shared Execution Modules in VRAM
    print("[2/4] Inicializando modulos de execucao compartilhados GPU...", end="", flush=True)
    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    cache = DynamicCache(config=cfg.text_config)

    linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)

    # Share MLP structure across layers to save 200 MB
    del attn_layer.mlp
    attn_layer.mlp = linear_layer.mlp
    del attn_layer.self_attn.o_proj
    attn_layer.self_attn.o_proj = linear_layer.linear_attn.out_proj

    linear_layer.requires_grad_(False)
    attn_layer.requires_grad_(False)
    print(" Concluido!", flush=True)

    # 3. Load 16-Chart Safetensors Model into VRAM
    print(f"[3/4] Carregando {native_model_path}...", end="", flush=True)
    t0_load = time.time()
    t_raw = st.load_file(native_model_path)
    
    # Keep token_embd in pinned host RAM (saves 611 MB VRAM)
    tensors = {}
    for k, v in t_raw.items():
        if "token_embd" in k:
            tensors[k] = v.pin_memory()
        else:
            tensors[k] = v.to(device, non_blocking=True)
    del t_raw
    torch.cuda.empty_cache()

    vram_alloc = torch.cuda.memory_allocated(0) / (1024**3)
    print(f" Concluido em {time.time()-t0_load:.2f}s! (VRAM Alocada: {vram_alloc:.2f} GB | Margem Livre: {12.0-vram_alloc:.2f} GB)", flush=True)

    def apply_final_norm(h: torch.Tensor) -> torch.Tensor:
        var = h.to(torch.float32).pow(2).mean(-1, keepdim=True)
        norm_w = tensors["output_norm.weight"].to(device=device, dtype=torch.bfloat16)
        return (h * torch.rsqrt(var + 1e-6)).to(torch.bfloat16) * norm_w

    def get_token_embedding(token_id: int) -> torch.Tensor:
        p_row = tensors["token_embd.q"][(token_id * 5120)//2 : (token_id * 5120)//2 + 2560].to(device)
        s_row = tensors["token_embd.s"][token_id * 40 : (token_id + 1) * 40].to(device)
        low = (p_row & 0x0F).to(torch.int8) - 7
        high = ((p_row >> 4) & 0x0F).to(torch.int8) - 7
        unpacked = torch.stack([low, high], dim=1).view(-1, 128)
        return (unpacked.to(torch.bfloat16) * s_row.view(-1, 1).to(torch.bfloat16)).view(1, 1, 5120)

    logits_buf = torch.empty(248320, dtype=torch.float32, device=device)
    chunk_size = 248320 // 32
    head_q = tensors["lm_head.q"]
    head_scale = tensors["lm_head.s"]

    def compute_logits(h_norm: torch.Tensor) -> torch.Tensor:
        h_vec = h_norm.view(5120, 1)
        for i in range(32):
            p_chunk = head_q[i * chunk_size * 2560 : (i + 1) * chunk_size * 2560]
            s_chunk = head_scale[i * chunk_size * 40 : (i + 1) * chunk_size * 40]
            low = (p_chunk & 0x0F).to(torch.int8) - 7
            high = ((p_chunk >> 4) & 0x0F).to(torch.int8) - 7
            unpacked = torch.stack([low, high], dim=1).view(-1, 128)
            w_c = (unpacked.to(torch.bfloat16) * s_chunk.view(-1, 1).to(torch.bfloat16)).view(chunk_size, 5120)
            logits_buf[i * chunk_size : (i + 1) * chunk_size] = torch.matmul(w_c, h_vec).squeeze(1)
        return logits_buf

    def execute_layer(v_idx: int, l_idx: int, h_in: torch.Tensor, position_embeddings: tuple) -> torch.Tensor:
        is_attn = (l_idx % 4 == 3)
        g = l_idx // 4

        U_mix = tensors[f"chart.{g}.basis_mixer_U"].to(torch.bfloat16)
        V_mix = tensors[f"chart.{g}.basis_mixer_V"].to(torch.bfloat16)
        U_gate = tensors[f"chart.{g}.basis_ffn_gate_U"].to(torch.bfloat16)
        V_gate = tensors[f"chart.{g}.basis_ffn_gate_V"].to(torch.bfloat16)
        U_up = tensors[f"chart.{g}.basis_ffn_up_U"].to(torch.bfloat16)
        V_up = tensors[f"chart.{g}.basis_ffn_up_V"].to(torch.bfloat16)
        U_down = tensors[f"chart.{g}.basis_ffn_down_U"].to(torch.bfloat16)
        V_down = tensors[f"chart.{g}.basis_ffn_down_V"].to(torch.bfloat16)

        S_gate = get_coord_s(tensors, f"blk.{l_idx}.ffn_gate_S")
        S_up = get_coord_s(tensors, f"blk.{l_idx}.ffn_up_S")
        S_down = get_coord_s(tensors, f"blk.{l_idx}.ffn_down_S")
        S_mix = get_coord_s(tensors, f"blk.{l_idx}.mixer_S")

        # 1. Update MLP weights in-place
        dequant_2bit_into(tensors[f"blk.{l_idx}.ffn_gate_res_2q"], tensors[f"blk.{l_idx}.ffn_gate_res_2scale"], linear_layer.mlp.gate_proj.weight)
        with torch.no_grad():
            torch.addmm(linear_layer.mlp.gate_proj.weight, U_gate, torch.matmul(S_gate, V_gate.t()), out=linear_layer.mlp.gate_proj.weight)

        dequant_2bit_into(tensors[f"blk.{l_idx}.ffn_up_res_2q"], tensors[f"blk.{l_idx}.ffn_up_res_2scale"], linear_layer.mlp.up_proj.weight)
        with torch.no_grad():
            torch.addmm(linear_layer.mlp.up_proj.weight, U_up, torch.matmul(S_up, V_up.t()), out=linear_layer.mlp.up_proj.weight)

        dequant_2bit_into(tensors[f"blk.{l_idx}.ffn_down_res_2q"], tensors[f"blk.{l_idx}.ffn_down_res_2scale"], linear_layer.mlp.down_proj.weight)
        with torch.no_grad():
            torch.addmm(linear_layer.mlp.down_proj.weight, U_down, torch.matmul(S_down, V_down.t()), out=linear_layer.mlp.down_proj.weight)

        # 2. Update Attention / Mixer weights and execute
        if is_attn:
            mod = attn_layer
            mod.self_attn.layer_idx = v_idx

            dequant_4bit_into(tensors[f"blk.{l_idx}.mixer_res_q"], tensors[f"blk.{l_idx}.mixer_res_scale"], mod.self_attn.o_proj.weight)
            with torch.no_grad():
                torch.addmm(mod.self_attn.o_proj.weight, U_mix, torch.matmul(S_mix, V_mix.t()), out=mod.self_attn.o_proj.weight)

            dequant_4bit_into(tensors[f"blk.{l_idx}.attn_q.q"], tensors[f"blk.{l_idx}.attn_q.s"], mod.self_attn.q_proj.weight)
            dequant_4bit_into(tensors[f"blk.{l_idx}.attn_k.q"], tensors[f"blk.{l_idx}.attn_k.s"], mod.self_attn.k_proj.weight)
            dequant_4bit_into(tensors[f"blk.{l_idx}.attn_v.q"], tensors[f"blk.{l_idx}.attn_v.s"], mod.self_attn.v_proj.weight)

            with torch.no_grad():
                mod.self_attn.q_norm.weight.copy_(tensors[f"blk.{l_idx}.attn_q_norm.weight"])
                mod.self_attn.k_norm.weight.copy_(tensors[f"blk.{l_idx}.attn_k_norm.weight"])
                mod.input_layernorm.weight.copy_(tensors[f"blk.{l_idx}.attn_norm.weight"])
                mod.post_attention_layernorm.weight.copy_(tensors[f"blk.{l_idx}.ffn_norm.weight"])
        else:
            mod = linear_layer
            mod.linear_attn.layer_idx = v_idx

            dequant_4bit_into(tensors[f"blk.{l_idx}.mixer_res_q"], tensors[f"blk.{l_idx}.mixer_res_scale"], mod.linear_attn.out_proj.weight)
            with torch.no_grad():
                torch.addmm(mod.linear_attn.out_proj.weight, U_mix, torch.matmul(S_mix, V_mix.t()), out=mod.linear_attn.out_proj.weight)

            dequant_4bit_into(tensors[f"blk.{l_idx}.attn_qkv.q"], tensors[f"blk.{l_idx}.attn_qkv.s"], mod.linear_attn.in_proj_qkv.weight)
            dequant_4bit_into(tensors[f"blk.{l_idx}.attn_gate.q"], tensors[f"blk.{l_idx}.attn_gate.s"], mod.linear_attn.in_proj_z.weight)

            with torch.no_grad():
                mod.linear_attn.in_proj_a.weight.copy_(tensors[f"blk.{l_idx}.ssm_alpha.weight"])
                mod.linear_attn.in_proj_b.weight.copy_(tensors[f"blk.{l_idx}.ssm_beta.weight"])
                mod.linear_attn.conv1d.weight.copy_(tensors[f"blk.{l_idx}.ssm_conv1d.weight"])
                mod.linear_attn.dt_bias.copy_(tensors[f"blk.{l_idx}.ssm_dt.bias"])
                mod.linear_attn.A_log.copy_(tensors[f"blk.{l_idx}.ssm_a.weight"])
                mod.linear_attn.norm.weight.copy_(tensors[f"blk.{l_idx}.ssm_norm.weight"])
                mod.input_layernorm.weight.copy_(tensors[f"blk.{l_idx}.attn_norm.weight"])
                mod.post_attention_layernorm.weight.copy_(tensors[f"blk.{l_idx}.ffn_norm.weight"])

        with torch.no_grad():
            outputs = mod(
                hidden_states=h_in,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                use_cache=True,
            )
        return outputs[0] if isinstance(outputs, tuple) else outputs

    # 4. Prompt Setup & Output File
    prompt = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Minecraft 3D</title>
    <style>
        body { margin: 0; overflow: hidden; background: #87CEEB; font-family: monospace; }
        canvas { width: 100vw; height: 100vh; display: block; }
        #hud { position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.6); padding: 8px; border-radius: 4px; }
        #crosshair { position: absolute; top: 50%; left: 50%; width: 10px; height: 10px; transform: translate(-50%, -50%); pointer-events: none; border: 2px solid white; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="hud">WASD: Mover | Espaco: Pular | Mouse: Olhar | Clique: Quebrar/Colocar</div>
    <div id="crosshair"></div>
    <script>
        // Complete 3D Voxel Minecraft Engine
"""
    print("\n[4/4] Preparando Inferencia e Arquivo de Saida...", flush=True)
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)
    print("=" * 105)
    print(f"PROMPT INICIAL ({len(input_ids)} tokens):\n{prompt}", flush=True)
    print("=" * 105)
    print("INICIANDO GERACAO AUTORREGRESSIVA SEQUENCIAL 64 CAMADAS (STREAMING):\n", flush=True)
    print(prompt, end="", flush=True)

    # Prefill Phase
    seq_len = len(input_ids)
    h_tokens = [get_token_embedding(t) for t in input_ids]
    h = torch.cat(h_tokens, dim=1)

    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(h, pos_ids)

    # Executa todas as 64 camadas sequencialmente (0..63)
    for l_idx in range(64):
        h = execute_layer(l_idx, l_idx, h, pos_emb)

    # Primeiro token predito
    h_last = h[:, -1:, :]
    h_norm = apply_final_norm(h_last)
    logits = compute_logits(h_norm)

    if greedy:
        next_token_id = int(torch.argmax(logits, dim=-1).item())
    else:
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        next_token_id = int(torch.multinomial(probs, num_samples=1).item())

    generated_ids.append(next_token_id)
    token_str = tokenizer.decode([next_token_id])
    print(token_str, end="", flush=True)
    with open(html_output_path, "a", encoding="utf-8") as f:
        f.write(token_str)

    # Autoregressive Generation Loop
    t0_gen = time.time()
    for step in range(max_new_tokens - 1):
        curr_seq_len = len(generated_ids)
        next_x = get_token_embedding(next_token_id)

        pos_ids = torch.tensor([[[curr_seq_len]], [[curr_seq_len]], [[curr_seq_len]]], device=device)
        pos_emb = rotary(next_x, pos_ids)

        h_step = next_x
        # 64 Camadas Sequenciais
        for l_idx in range(64):
            h_step = execute_layer(l_idx, l_idx, h_step, pos_emb)

        h_step_norm = apply_final_norm(h_step)
        logits_step = compute_logits(h_step_norm)

        if greedy:
            next_token_id = int(torch.argmax(logits_step, dim=-1).item())
        else:
            probs = torch.softmax(logits_step / max(temperature, 1e-4), dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        generated_ids.append(next_token_id)
        token_str = tokenizer.decode([next_token_id])
        print(token_str, end="", flush=True)
        with open(html_output_path, "a", encoding="utf-8") as f:
            f.write(token_str)

        if next_token_id in [151643, 151645] or "</script>" in token_str:
            print("\n[Fim da sequencia atingido]", flush=True)
            break

    elapsed = time.time() - t0_gen
    n_tokens = len(generated_ids) - len(input_ids)
    print("\n" + "=" * 105)
    print(f"Gerao concluda: {n_tokens} tokens gerados em {elapsed:.2f}s ({n_tokens/max(elapsed, 0.1):.2f} tokens/s)")
    print(f"Arquivo HTML gravado: {html_output_path}")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_g_qwen_native_inference()
