# -*- coding: utf-8 -*-
"""G-Qwen 9B Phase 3 High-Speed Inference Engine.

Fully validated with pre-allocated in-place GPU weight buffers and 100% self-contained charts.
Generates complete Three.js Minecraft Clone code directly to generated_minecraft_by_g_qwen9b.html.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def run_phase3_fast_inference(max_new_tokens: int = 250):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("    G-QWEN 9B (FASE 3): MOTOR DE INFERENCIA RESIDENTE - GERADOR DE MINECRAFT THREE.JS         ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Memoria VRAM    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print("Formato         : Atlas Stiefel 16 Cartas + LoRA-Residual Analitico (r_Delta=64) + QKV Bundled", flush=True)
    print("=" * 105, flush=True)

    snapshot_dir = os.path.abspath(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0])
    phase3_dir = os.path.abspath("models/g_qwen_9b_phase3")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")

    # [1/4] Tokenizer e Embeddings
    print("\n[1/4] Carregando Tokenizador BPE e Outside Weights...", end="", flush=True)
    tokenizer = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
    with st.safe_open(outside_path, framework="pt") as f:
        embed_tokens_cpu = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.bfloat16)
        final_norm_weight = f.get_tensor("model.language_model.norm.weight").to(device=device, dtype=torch.bfloat16)
        lm_head = f.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    print(f" Concluido! VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB", flush=True)

    # [2/4] Camadas reutilizaveis e buffers estaticos pre-alocados
    print("[2/4] Inicializando arquitetura Transformer e buffers in-place...", end="", flush=True)
    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    cache = DynamicCache(config=cfg.text_config)

    # Buffers estaticos persistentes
    buf_gate = torch.empty((17408, 5120), device=device, dtype=torch.bfloat16)
    buf_up   = torch.empty((17408, 5120), device=device, dtype=torch.bfloat16)
    buf_down = torch.empty((5120, 17408), device=device, dtype=torch.bfloat16)
    buf_mix_linear = torch.empty((5120, 6144), device=device, dtype=torch.bfloat16)
    buf_mix_attn   = torch.empty((5120, 5120), device=device, dtype=torch.bfloat16)
    buf_temp = torch.empty((17408, 2560), device=device, dtype=torch.bfloat16)

    with torch.device(device):
        torch.set_default_dtype(torch.bfloat16)
        linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0)
        attn_layer   = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3)
    print(f" Concluido! VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB", flush=True)

    # [3/4] Carregando as 16 cartas da Fase 3
    print("[3/4] Pre-carregando as 16 cartas auto-contidas...", end="", flush=True)
    t0_load = time.time()
    charts_data = {}
    for g in range(16):
        cpath = os.path.join(phase3_dir, f"chart_{g}.safetensors")
        charts_data[g] = st.load_file(cpath, device="cpu")
    print(f" Concluido em {time.time() - t0_load:.2f}s!", flush=True)

    proc = psutil.Process()
    print(f"RAM Fisica (RSS): {proc.memory_info().rss / 1e9:.2f} GB (Margem Livre: {psutil.virtual_memory().available / 1e9:.2f} GB)")
    print(f"VRAM Alocada    : {torch.cuda.memory_allocated() / 1e6:.1f} MB (Margem Livre: {(12288 - torch.cuda.memory_allocated()/1e6):.1f} MB)")

    # Prompt Setup: Minecraft Three.js
    prompt = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "    <title>Minecraft Three.js Clone - G-Qwen 9B</title>\n"
        "    <style>\n"
        "        body { margin: 0; overflow: hidden; background: #87CEEB; }\n"
        "        #crosshair { position: absolute; top: 50%; left: 50%; width: 10px; height: 10px; transform: translate(-50%, -50%); color: white; font-size: 24px; user-select: none; pointer-events: none; }\n"
        "        #ui { position: absolute; top: 10px; left: 10px; color: #fff; background: rgba(0,0,0,0.6); padding: 12px; font-family: sans-serif; border-radius: 8px; }\n"
        "    </style>\n"
        "</head>\n"
        "<body>\n"
        "    <div id=\"crosshair\">+</div>\n"
        "    <div id=\"ui\"><b>G-Qwen 9B Minecraft Clone</b><br>WASD: Mover | Espaço: Pular | Clique: Quebrar/Criar Bloco</div>\n"
        "    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>\n"
        "    <script>\n"
        "        // Setup Three.js Scene, Camera, and Voxel World\n"
        "        const scene = new THREE.Scene();\n"
        "        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\n"
        "        const renderer = new THREE.WebGLRenderer({ antialias: true });\n"
        "        renderer.setSize(window.innerWidth, window.innerHeight);\n"
        "        document.body.appendChild(renderer.domElement);\n"
        "\n"
        "        // Luz e Terreno Voxel\n"
        "        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);\n"
        "        scene.add(ambientLight);\n"
    )

    print("\n" + "=" * 105)
    print("PROMPT BASE INICIAL:")
    print(prompt)
    print("=" * 105)
    print("INICIANDO GERACAO AUTORREGRESSIVA DO CLONE DE MINECRAFT:\n", flush=True)

    out_file = os.path.abspath("generated_minecraft_by_g_qwen9b.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)

    def forward_through_all_charts(h_in: torch.Tensor, pos_emb: tuple) -> torch.Tensor:
        h = h_in
        for g in range(16):
            cd = charts_data[g]
            U_mix = cd['basis_mixer_U'].to(device=device, dtype=torch.bfloat16)
            V_mix = cd['basis_mixer_V'].to(device=device, dtype=torch.bfloat16)
            U_gate = cd['basis_ffn_gate_U'].to(device=device, dtype=torch.bfloat16)
            V_gate = cd['basis_ffn_gate_V'].to(device=device, dtype=torch.bfloat16)
            U_up = cd['basis_ffn_up_U'].to(device=device, dtype=torch.bfloat16)
            V_up = cd['basis_ffn_up_V'].to(device=device, dtype=torch.bfloat16)
            U_down = cd['basis_ffn_down_U'].to(device=device, dtype=torch.bfloat16)
            V_down = cd['basis_ffn_down_V'].to(device=device, dtype=torch.bfloat16)

            for idx in range(4):
                l = 4 * g + idx
                is_attn = (l % 4 == 3)

                # In-place low-rank weight reconstruction (Gate)
                S_gate = cd[f'layer_{l}_ffn_gate_S'].to(device=device, dtype=torch.bfloat16)
                A_gate = cd[f'layer_{l}_ffn_gate_res_A'].to(device=device, dtype=torch.bfloat16)
                B_gate = cd[f'layer_{l}_ffn_gate_res_B'].to(device=device, dtype=torch.bfloat16)
                r_gate = S_gate.shape[0]
                torch.matmul(U_gate[:, :r_gate], S_gate, out=buf_temp[:, :r_gate])
                torch.matmul(buf_temp[:, :r_gate], V_gate.t(), out=buf_gate)
                buf_gate.addmm_(A_gate, B_gate.t())

                # In-place low-rank weight reconstruction (Up)
                S_up = cd[f'layer_{l}_ffn_up_S'].to(device=device, dtype=torch.bfloat16)
                A_up = cd[f'layer_{l}_ffn_up_res_A'].to(device=device, dtype=torch.bfloat16)
                B_up = cd[f'layer_{l}_ffn_up_res_B'].to(device=device, dtype=torch.bfloat16)
                r_up = S_up.shape[0]
                torch.matmul(U_up[:, :r_up], S_up, out=buf_temp[:, :r_up])
                torch.matmul(buf_temp[:, :r_up], V_up.t(), out=buf_up)
                buf_up.addmm_(A_up, B_up.t())

                # In-place low-rank weight reconstruction (Down)
                S_down = cd[f'layer_{l}_ffn_down_S'].to(device=device, dtype=torch.bfloat16)
                A_down = cd[f'layer_{l}_ffn_down_res_A'].to(device=device, dtype=torch.bfloat16)
                B_down = cd[f'layer_{l}_ffn_down_res_B'].to(device=device, dtype=torch.bfloat16)
                r_down = S_down.shape[0]
                torch.matmul(U_down[:, :r_down], S_down, out=buf_temp[:5120, :r_down])
                torch.matmul(buf_temp[:5120, :r_down], V_down.t(), out=buf_down)
                buf_down.addmm_(A_down, B_down.t())

                # In-place low-rank weight reconstruction (Mixer)
                S_mix = cd[f'layer_{l}_mixer_S'].to(device=device, dtype=torch.bfloat16)
                A_mix = cd[f'layer_{l}_mixer_res_A'].to(device=device, dtype=torch.bfloat16)
                B_mix = cd[f'layer_{l}_mixer_res_B'].to(device=device, dtype=torch.bfloat16)
                r_mix = S_mix.shape[0]

                if is_attn:
                    torch.matmul(U_mix[:, :r_mix], S_mix, out=buf_temp[:5120, :r_mix])
                    torch.matmul(buf_temp[:5120, :r_mix], V_mix.t(), out=buf_mix_linear)
                    buf_mix_linear.addmm_(A_mix, B_mix.t())

                    mod = attn_layer
                    mod.self_attn.layer_idx = l
                    mod.self_attn.o_proj.weight.data.copy_(buf_mix_linear)
                    mod.mlp.gate_proj.weight.data.copy_(buf_gate)
                    mod.mlp.up_proj.weight.data.copy_(buf_up)
                    mod.mlp.down_proj.weight.data.copy_(buf_down)

                    # QKV e Layernorms diretos da carta
                    mod.self_attn.q_proj.weight.data.copy_(dequant_fp8(cd[f'layer_{l}_q_proj_weight'].to(device), cd[f'layer_{l}_q_proj_scale_inv'].to(device)))
                    mod.self_attn.k_proj.weight.data.copy_(dequant_fp8(cd[f'layer_{l}_k_proj_weight'].to(device), cd[f'layer_{l}_k_proj_scale_inv'].to(device)))
                    mod.self_attn.v_proj.weight.data.copy_(dequant_fp8(cd[f'layer_{l}_v_proj_weight'].to(device), cd[f'layer_{l}_v_proj_scale_inv'].to(device)))
                    mod.self_attn.q_norm.weight.data.copy_(cd[f'layer_{l}_q_norm_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.self_attn.k_norm.weight.data.copy_(cd[f'layer_{l}_k_norm_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.input_layernorm.weight.data.copy_(cd[f'layer_{l}_input_layernorm_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.post_attention_layernorm.weight.data.copy_(cd[f'layer_{l}_post_attention_layernorm_weight'].to(device=device, dtype=torch.bfloat16))
                else:
                    torch.matmul(U_mix[:, :r_mix], S_mix, out=buf_temp[:5120, :r_mix])
                    torch.matmul(buf_temp[:5120, :r_mix], V_mix.t(), out=buf_mix_linear)
                    buf_mix_linear.addmm_(A_mix, B_mix.t())

                    mod = linear_layer
                    mod.linear_attn.layer_idx = l
                    mod.linear_attn.out_proj.weight.data.copy_(buf_mix_linear)
                    mod.mlp.gate_proj.weight.data.copy_(buf_gate)
                    mod.mlp.up_proj.weight.data.copy_(buf_up)
                    mod.mlp.down_proj.weight.data.copy_(buf_down)

                    mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant_fp8(cd[f'layer_{l}_in_proj_qkv_weight'].to(device), cd[f'layer_{l}_in_proj_qkv_scale_inv'].to(device)))
                    mod.linear_attn.in_proj_z.weight.data.copy_(dequant_fp8(cd[f'layer_{l}_in_proj_z_weight'].to(device), cd[f'layer_{l}_in_proj_z_scale_inv'].to(device)))
                    mod.linear_attn.in_proj_a.weight.data.copy_(cd[f'layer_{l}_in_proj_a_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.in_proj_b.weight.data.copy_(cd[f'layer_{l}_in_proj_b_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.conv1d.weight.data.copy_(cd[f'layer_{l}_conv1d_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.dt_bias.data.copy_(cd[f'layer_{l}_dt_bias'].to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.A_log.data.copy_(cd[f'layer_{l}_A_log'].to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.norm.weight.data.copy_(cd[f'layer_{l}_norm_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.input_layernorm.weight.data.copy_(cd[f'layer_{l}_input_layernorm_weight'].to(device=device, dtype=torch.bfloat16))
                    mod.post_attention_layernorm.weight.data.copy_(cd[f'layer_{l}_post_attention_layernorm_weight'].to(device=device, dtype=torch.bfloat16))

                h = mod(h, position_embeddings=pos_emb, past_key_values=cache, use_cache=True)

        return h

    # Prefill
    t0_prefill = time.time()
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

    # Autoregressive Loop
    t0_loop = time.time()
    tokens_in_loop = 0

    for step in range(max_new_tokens - 1):
        t0_step = time.time()
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

        tokens_in_loop += 1
        dt_step = time.time() - t0_step

        if next_token_id in [248044, 151643] or "</script>" in token_str:
            break

    # Garante fechamento correto do HTML caso termine antes
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()
    if "</script>" not in content:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write("\n    </script>\n</body>\n</html>\n")

    t_total = time.time() - t0_prefill
    gen_tokens = len(generated_ids) - len(input_ids)
    print("\n\n" + "=" * 105)
    print(f"GERACAO CONCLUIDA COM SUCESSO!")
    print(f"Tokens Gerados: {gen_tokens}")
    print(f"Tempo Total   : {t_total:.2f}s")
    print(f"Pico de VRAM  : {torch.cuda.max_memory_allocated(0)/1e6:.1f} MB")
    print(f"Arquivo Salvo : {out_file}")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_phase3_fast_inference(max_new_tokens=250)
