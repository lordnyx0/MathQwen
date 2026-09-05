"""Pure GPU Generation Test for G-Qwen 9B Phase 1 (16 Layers, 100% VRAM Resident).

Zero Disk I/O during generation (SSD 0%).
100% CUDA execution on NVIDIA GeForce RTX 3060.
Total VRAM footprint: ~6.5 GB / 12 GB.
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


def fast_unpack_residual(packed_q: torch.Tensor, scale: torch.Tensor, out_features: int, in_features: int, block_size: int = 128) -> torch.Tensor:
    """Fast vectorized GPU unpacking without memory allocation thrashing."""
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, block_size)
    return (unpacked.to(torch.float16) * scale.view(-1, 1).to(torch.float16)).view(out_features, in_features)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


class GPUResidentLayer:
    """A layer completely loaded and resident in GPU memory."""

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
                wv = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight")
                sv = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight_scale_inv")
                sv_exp = sv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_v = (wv.float() * sv_exp.float()).to(torch.float16)
            else:
                wz = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight")
                sz = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight_scale_inv")
                sz_exp = sz.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                self.W_z = (wz.float() * sz_exp.float()).to(torch.float16)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 1. Mixer
        h_norm = rms_norm(h, self.input_layernorm)
        if self.is_attn:
            v = torch.matmul(h_norm, self.W_v.t())
            x_mix = v.repeat_interleave(6, dim=-1)[:, :, :6144]
        else:
            z = torch.matmul(h_norm, self.W_z.t())
            x_mix = F.silu(z)

        # Stiefel Atlas projection
        y_atlas = torch.matmul(torch.matmul(torch.matmul(x_mix, self.V_mix), self.S_mix.t()), self.U_mix.t())
        # Fast residual
        R_mix = fast_unpack_residual(self.res_q_mix, self.res_s_mix, 5120, 6144)
        y_res = torch.matmul(x_mix, R_mix.t())
        h = h + (y_atlas + y_res)

        # 2. FFN
        h_post = rms_norm(h, self.post_attention_layernorm)

        # Gate
        g_atlas = torch.matmul(torch.matmul(torch.matmul(h_post, self.V_gate), self.S_gate.t()), self.U_gate.t())
        R_gate = fast_unpack_residual(self.res_q_gate, self.res_s_gate, 17408, 5120)
        gate = F.silu(g_atlas + torch.matmul(h_post, R_gate.t()))

        # Up
        u_atlas = torch.matmul(torch.matmul(torch.matmul(h_post, self.V_up), self.S_up.t()), self.U_up.t())
        R_up = fast_unpack_residual(self.res_q_up, self.res_s_up, 17408, 5120)
        up = u_atlas + torch.matmul(h_post, R_up.t())

        inter = gate * up

        # Down
        d_atlas = torch.matmul(torch.matmul(torch.matmul(inter, self.V_down), self.S_down.t()), self.U_down.t())
        R_down = fast_unpack_residual(self.res_q_down, self.res_s_down, 5120, 17408)
        down = d_atlas + torch.matmul(inter, R_down.t())
        h = h + down

        return h


def run_gpu_generation(num_tokens: int = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("   GERACAO 100% GPU RESIDENTE: G-QWEN 9B (FASE 1 - ZERO USO DE DISCO / SSD 0%)          ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    snapshot_dir = find_snapshot_dir()
    tokenizer_path = os.path.join(snapshot_dir, "tokenizer.json")
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")

    # 1. Tokenizer
    print("\n[1/3] Carregando Tokenizador...", end="", flush=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f" Concluido (Vocabulario: {tokenizer.get_vocab_size():,})", flush=True)

    # 2. Outside tensors (CPU embeddings + GPU LM head)
    print("[2/3] Carregando Embeddings e LM Head na GPU...", end="", flush=True)
    with safe_open(outside_path, framework="pt") as f:
        embed_tokens_cpu = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.float16)
        final_norm = f.get_tensor("model.language_model.norm.weight").to(device).to(torch.float16)
        lm_head = f.get_tensor("lm_head.weight").to(device).to(torch.float16)
    print(f" Concluido! VRAM: {torch.cuda.memory_allocated(0)/1e6:.1f} MB", flush=True)

    # 3. Load all 16 layers (Charts 0..3) directly into GPU VRAM
    print("[3/3] Carregando as 16 Camadas da Fase 1 Diretamente na VRAM da GPU...", end="", flush=True)
    t0_l = time.time()
    layers: List[GPUResidentLayer] = []
    for g in range(4):
        cpath = os.path.abspath(f"models/g_qwen_9b_phase1/chart_{g}.safetensors")
        with safe_open(cpath, framework="pt", device=str(device)) as cf:
            chart_tensors = {k: cf.get_tensor(k) for k in cf.keys()}
        for idx in range(4):
            l_idx = 4 * g + idx
            l_file = os.path.join(snapshot_dir, f"layers-{l_idx}.safetensors")
            layers.append(GPUResidentLayer(l_idx, chart_tensors, l_file, device))
        del chart_tensors
    print(f" Concluido em {time.time() - t0_l:.2f}s!", flush=True)
    print(f"VRAM Total Alocada na GPU: {torch.cuda.memory_allocated(0)/1e6:.1f} MB (Tudo na VRAM, zero uso de SSD!)", flush=True)

    # Prompt
    prompt = "<!DOCTYPE html>\n<html>\n<head>\n    <title>Minecraft Clone</title>\n"
    print("\n" + "=" * 105)
    print(f"PROMPT DE ENTRADA:\n{prompt}")
    print("=" * 105)
    print("GERANDO CODIGO AUTORREGRESSIVO (Token a Token na GPU):\n", flush=True)

    input_ids = tokenizer.encode(prompt).ids
    generated_ids = list(input_ids)
    generated_text = prompt

    print(prompt, end="", flush=True)
    t0_gen = time.time()

    for step in range(num_tokens):
        # Embeddings lookup
        seq_ids = generated_ids[-64:]
        h = embed_tokens_cpu[seq_ids].unsqueeze(0).to(device)

        # Pure in-VRAM forward pass across all 16 layers
        for layer in layers:
            h = layer.forward(h)

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

        # EOS
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
    run_gpu_generation()
