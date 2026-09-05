"""Streaming Macro-Chart Test: Zero-RAM-Thrashing G-Qwen 9B Forward Pass."""
import os
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

def dequant_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, 128)
    return (unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape)

def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.bfloat16)

def test_streaming():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    charts_dir = "models/g_qwen_9b_phase1"

    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    cache = DynamicCache(config=cfg.text_config)

    linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)

    # Pre-test Chart 0 (4 layers)
    t0 = time.time()
    cpath = os.path.join(charts_dir, "chart_0.safetensors")
    cd = st.load_file(cpath, device=str(device))

    U_mix = cd['basis_mixer_U'].to(torch.bfloat16)
    V_mix = cd['basis_mixer_V'].to(torch.bfloat16)
    U_gate = cd['basis_ffn_gate_U'].to(torch.bfloat16)
    V_gate = cd['basis_ffn_gate_V'].to(torch.bfloat16)
    U_up = cd['basis_ffn_up_U'].to(torch.bfloat16)
    V_up = cd['basis_ffn_up_V'].to(torch.bfloat16)
    U_down = cd['basis_ffn_down_U'].to(torch.bfloat16)
    V_down = cd['basis_ffn_down_V'].to(torch.bfloat16)

    x = torch.randn(1, 5, 5120, device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(5, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(x, pos_ids)

    h = x
    for idx in range(4):
        l = idx
        is_attn = (l % 4 == 3)
        R_mix = dequant_4bit(cd[f'layer_{l}_mixer_res_q'], cd[f'layer_{l}_mixer_res_scale'], torch.Size([5120, 6144]))
        W_mix = torch.matmul(U_mix, torch.matmul(cd[f'layer_{l}_mixer_S'].to(torch.bfloat16), V_mix.t())) + R_mix

        R_gate = dequant_4bit(cd[f'layer_{l}_ffn_gate_res_q'], cd[f'layer_{l}_ffn_gate_res_scale'], torch.Size([17408, 5120]))
        W_gate = torch.matmul(U_gate, torch.matmul(cd[f'layer_{l}_ffn_gate_S'].to(torch.bfloat16), V_gate.t())) + R_gate

        R_up = dequant_4bit(cd[f'layer_{l}_ffn_up_res_q'], cd[f'layer_{l}_ffn_up_res_scale'], torch.Size([17408, 5120]))
        W_up = torch.matmul(U_up, torch.matmul(cd[f'layer_{l}_ffn_up_S'].to(torch.bfloat16), V_up.t())) + R_up

        R_down = dequant_4bit(cd[f'layer_{l}_ffn_down_res_q'], cd[f'layer_{l}_ffn_down_res_scale'], torch.Size([5120, 17408]))
        W_down = torch.matmul(U_down, torch.matmul(cd[f'layer_{l}_ffn_down_S'].to(torch.bfloat16), V_down.t())) + R_down

        # Load layer l exclusive
        layer_f = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
        ld = st.load_file(layer_f)

        if is_attn:
            mod = attn_layer
            mod.self_attn.layer_idx = l
            mod.self_attn.o_proj.weight.data.copy_(W_mix)
            mod.mlp.gate_proj.weight.data.copy_(W_gate)
            mod.mlp.up_proj.weight.data.copy_(W_up)
            mod.mlp.down_proj.weight.data.copy_(W_down)

            mod.self_attn.q_proj.weight.data.copy_(dequant_fp8(ld[f'model.language_model.layers.{l}.self_attn.q_proj.weight'], ld[f'model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv']).to(device))
            mod.self_attn.k_proj.weight.data.copy_(dequant_fp8(ld[f'model.language_model.layers.{l}.self_attn.k_proj.weight'], ld[f'model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv']).to(device))
            mod.self_attn.v_proj.weight.data.copy_(dequant_fp8(ld[f'model.language_model.layers.{l}.self_attn.v_proj.weight'], ld[f'model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv']).to(device))
            mod.self_attn.q_norm.weight.data.copy_(ld[f'model.language_model.layers.{l}.self_attn.q_norm.weight'].to(device=device, dtype=torch.bfloat16))
            mod.self_attn.k_norm.weight.data.copy_(ld[f'model.language_model.layers.{l}.self_attn.k_norm.weight'].to(device=device, dtype=torch.bfloat16))
            mod.input_layernorm.weight.data.copy_(ld[f'model.language_model.layers.{l}.input_layernorm.weight'].to(device=device, dtype=torch.bfloat16))
            mod.post_attention_layernorm.weight.data.copy_(ld[f'model.language_model.layers.{l}.post_attention_layernorm.weight'].to(device=device, dtype=torch.bfloat16))
        else:
            mod = linear_layer
            mod.linear_attn.layer_idx = l
            mod.linear_attn.out_proj.weight.data.copy_(W_mix)
            mod.mlp.gate_proj.weight.data.copy_(W_gate)
            mod.mlp.up_proj.weight.data.copy_(W_up)
            mod.mlp.down_proj.weight.data.copy_(W_down)

            mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant_fp8(ld[f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight'], ld[f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv']).to(device))
            mod.linear_attn.in_proj_z.weight.data.copy_(dequant_fp8(ld[f'model.language_model.layers.{l}.linear_attn.in_proj_z.weight'], ld[f'model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv']).to(device))
            mod.linear_attn.in_proj_a.weight.data.copy_(ld[f'model.language_model.layers.{l}.linear_attn.in_proj_a.weight'].to(device=device, dtype=torch.bfloat16))
            mod.linear_attn.in_proj_b.weight.data.copy_(ld[f'model.language_model.layers.{l}.linear_attn.in_proj_b.weight'].to(device=device, dtype=torch.bfloat16))
            mod.linear_attn.conv1d.weight.data.copy_(ld[f'model.language_model.layers.{l}.linear_attn.conv1d.weight'].to(device=device, dtype=torch.bfloat16))
            mod.linear_attn.dt_bias.data.copy_(ld[f'model.language_model.layers.{l}.linear_attn.dt_bias'].to(device=device, dtype=torch.bfloat16))
            mod.linear_attn.A_log.data.copy_(ld[f'model.language_model.layers.{l}.linear_attn.A_log'].to(device=device, dtype=torch.bfloat16))
            mod.linear_attn.norm.weight.data.copy_(ld[f'model.language_model.layers.{l}.linear_attn.norm.weight'].to(device=device, dtype=torch.bfloat16))
            mod.input_layernorm.weight.data.copy_(ld[f'model.language_model.layers.{l}.input_layernorm.weight'].to(device=device, dtype=torch.bfloat16))
            mod.post_attention_layernorm.weight.data.copy_(ld[f'model.language_model.layers.{l}.post_attention_layernorm.weight'].to(device=device, dtype=torch.bfloat16))

        h = mod(h, position_embeddings=pos_emb, past_key_values=cache, use_cache=True)
        del ld

    del cd
    torch.cuda.synchronize()
    proc = psutil.Process()
    print(f"Chart 0 (4 layers) streamed and executed in {time.time()-t0:.2f}s!")
    print(f"Physical RAM (RSS): {proc.memory_info().rss / 1e9:.2f} GB")
    print(f"Virtual RAM  (VMS): {proc.memory_info().vms / 1e9:.2f} GB")
    print(f"GPU VRAM Allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")

if __name__ == "__main__":
    test_streaming()
