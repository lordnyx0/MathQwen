"""Benchmark of 1 Macro-Chart (4 layers) using official Qwen3_5DecoderLayer."""
import os
import glob
import time
import torch
import safetensors.torch as st
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding
)

def fast_dequantize_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size, block_size: int = 128) -> torch.Tensor:
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, block_size)
    return (unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape)

def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    s_exp = scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.bfloat16)

def test_chart_0():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    cfg = AutoConfig.from_pretrained(snapshot_dir)

    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)

    # Instantiate 4 layers
    print("Instantiating 4 Qwen3_5DecoderLayers...")
    layers = []
    for i in range(4):
        layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=i).to(device=device, dtype=torch.bfloat16)
        layers.append(layer)
    print(f"GPU VRAM after 4 layers: {torch.cuda.memory_allocated(0)/1e6:.1f} MB")

    # Load Chart 0
    t0 = time.time()
    cpath = "models/g_qwen_9b_phase1/chart_0.safetensors"
    cd = st.load_file(cpath)
    print(f"Loaded chart_0 in {time.time() - t0:.3f}s")

    # Reconstruct weights for layer 0
    t_rec0 = time.time()
    U_mix = cd["basis_mixer_U"].to(device=device, dtype=torch.bfloat16)
    V_mix = cd["basis_mixer_V"].to(device=device, dtype=torch.bfloat16)
    S_mix = cd["layer_0_mixer_S"].to(device=device, dtype=torch.bfloat16)
    q_mix = cd["layer_0_mixer_res_q"].to(device=device)
    s_mix = cd["layer_0_mixer_res_scale"].to(device=device)
    R_mix = fast_dequantize_4bit(q_mix, s_mix, torch.Size([5120, 6144]))
    W_mix_0 = torch.matmul(U_mix, torch.matmul(S_mix, V_mix.t())) + R_mix
    print(f"Reconstructed W_mix_0 in {time.time() - t_rec0:.4f}s, shape: {W_mix_0.shape}")

    # Dummy forward
    x = torch.randn(1, 1, 5120, device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(1, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(x, pos_ids)

    # Assign W_mix to layer 0
    layers[0].linear_attn.out_proj.weight.data.copy_(W_mix_0)

    t_fwd = time.time()
    h = x
    for i in range(4):
        h = layers[i](h, position_embeddings=pos_emb)
    torch.cuda.synchronize()
    print(f"Forward through 4 layers took: {(time.time() - t_fwd)*1000:.2f} ms")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(0)/1e6:.1f} MB")

if __name__ == "__main__":
    test_chart_0()
