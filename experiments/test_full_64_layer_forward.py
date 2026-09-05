"""Test end-to-end forward pass of all 64 layers using Qwen3_5DecoderLayer."""
import os
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

def test_full_pipeline():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    cfg = AutoConfig.from_pretrained(snapshot_dir)

    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)
    cache = DynamicCache(config=cfg.text_config)

    linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)

    print("Modules initialized on GPU!")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0)/1e6:.1f} MB")

    # Dummy prompt of 10 tokens
    x = torch.randn(1, 10, 5120, device=device, dtype=torch.bfloat16)
    pos_ids = torch.arange(10, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos_emb = rotary(x, pos_ids)

    t0 = time.time()
    h = x
    for l in range(64):
        is_attn = (l % 4 == 3)
        if is_attn:
            attn_layer.self_attn.layer_idx = l
            h = attn_layer(h, position_embeddings=pos_emb, past_key_values=cache, use_cache=True)
        else:
            linear_layer.linear_attn.layer_idx = l
            h = linear_layer(h, position_embeddings=pos_emb, past_key_values=cache, use_cache=True)

    torch.cuda.synchronize()
    print(f"Prefill 10 tokens through ALL 64 layers completed in {time.time()-t0:.2f}s!")
    print(f"Output shape: {h.shape}")
    print(f"Cache sequence length: {cache.get_seq_length()}")

    # Now test 1 decoding step
    t1 = time.time()
    next_x = torch.randn(1, 1, 5120, device=device, dtype=torch.bfloat16)
    next_pos_ids = torch.tensor([[[10]], [[10]], [[10]]], device=device)
    next_pos_emb = rotary(next_x, next_pos_ids)

    h_dec = next_x
    for l in range(64):
        is_attn = (l % 4 == 3)
        if is_attn:
            attn_layer.self_attn.layer_idx = l
            h_dec = attn_layer(h_dec, position_embeddings=next_pos_emb, past_key_values=cache, use_cache=True)
        else:
            linear_layer.linear_attn.layer_idx = l
            h_dec = linear_layer(h_dec, position_embeddings=next_pos_emb, past_key_values=cache, use_cache=True)

    torch.cuda.synchronize()
    print(f"1 decoding step through ALL 64 layers completed in {(time.time()-t1)*1000:.1f} ms!")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(0)/1e6:.1f} MB")

if __name__ == "__main__":
    test_full_pipeline()
