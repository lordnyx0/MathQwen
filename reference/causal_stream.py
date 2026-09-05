# -*- coding: utf-8 -*-
"""Loop causal de inferência pelas 64 camadas oficiais do Qwen."""
import torch
import torch.nn.functional as F
from .loader import load_qwen_reference_components, load_layer_module

def run_official_causal_stream(input_ids: torch.Tensor, device=None, return_all_h=False):
    if device is None:
        device = input_ids.device
    num_seqs, seq_len = input_ids.shape
    comp = load_qwen_reference_components(device=device, seq_len=seq_len, num_seqs=num_seqs)

    x = F.embedding(input_ids.cpu(), comp["embed_w"]).to(device=device, dtype=torch.bfloat16)
    pos_emb = comp["pos_emb"]
    cfg = comp["cfg"]
    snapshot_dir = comp["snapshot_dir"]

    h_all = [x.clone()] if return_all_h else None

    for l in range(64):
        layer_mod = load_layer_module(l, cfg, device, snapshot_dir=snapshot_dir)
        with torch.no_grad():
            out = layer_mod(x, position_embeddings=pos_emb)
            x = out[0] if isinstance(out, tuple) else out
        if return_all_h:
            h_all.append(x.clone())
        del layer_mod
        torch.cuda.empty_cache()

    # Projeção final RMSNorm + lm_head
    with torch.no_grad():
        x_norm = comp["norm_mod"](x)
        logits = F.linear(x_norm, comp["lm_head_w"])

    if return_all_h:
        return logits, h_all
    return logits
