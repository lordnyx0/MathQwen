# -*- coding: utf-8 -*-
"""Álgebra de projeção das bases do Atlas Assimétrico (V_joint, U_mix, U_down)."""
import torch

def compute_chart_atlas_bases(chart_weights: dict, r_base: int = 2048, device=None):
    if device is None:
        device = list(chart_weights.values())[0]["gate"].device

    G_mix = torch.zeros(5120, 5120, device=device, dtype=torch.float32)
    G_down = torch.zeros(5120, 5120, device=device, dtype=torch.float32)
    G_joint = torch.zeros(5120, 5120, device=device, dtype=torch.float32)

    for l_idx, w in chart_weights.items():
        W_m = w["mix"].to(device=device, dtype=torch.float32)
        G_mix += torch.matmul(W_m, W_m.t())

        W_g = w["gate"].to(device=device, dtype=torch.float32)
        W_u = w["up"].to(device=device, dtype=torch.float32)
        G_joint += torch.matmul(W_g.t(), W_g) + torch.matmul(W_u.t(), W_u)

        W_d = w["down"].to(device=device, dtype=torch.float32)
        G_down += torch.matmul(W_d, W_d.t())

    _, U_mix_full = torch.linalg.eigh(G_mix)
    _, U_down_full = torch.linalg.eigh(G_down)
    _, V_joint_full = torch.linalg.eigh(G_joint)

    U_m = U_mix_full[:, -r_base:].to(dtype=torch.bfloat16)
    U_d = U_down_full[:, -r_base:].to(dtype=torch.bfloat16)
    V_j = V_joint_full[:, -r_base:].to(dtype=torch.bfloat16)

    return {"U_mix": U_m, "U_down": U_d, "V_joint": V_j}

def project_layer_weights(orig_weights: dict, bases: dict):
    U_m = bases["U_mix"]
    U_d = bases["U_down"]
    V_j = bases["V_joint"]
    device = U_m.device
    dtype = U_m.dtype

    w_mix = orig_weights["mix"].to(device=device, dtype=dtype)
    w_gate = orig_weights["gate"].to(device=device, dtype=dtype)
    w_up = orig_weights["up"].to(device=device, dtype=dtype)
    w_down = orig_weights["down"].to(device=device, dtype=dtype)

    W_mix_r = torch.matmul(U_m, torch.matmul(U_m.t(), w_mix))
    W_gate_r = torch.matmul(torch.matmul(w_gate, V_j), V_j.t())
    W_up_r = torch.matmul(torch.matmul(w_up, V_j), V_j.t())
    W_down_r = torch.matmul(U_d, torch.matmul(U_d.t(), w_down))

    return {
        "mix": W_mix_r,
        "gate": W_gate_r,
        "up": W_up_r,
        "down": W_down_r,
        "is_attn": orig_weights["is_attn"]
    }
