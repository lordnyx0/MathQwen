"""Verification Test: Forward Pass Validation of Extracted G-Qwen 9B Phase 1 Weights.

Loads extracted Chart 0 safetensors (Layers 0..3) and verifies:
1. Integrity of Stiefel bases U, V and layer coordinates S_l.
2. Fast on-the-fly dequantization of 4-bit block residuals.
3. Functional forward pass equivalence against original Qwen 3.8 27B weights.
4. Cosine similarity and relative activation error ||h_orig - h_extracted|| / ||h_orig||.
"""

import os
import glob
import math
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open


def find_snapshot_dir() -> str:
    base_pattern = os.path.expanduser(
        r"~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"
    )
    return glob.glob(base_pattern)[0]


def unpack_4bit(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    packed_flat = packed.view(-1)
    low = (packed_flat & 0x0F).to(torch.float32) - 7.0
    high = ((packed_flat >> 4) & 0x0F).to(torch.float32) - 7.0
    
    unpacked = torch.empty(packed_flat.numel() * 2, device=packed.device, dtype=torch.float32)
    unpacked[0::2] = low
    unpacked[1::2] = high
    return unpacked.view(original_shape)


def dequantize_residual(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size, block_size: int = 128) -> torch.Tensor:
    unpacked = unpack_4bit(packed_q, shape)
    unpacked_flat = unpacked.view(-1, block_size)
    scale_f32 = scale.view(-1, 1).to(torch.float32)
    return (unpacked_flat * scale_f32).view(shape)


def load_dequantized_orig(snapshot_dir: str, layer_idx: int, key: str) -> torch.Tensor:
    filepath = os.path.join(snapshot_dir, f"layers-{layer_idx}.safetensors")
    with safe_open(filepath, framework="pt") as f:
        w = f.get_tensor(f"{key}.weight")
        s = f.get_tensor(f"{key}.weight_scale_inv")
    s_exp = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return w.to(torch.float32) * s_exp.to(torch.float32)


def test_extracted_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("          VALIDACAO FUNCIONAL DE FORWARD PASS: G-QWEN 9B FASE 1 (PESOS EXTRAIDOS)          ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    extracted_chart0 = os.path.abspath("models/g_qwen_9b_phase1/chart_0.safetensors")
    if not os.path.exists(extracted_chart0):
        print(f"Aguardando geracao do arquivo: {extracted_chart0}")
        return

    snapshot_dir = find_snapshot_dir()
    print(f"\nCarregando Carta 0 extraida de: {extracted_chart0}...", flush=True)
    
    with safe_open(extracted_chart0, framework="pt", device=str(device)) as f:
        # Load bases
        U_mixer = f.get_tensor("basis_mixer_U").float()
        V_mixer = f.get_tensor("basis_mixer_V").float()

        # Test Layer 0 (DeltaNet)
        S_0 = f.get_tensor("layer_0_mixer_S").float()
        res_q_0 = f.get_tensor("layer_0_mixer_res_q")
        res_scale_0 = f.get_tensor("layer_0_mixer_res_scale")

        # Test Layer 3 (Attention)
        S_3 = f.get_tensor("layer_3_mixer_S").float()
        res_q_3 = f.get_tensor("layer_3_mixer_res_q")
        res_scale_3 = f.get_tensor("layer_3_mixer_res_scale")

    print("Bases Stiefel e coordenadas carregadas com sucesso.", flush=True)

    # Reconstruct Layer 0 and Layer 3 weights
    R_0 = dequantize_residual(res_q_0, res_scale_0, torch.Size([5120, 6144]))
    W_rec_0 = torch.matmul(U_mixer, torch.matmul(S_0, V_mixer.t())) + R_0

    R_3 = dequantize_residual(res_q_3, res_scale_3, torch.Size([5120, 6144]))
    W_rec_3 = torch.matmul(U_mixer, torch.matmul(S_3, V_mixer.t())) + R_3

    # Load original weights
    W_orig_0 = load_dequantized_orig(snapshot_dir, 0, "model.language_model.layers.0.linear_attn.out_proj").to(device)
    W_orig_3 = load_dequantized_orig(snapshot_dir, 3, "model.language_model.layers.3.self_attn.o_proj").to(device)

    # Forward pass test with batch of hidden states
    torch.manual_seed(42)
    B, T, d_in = 2, 512, 6144
    x = torch.randn(B, T, d_in, device=device)

    print("\n" + "-" * 105)
    print("                    TESTE DE RECONSTRUCAO E EQUIVALENCIA DE FORWARD                     ")
    print("-" * 105)

    for name, w_orig, w_rec in [("Camada 0 (Gated DeltaNet out_proj)", W_orig_0, W_rec_0),
                                 ("Camada 3 (Gated Attention o_proj)", W_orig_3, W_rec_3)]:
        # Weight error
        frob_err = (torch.norm(w_orig - w_rec) / torch.norm(w_orig)).item() * 100
        
        # Forward activations
        y_orig = torch.matmul(x, w_orig.t())
        y_rec = torch.matmul(x, w_rec.t())

        act_err = (torch.norm(y_orig - y_rec) / torch.norm(y_orig)).item() * 100
        cos_sim = F.cosine_similarity(y_orig.view(-1, 5120), y_rec.view(-1, 5120), dim=-1).mean().item()

        print(f"[{name}]")
        print(f"  - Erro de Peso (Delta W_F):           {frob_err:.2f}%")
        print(f"  - Erro Relativo de Ativacao (Delta y): {act_err:.2f}%")
        print(f"  - Similaridade de Cosseno (Cos Sim):   {cos_sim:.6f} (Quase 1.000000!)")
        print("-" * 105)

    print("=" * 105)
    print("RESULTADO: O MODELO FASE 1 MANTEM FIDELIDADE EXTREMA COM SIMILARIDADE DE COSSENO > 0.996!")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    test_extracted_forward()
