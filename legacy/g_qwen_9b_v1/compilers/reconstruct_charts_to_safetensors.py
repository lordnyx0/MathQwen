"""Reconstructor of G-Qwen 9B Charts: Stiefel Atlas + RVQ 4-Bit -> Direct Float16 cuBLAS Safetensors.

Converts compressed Stiefel charts into ready-to-stream float16 layer blocks on disk.
Zero runtime dequantization overhead during token generation!
"""

import os
import sys
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def fast_dequantize_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size, block_size: int = 128) -> torch.Tensor:
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, block_size)
    return (unpacked.to(torch.float16) * scale.view(-1, 1).to(torch.float16)).view(shape)


def reconstruct_all_charts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = os.path.abspath("models/g_qwen_9b_phase1")
    output_dir = os.path.abspath("models/g_qwen_9b_reconstructed")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 105, flush=True)
    print("   RECONSTRUCAO DIRETA DOS PESOS DAS 16 CARTAS (STIEFEL ATLAS + RVQ 4-BIT -> FLOAT16 CUBLAS)   ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    t0_all = time.time()

    for g in range(16):
        out_path = os.path.join(output_dir, f"chart_{g}.safetensors")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 2.0 * (1024**3):
            print(f"[{g+1}/16] Carta {g} ja reconstruida ({os.path.getsize(out_path)/(1024**3):.2f} GB) - Pulando.", flush=True)
            continue

        in_path = os.path.join(input_dir, f"chart_{g}.safetensors")
        layers = [4 * g + i for i in range(4)]
        t0 = time.time()
        print(f"\n[{g+1}/16] Reconstruindo Carta {g} (Camadas {layers[0]}..{layers[-1]})...", end="", flush=True)

        rec_dict: Dict[str, torch.Tensor] = {}

        with safe_open(in_path, framework="pt", device=str(device)) as cf:
            # 1. Bases
            U_mix = cf.get_tensor("basis_mixer_U").to(torch.float16)
            V_mix = cf.get_tensor("basis_mixer_V").to(torch.float16)
            U_gate = cf.get_tensor("basis_ffn_gate_U").to(torch.float16)
            V_gate = cf.get_tensor("basis_ffn_gate_V").to(torch.float16)
            U_up = cf.get_tensor("basis_ffn_up_U").to(torch.float16)
            V_up = cf.get_tensor("basis_ffn_up_V").to(torch.float16)
            U_down = cf.get_tensor("basis_ffn_down_U").to(torch.float16)
            V_down = cf.get_tensor("basis_ffn_down_V").to(torch.float16)

            for l in layers:
                # Mixer
                S_mix = cf.get_tensor(f"layer_{l}_mixer_S").to(torch.float16)
                q_mix = cf.get_tensor(f"layer_{l}_mixer_res_q")
                s_mix = cf.get_tensor(f"layer_{l}_mixer_res_scale")
                R_mix = fast_dequantize_4bit(q_mix, s_mix, torch.Size([5120, 6144]))
                W_mix = torch.matmul(U_mix, torch.matmul(S_mix, V_mix.t())) + R_mix
                rec_dict[f"layer_{l}_W_mix"] = W_mix.contiguous().cpu()

                # Gate
                S_gate = cf.get_tensor(f"layer_{l}_ffn_gate_S").to(torch.float16)
                q_gate = cf.get_tensor(f"layer_{l}_ffn_gate_res_q")
                s_gate = cf.get_tensor(f"layer_{l}_ffn_gate_res_scale")
                R_gate = fast_dequantize_4bit(q_gate, s_gate, torch.Size([17408, 5120]))
                W_gate = torch.matmul(U_gate, torch.matmul(S_gate, V_gate.t())) + R_gate
                rec_dict[f"layer_{l}_W_gate"] = W_gate.contiguous().cpu()

                # Up
                S_up = cf.get_tensor(f"layer_{l}_ffn_up_S").to(torch.float16)
                q_up = cf.get_tensor(f"layer_{l}_ffn_up_res_q")
                s_up = cf.get_tensor(f"layer_{l}_ffn_up_res_scale")
                R_up = fast_dequantize_4bit(q_up, s_up, torch.Size([17408, 5120]))
                W_up = torch.matmul(U_up, torch.matmul(S_up, V_up.t())) + R_up
                rec_dict[f"layer_{l}_W_up"] = W_up.contiguous().cpu()

                # Down
                S_down = cf.get_tensor(f"layer_{l}_ffn_down_S").to(torch.float16)
                q_down = cf.get_tensor(f"layer_{l}_ffn_down_res_q")
                s_down = cf.get_tensor(f"layer_{l}_ffn_down_res_scale")
                R_down = fast_dequantize_4bit(q_down, s_down, torch.Size([5120, 17408]))
                W_down = torch.matmul(U_down, torch.matmul(S_down, V_down.t())) + R_down
                rec_dict[f"layer_{l}_W_down"] = W_down.contiguous().cpu()

        torch.cuda.empty_cache()
        save_file(rec_dict, out_path)
        dt = time.time() - t0
        mb = os.path.getsize(out_path) / (1024**2)
        print(f" Concluido! Salvo: {mb:.1f} MB em {dt:.2f}s ({mb/dt:.1f} MB/s)", flush=True)

    total_time = time.time() - t0_all
    print("\n" + "=" * 105, flush=True)
    print(f"[CONCLUIDO] Todas as 16 cartas reconstruidas em {total_time:.2f}s ({total_time/60:.2f} min)!", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    reconstruct_all_charts()
