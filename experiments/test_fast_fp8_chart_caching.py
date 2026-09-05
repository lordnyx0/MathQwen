"""Test pre-reconstructing all 16 charts on GPU and holding in CPU RAM as FP8."""
import os
import glob
import time
import torch
import safetensors.torch as st

def dequant_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, 128)
    return (unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape)

def test_caching():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    charts_dir = "models/g_qwen_9b_phase1"

    t0 = time.time()
    cached_layers = []

    for g in range(16):
        t_chart = time.time()
        cpath = os.path.join(charts_dir, f"chart_{g}.safetensors")
        cd = st.load_file(cpath, device=str(device))

        U_mix = cd['basis_mixer_U'].to(torch.bfloat16)
        V_mix = cd['basis_mixer_V'].to(torch.bfloat16)
        U_gate = cd['basis_ffn_gate_U'].to(torch.bfloat16)
        V_gate = cd['basis_ffn_gate_V'].to(torch.bfloat16)
        U_up = cd['basis_ffn_up_U'].to(torch.bfloat16)
        V_up = cd['basis_ffn_up_V'].to(torch.bfloat16)
        U_down = cd['basis_ffn_down_U'].to(torch.bfloat16)
        V_down = cd['basis_ffn_down_V'].to(torch.bfloat16)

        for idx in range(4):
            l = 4 * g + idx
            # Mixer
            R_mix = dequant_4bit(cd[f'layer_{l}_mixer_res_q'], cd[f'layer_{l}_mixer_res_scale'], torch.Size([5120, 6144]))
            W_mix = (torch.matmul(U_mix, torch.matmul(cd[f'layer_{l}_mixer_S'].to(torch.bfloat16), V_mix.t())) + R_mix).to(torch.float8_e4m3fn).cpu()

            # Gate
            R_gate = dequant_4bit(cd[f'layer_{l}_ffn_gate_res_q'], cd[f'layer_{l}_ffn_gate_res_scale'], torch.Size([17408, 5120]))
            W_gate = (torch.matmul(U_gate, torch.matmul(cd[f'layer_{l}_ffn_gate_S'].to(torch.bfloat16), V_gate.t())) + R_gate).to(torch.float8_e4m3fn).cpu()

            # Up
            R_up = dequant_4bit(cd[f'layer_{l}_ffn_up_res_q'], cd[f'layer_{l}_ffn_up_res_scale'], torch.Size([17408, 5120]))
            W_up = (torch.matmul(U_up, torch.matmul(cd[f'layer_{l}_ffn_up_S'].to(torch.bfloat16), V_up.t())) + R_up).to(torch.float8_e4m3fn).cpu()

            # Down
            R_down = dequant_4bit(cd[f'layer_{l}_ffn_down_res_q'], cd[f'layer_{l}_ffn_down_res_scale'], torch.Size([5120, 17408]))
            W_down = (torch.matmul(U_down, torch.matmul(cd[f'layer_{l}_ffn_down_S'].to(torch.bfloat16), V_down.t())) + R_down).to(torch.float8_e4m3fn).cpu()

            cached_layers.append((W_mix, W_gate, W_up, W_down))

        print(f"Chart {g+1}/16 cached in {(time.time()-t_chart)*1000:.1f} ms", flush=True)

    print(f"\nAll 64 layers (16 charts) reconstructed and cached in CPU RAM in {time.time()-t0:.2f}s!")
    print(f"Total layers cached: {len(cached_layers)}")
    mem_mb = sum(sum(m.numel() for m in l) for l in cached_layers) / 1e6
    print(f"Total CPU RAM used for 64 layers: {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")

if __name__ == "__main__":
    test_caching()
