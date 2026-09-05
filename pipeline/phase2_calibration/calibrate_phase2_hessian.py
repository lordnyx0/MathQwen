r"""Fase 2: Calibrador de Coordenadas de Foliacao Stiefel via Hessiana de Ativacao (Hessian/AWQ).

Objetivo:
- Ajustar as coordenadas S_l para minimizar ||X (W_l - \hat{W}_l)^T||_F^2
- Utiliza corpus de raciocinio e logica (ianncity___glm-5.2-logic-puzzles)
- Processamento estritamente camada a camada (0..63) em VRAM controlada (~6.0 GB)
- Salva os checkpoints calibrados em models/g_qwen_9b_phase2/
"""

import os
import sys
import glob
import time
import psutil
import torch
import pyarrow.ipc as ipc
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
    return (w.to(torch.float32) * s_exp.to(torch.float32)).to(torch.float16)


def dequant_4bit(packed_q: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, 128)
    return (unpacked.to(torch.float32) * scale.view(-1, 1).to(torch.float32)).view(shape)


def solve_optimal_s(U: torch.Tensor, V: torch.Tensor, E: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Resolve analiticamente S* = (U^T E H V) (V^T H V + lambda I)^-1 em CUDA."""
    r = V.shape[1]
    VHV = torch.matmul(V.t(), torch.matmul(H, V))
    trace_val = torch.trace(VHV).item()
    lambda_reg = 1e-4 * (trace_val / max(r, 1))
    reg = lambda_reg * torch.eye(r, dtype=torch.float32, device=H.device)
    
    VHV_inv = torch.linalg.inv(VHV + reg)
    HV = torch.matmul(H, V)
    E_HV = torch.matmul(E, HV)
    UT_E_HV = torch.matmul(U.t(), E_HV)
    S_opt = torch.matmul(UT_E_HV, VHV_inv)
    return S_opt.to(torch.float16)


def run_phase2_calibration():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print("    FASE 2: CALIBRACAO HESSIAN / AWQ DAS COORDENADAS STIEFEL (G-QWEN 9B)                   ", flush=True)
    print("=" * 105, flush=True)
    print(f"Dispositivo GPU : {device} ({torch.cuda.get_device_name(0)})", flush=True)

    snapshot_dir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    phase1_dir = os.path.abspath("models/g_qwen_9b_phase1")
    phase2_dir = os.path.abspath("models/g_qwen_9b_phase2")
    os.makedirs(phase2_dir, exist_ok=True)
    outside_path = os.path.join(snapshot_dir, "outside.safetensors")

    # 1. Carregar Tokenizador e Dataset de Calibracao
    print("\n[1/4] Carregando Tokenizador e Dataset de Raciocinio...", flush=True)
    tokenizer = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))

    arrow_path = glob.glob(os.path.expanduser("~/.cache/huggingface/datasets/ianncity___glm-5.2-logic-puzzles/**/*.arrow"), recursive=True)[0]
    reader = ipc.open_stream(arrow_path)
    table = reader.read_all()

    target_seq_len = 128
    target_batches = 64
    token_batches = []

    for i in range(table.num_rows):
        msgs = table["messages"][i].as_py()
        full_txt = ""
        for m in msgs:
            full_txt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        ids = tokenizer.encode(full_txt).ids
        if len(ids) >= target_seq_len:
            token_batches.append(ids[:target_seq_len])
        if len(token_batches) == target_batches:
            break

    print(f" Dataset de calibracao preparado: {len(token_batches)} sequencias de exatamente {target_seq_len} tokens.", flush=True)

    # 2. Modulos de Execucao e Embeddings
    print("\n[2/4] Preparando Embeddings e Modulos de Camada...", flush=True)
    cfg = AutoConfig.from_pretrained(snapshot_dir)
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)

    linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)
    del attn_layer.mlp
    attn_layer.mlp = linear_layer.mlp
    del attn_layer.self_attn.o_proj
    attn_layer.self_attn.o_proj = linear_layer.linear_attn.out_proj
    linear_layer.requires_grad_(False)
    attn_layer.requires_grad_(False)

    with st.safe_open(outside_path, framework="pt") as f:
        embed_tokens = f.get_tensor("model.language_model.embed_tokens.weight").to(device=device, dtype=torch.bfloat16)

    print(" Computando ativacoes de embedding X_0...", end="", flush=True)
    batch_tensor = torch.tensor(token_batches, dtype=torch.long, device=device)
    batch_size, seq_len = batch_tensor.shape
    h_current = embed_tokens[batch_tensor]
    del embed_tokens
    torch.cuda.empty_cache()
    print(" Concluido!", flush=True)

    pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
    pos_emb = rotary(h_current, pos_ids)

    # 3. Calibracao Camada por Camada (64 Camadas)
    print("\n[3/4] Iniciando Calibracao Hessian Sequencial (Camadas 0 a 63)...", flush=True)
    t0_all = time.time()

    for g in range(16):
        t_chart = time.time()
        chart_p1_path = os.path.join(phase1_dir, f"chart_{g}.safetensors")
        chart_p2_path = os.path.join(phase2_dir, f"chart_{g}.safetensors")

        cd = st.load_file(chart_p1_path)
        chart_dict = {k: v for k, v in cd.items()}

        U_mix = cd["basis_mixer_U"].to(device=device, dtype=torch.float32)
        V_mix = cd["basis_mixer_V"].to(device=device, dtype=torch.float32)
        U_gate = cd["basis_ffn_gate_U"].to(device=device, dtype=torch.float32)
        V_gate = cd["basis_ffn_gate_V"].to(device=device, dtype=torch.float32)
        U_up = cd["basis_ffn_up_U"].to(device=device, dtype=torch.float32)
        V_up = cd["basis_ffn_up_V"].to(device=device, dtype=torch.float32)
        U_down = cd["basis_ffn_down_U"].to(device=device, dtype=torch.float32)
        V_down = cd["basis_ffn_down_V"].to(device=device, dtype=torch.float32)

        for idx in range(4):
            l = 4 * g + idx
            is_attn = (l % 4 == 3)
            t_layer = time.time()

            X_flat = h_current.view(-1, 5120).to(torch.float32)
            N_tokens = X_flat.shape[0]
            H = torch.matmul(X_flat.t(), X_flat) / N_tokens

            layer_f = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
            with st.safe_open(layer_f, framework="pt") as lf:
                # FFN Gate
                w_gate_orig = dequant_fp8(
                    lf.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight").to(device),
                    lf.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv").to(device)
                ).to(torch.float32)
                res_gate = dequant_4bit(cd[f"layer_{l}_ffn_gate_res_q"].to(device), cd[f"layer_{l}_ffn_gate_res_scale"].to(device), torch.Size([17408, 5120]))
                E_gate = w_gate_orig - res_gate
                S_gate_opt = solve_optimal_s(U_gate, V_gate, E_gate, H)
                chart_dict[f"layer_{l}_ffn_gate_S"] = S_gate_opt.cpu()
                del w_gate_orig, res_gate, E_gate

                # FFN Up
                w_up_orig = dequant_fp8(
                    lf.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight").to(device),
                    lf.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv").to(device)
                ).to(torch.float32)
                res_up = dequant_4bit(cd[f"layer_{l}_ffn_up_res_q"].to(device), cd[f"layer_{l}_ffn_up_res_scale"].to(device), torch.Size([17408, 5120]))
                E_up = w_up_orig - res_up
                S_up_opt = solve_optimal_s(U_up, V_up, E_up, H)
                chart_dict[f"layer_{l}_ffn_up_S"] = S_up_opt.cpu()
                del w_up_orig, res_up, E_up

                # FFN Down
                w_down_orig = dequant_fp8(
                    lf.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight").to(device),
                    lf.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv").to(device)
                ).to(torch.float32)
                res_down = dequant_4bit(cd[f"layer_{l}_ffn_down_res_q"].to(device), cd[f"layer_{l}_ffn_down_res_scale"].to(device), torch.Size([5120, 17408]))
                E_down = w_down_orig - res_down
                S_down_opt = torch.matmul(U_down.t(), torch.matmul(E_down, V_down)).to(torch.float16)
                chart_dict[f"layer_{l}_ffn_down_S"] = S_down_opt.cpu()
                del w_down_orig, res_down, E_down

                # Mixer
                proj_key = "self_attn.o_proj" if is_attn else "linear_attn.out_proj"
                w_mix_orig = dequant_fp8(
                    lf.get_tensor(f"model.language_model.layers.{l}.{proj_key}.weight").to(device),
                    lf.get_tensor(f"model.language_model.layers.{l}.{proj_key}.weight_scale_inv").to(device)
                ).to(torch.float32)
                res_mix = dequant_4bit(cd[f"layer_{l}_mixer_res_q"].to(device), cd[f"layer_{l}_mixer_res_scale"].to(device), torch.Size([5120, 6144]))
                E_mix = w_mix_orig - res_mix
                S_mix_opt = torch.matmul(U_mix.t(), torch.matmul(E_mix, V_mix)).to(torch.float16)
                chart_dict[f"layer_{l}_mixer_S"] = S_mix_opt.cpu()
                del w_mix_orig, res_mix, E_mix

                # Montar modulos para forward pass
                if is_attn:
                    mod = attn_layer
                    mod.self_attn.layer_idx = l
                    mod.self_attn.o_proj.weight.copy_((torch.matmul(U_mix, torch.matmul(S_mix_opt.to(torch.float32), V_mix.t()))).to(torch.bfloat16))
                    mod.self_attn.q_proj.weight.copy_(dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight").to(device), lf.get_tensor(f"model.language_model.layers.{l}.self_attn.q_proj.weight_scale_inv").to(device)).to(torch.bfloat16))
                    mod.self_attn.k_proj.weight.copy_(dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight").to(device), lf.get_tensor(f"model.language_model.layers.{l}.self_attn.k_proj.weight_scale_inv").to(device)).to(torch.bfloat16))
                    mod.self_attn.v_proj.weight.copy_(dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight").to(device), lf.get_tensor(f"model.language_model.layers.{l}.self_attn.v_proj.weight_scale_inv").to(device)).to(torch.bfloat16))
                    mod.self_attn.q_norm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.self_attn.q_norm.weight").to(device=device, dtype=torch.bfloat16))
                    mod.self_attn.k_norm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.self_attn.k_norm.weight").to(device=device, dtype=torch.bfloat16))
                    mod.input_layernorm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.input_layernorm.weight").to(device=device, dtype=torch.bfloat16))
                    mod.post_attention_layernorm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.post_attention_layernorm.weight").to(device=device, dtype=torch.bfloat16))
                else:
                    mod = linear_layer
                    mod.linear_attn.layer_idx = l
                    mod.linear_attn.out_proj.weight.copy_((torch.matmul(U_mix, torch.matmul(S_mix_opt.to(torch.float32), V_mix.t()))).to(torch.bfloat16))
                    mod.linear_attn.in_proj_qkv.weight.copy_(dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight").to(device), lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight_scale_inv").to(device)).to(torch.bfloat16))
                    mod.linear_attn.in_proj_z.weight.copy_(dequant_fp8(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight").to(device), lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_z.weight_scale_inv").to(device)).to(torch.bfloat16))
                    mod.linear_attn.in_proj_a.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_a.weight").to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.in_proj_b.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.in_proj_b.weight").to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.conv1d.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.conv1d.weight").to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.dt_bias.copy_(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.dt_bias").to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.A_log.copy_(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.A_log").to(device=device, dtype=torch.bfloat16))
                    mod.linear_attn.norm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.linear_attn.norm.weight").to(device=device, dtype=torch.bfloat16))
                    mod.input_layernorm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.input_layernorm.weight").to(device=device, dtype=torch.bfloat16))
                    mod.post_attention_layernorm.weight.copy_(lf.get_tensor(f"model.language_model.layers.{l}.post_attention_layernorm.weight").to(device=device, dtype=torch.bfloat16))

                mod.mlp.gate_proj.weight.copy_((torch.matmul(U_gate, torch.matmul(S_gate_opt.to(torch.float32), V_gate.t()))).to(torch.bfloat16))
                mod.mlp.up_proj.weight.copy_((torch.matmul(U_up, torch.matmul(S_up_opt.to(torch.float32), V_up.t()))).to(torch.bfloat16))
                mod.mlp.down_proj.weight.copy_((torch.matmul(U_down, torch.matmul(S_down_opt.to(torch.float32), V_down.t()))).to(torch.bfloat16))

            # Forward pass para atualizar ativacoes da proxima camada
            cache_l = DynamicCache(config=cfg.text_config)
            with torch.no_grad():
                out = mod(hidden_states=h_current, position_embeddings=pos_emb, past_key_values=cache_l, use_cache=False)
                h_current = out[0] if isinstance(out, tuple) else out

            del H, X_flat
            torch.cuda.empty_cache()

            vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"  [Camada {l:2d}/63] Calibrada em {time.time()-t_layer:.2f}s | VRAM: {vram_gb:.2f} GB | h_norm: {h_current.norm().item():.1f}", flush=True)

        st.save_file(chart_dict, chart_p2_path)
        print(f" > Macro-Carta {g+1:2d}/16 gravada em {chart_p2_path} ({time.time()-t_chart:.2f}s)\n", flush=True)

        del U_mix, V_mix, U_gate, V_gate, U_up, V_up, U_down, V_down, chart_dict, cd
        torch.cuda.empty_cache()

    print("=" * 105)
    print(f"FASE 2 CONCLUIDA COM SUCESSO EM {time.time()-t0_all:.2f}s!")
    print(f"Diretorio de Saida: {phase2_dir}")
    print("Todas as 16 Macro-Cartas foram calibradas com matrizes Hessian de raciocinio.")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_phase2_calibration()
