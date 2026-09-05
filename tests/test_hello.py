import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch, glob, os, time, safetensors.torch as st
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from tokenizers import Tokenizer

snapshot_dir = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*'))[0]
cfg = AutoConfig.from_pretrained(snapshot_dir)
device = 'cuda:0'

linear_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
attn_layer = Qwen3_5DecoderLayer(cfg.text_config, layer_idx=3).to(device=device, dtype=torch.bfloat16)
del attn_layer.mlp
attn_layer.mlp = linear_layer.mlp
del attn_layer.self_attn.o_proj
attn_layer.self_attn.o_proj = linear_layer.linear_attn.out_proj
linear_layer.requires_grad_(False)
attn_layer.requires_grad_(False)
rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config).to(device)

model_path = 'models/g_qwen_9b_native_16charts.safetensors'
print(f"Carregando {model_path}...")
t_raw = st.load_file(model_path)
tensors = {k: (v.pin_memory() if 'token_embd' in k else v.to(device, non_blocking=True)) for k, v in t_raw.items()}
del t_raw
torch.cuda.empty_cache()

def dequant_4bit_into(packed_q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor):
    shape = out.shape
    low = (packed_q & 0x0F).to(torch.int8) - 7
    high = ((packed_q >> 4) & 0x0F).to(torch.int8) - 7
    unpacked = torch.stack([low, high], dim=1).view(-1, 128)
    with torch.no_grad():
        out.copy_((unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape))

def dequant_2bit_into(packed_2q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor):
    shape = out.shape
    v0 = (packed_2q & 0x03).to(torch.int8) - 2
    v1 = ((packed_2q >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed_2q >> 4) & 0x03).to(torch.int8) - 2
    v3 = ((packed_2q >> 6) & 0x03).to(torch.int8) - 2
    unpacked = torch.stack([v0, v1, v2, v3], dim=1).view(-1, 128)
    with torch.no_grad():
        out.copy_((unpacked.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)).view(shape))

def get_coord_s(tensors: dict, prefix: str) -> torch.Tensor:
    fp8_val = tensors[f"{prefix}.fp8"].to(torch.float32)
    scale = tensors[f"{prefix}.scale"].to(torch.float32)
    return (fp8_val * scale).to(torch.bfloat16)

def execute_layer(cache, v_idx, l_idx, h_in, position_embeddings):
    is_attn = (l_idx % 4 == 3)
    g = l_idx // 4
    U_mix = tensors[f'chart.{g}.basis_mixer_U'].to(torch.bfloat16)
    V_mix = tensors[f'chart.{g}.basis_mixer_V'].to(torch.bfloat16)
    U_gate = tensors[f'chart.{g}.basis_ffn_gate_U'].to(torch.bfloat16)
    V_gate = tensors[f'chart.{g}.basis_ffn_gate_V'].to(torch.bfloat16)
    U_up = tensors[f'chart.{g}.basis_ffn_up_U'].to(torch.bfloat16)
    V_up = tensors[f'chart.{g}.basis_ffn_up_V'].to(torch.bfloat16)
    U_down = tensors[f'chart.{g}.basis_ffn_down_U'].to(torch.bfloat16)
    V_down = tensors[f'chart.{g}.basis_ffn_down_V'].to(torch.bfloat16)

    S_gate = get_coord_s(tensors, f'blk.{l_idx}.ffn_gate_S')
    S_up = get_coord_s(tensors, f'blk.{l_idx}.ffn_up_S')
    S_down = get_coord_s(tensors, f'blk.{l_idx}.ffn_down_S')
    S_mix = get_coord_s(tensors, f'blk.{l_idx}.mixer_S')

    dequant_2bit_into(tensors[f'blk.{l_idx}.ffn_gate_res_2q'], tensors[f'blk.{l_idx}.ffn_gate_res_2scale'], linear_layer.mlp.gate_proj.weight)
    with torch.no_grad():
        torch.addmm(linear_layer.mlp.gate_proj.weight, U_gate, torch.matmul(S_gate, V_gate.t()), out=linear_layer.mlp.gate_proj.weight)
    dequant_2bit_into(tensors[f'blk.{l_idx}.ffn_up_res_2q'], tensors[f'blk.{l_idx}.ffn_up_res_2scale'], linear_layer.mlp.up_proj.weight)
    with torch.no_grad():
        torch.addmm(linear_layer.mlp.up_proj.weight, U_up, torch.matmul(S_up, V_up.t()), out=linear_layer.mlp.up_proj.weight)
    dequant_2bit_into(tensors[f'blk.{l_idx}.ffn_down_res_2q'], tensors[f'blk.{l_idx}.ffn_down_res_2scale'], linear_layer.mlp.down_proj.weight)
    with torch.no_grad():
        torch.addmm(linear_layer.mlp.down_proj.weight, U_down, torch.matmul(S_down, V_down.t()), out=linear_layer.mlp.down_proj.weight)

    if is_attn:
        mod = attn_layer
        mod.self_attn.layer_idx = v_idx
        dequant_4bit_into(tensors[f'blk.{l_idx}.mixer_res_q'], tensors[f'blk.{l_idx}.mixer_res_scale'], mod.self_attn.o_proj.weight)
        with torch.no_grad():
            torch.addmm(mod.self_attn.o_proj.weight, U_mix, torch.matmul(S_mix, V_mix.t()), out=mod.self_attn.o_proj.weight)
        dequant_4bit_into(tensors[f'blk.{l_idx}.attn_q.q'], tensors[f'blk.{l_idx}.attn_q.s'], mod.self_attn.q_proj.weight)
        dequant_4bit_into(tensors[f'blk.{l_idx}.attn_k.q'], tensors[f'blk.{l_idx}.attn_k.s'], mod.self_attn.k_proj.weight)
        dequant_4bit_into(tensors[f'blk.{l_idx}.attn_v.q'], tensors[f'blk.{l_idx}.attn_v.s'], mod.self_attn.v_proj.weight)
        with torch.no_grad():
            mod.self_attn.q_norm.weight.copy_(tensors[f'blk.{l_idx}.attn_q_norm.weight'])
            mod.self_attn.k_norm.weight.copy_(tensors[f'blk.{l_idx}.attn_k_norm.weight'])
            mod.input_layernorm.weight.copy_(tensors[f'blk.{l_idx}.attn_norm.weight'])
            mod.post_attention_layernorm.weight.copy_(tensors[f'blk.{l_idx}.ffn_norm.weight'])
    else:
        mod = linear_layer
        mod.linear_attn.layer_idx = v_idx
        dequant_4bit_into(tensors[f'blk.{l_idx}.mixer_res_q'], tensors[f'blk.{l_idx}.mixer_res_scale'], mod.linear_attn.out_proj.weight)
        with torch.no_grad():
            torch.addmm(mod.linear_attn.out_proj.weight, U_mix, torch.matmul(S_mix, V_mix.t()), out=mod.linear_attn.out_proj.weight)
        dequant_4bit_into(tensors[f'blk.{l_idx}.attn_qkv.q'], tensors[f'blk.{l_idx}.attn_qkv.s'], mod.linear_attn.in_proj_qkv.weight)
        dequant_4bit_into(tensors[f'blk.{l_idx}.attn_gate.q'], tensors[f'blk.{l_idx}.attn_gate.s'], mod.linear_attn.in_proj_z.weight)
        with torch.no_grad():
            mod.linear_attn.in_proj_a.weight.copy_(tensors[f'blk.{l_idx}.ssm_alpha.weight'])
            mod.linear_attn.in_proj_b.weight.copy_(tensors[f'blk.{l_idx}.ssm_beta.weight'])
            mod.linear_attn.conv1d.weight.copy_(tensors[f'blk.{l_idx}.ssm_conv1d.weight'])
            mod.linear_attn.dt_bias.copy_(tensors[f'blk.{l_idx}.ssm_dt.bias'])
            mod.linear_attn.A_log.copy_(tensors[f'blk.{l_idx}.ssm_a.weight'])
            mod.linear_attn.norm.weight.copy_(tensors[f'blk.{l_idx}.ssm_norm.weight'])
            mod.input_layernorm.weight.copy_(tensors[f'blk.{l_idx}.attn_norm.weight'])
            mod.post_attention_layernorm.weight.copy_(tensors[f'blk.{l_idx}.ffn_norm.weight'])

    with torch.no_grad():
        out = mod(hidden_states=h_in, position_embeddings=position_embeddings, past_key_values=cache, use_cache=True)
    return out[0] if isinstance(out, tuple) else out

tok = Tokenizer.from_file(os.path.join(snapshot_dir, 'tokenizer.json'))
t_id = tok.encode('Hello').ids[0]
p_row = tensors['token_embd.q'][(t_id * 5120)//2 : (t_id * 5120)//2 + 2560].to(device)
s_row = tensors['token_embd.s'][t_id * 40 : (t_id + 1) * 40].to(device)
low = (p_row & 0x0F).to(torch.int8) - 7
high = ((p_row >> 4) & 0x0F).to(torch.int8) - 7
h_init = (torch.stack([low, high], dim=1).view(-1, 128).to(torch.bfloat16) * s_row.view(-1, 1).to(torch.bfloat16)).view(1, 1, 5120)

pos_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
pos_emb = rotary(h_init, pos_ids)

chunk_size = 248320 // 32
head_q = tensors['lm_head.q']
head_scale = tensors['lm_head.s']

def compute_and_print_top(h, label):
    var = h.to(torch.float32).pow(2).mean(-1, keepdim=True)
    h_norm = (h * torch.rsqrt(var + 1e-6)).to(torch.bfloat16) * tensors['output_norm.weight'].to(device=device, dtype=torch.bfloat16)
    h_vec = h_norm.view(5120, 1)
    logits = torch.empty(248320, dtype=torch.float32, device=device)
    for i in range(32):
        p_c = head_q[i * chunk_size * 2560 : (i + 1) * chunk_size * 2560]
        s_c = head_scale[i * chunk_size * 40 : (i + 1) * chunk_size * 40]
        low = (p_c & 0x0F).to(torch.int8) - 7
        high = ((p_c >> 4) & 0x0F).to(torch.int8) - 7
        w_c = (torch.stack([low, high], dim=1).view(-1, 128).to(torch.bfloat16) * s_c.view(-1, 1).to(torch.bfloat16)).view(chunk_size, 5120)
        logits[i*chunk_size : (i+1)*chunk_size] = torch.matmul(w_c, h_vec).squeeze(1)

    vals, idxs = torch.topk(logits, 5)
    print(f'\n--- {label} (h norm = {h.norm().item():.1f}) ---', flush=True)
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        print(f'Token {idx} ({repr(tok.decode([idx]))}): logit = {v:.2f}', flush=True)

# Teste: Forward Sequencial 64 Camadas Fisicas (Zero Triple Loop)
print("\nIniciando Teste com 64 Camadas Integrais (0..63)...", flush=True)
cache = DynamicCache(config=cfg.text_config)
h = h_init.clone()
t0 = time.time()
for l in range(64):
    h = execute_layer(cache, l, l, h, pos_emb)
print(f"Forward pass de 64 camadas finalizado em {time.time()-t0:.2f}s!")
compute_and_print_top(h, "Passo Unico Sequencial 64 Camadas")
