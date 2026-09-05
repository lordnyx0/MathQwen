# -*- coding: utf-8 -*-
"""Modelo de inferência Atlas 100% autônomo e autocontido (sem dependência do checkpoint original)."""
import os
import json
import torch
import torch.nn.functional as F
import safetensors.torch as st
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5RMSNorm
)
from transformers import AutoConfig
from .config import AtlasConfig
from .residual import LinearResidualStabilizer, NonLinearResidualStabilizer
from reference.loader import dequant

class AtlasAutonomousModel:
    """
    Runtime de inferência do Atlas que consome exclusivamente o modelo comprimido exportado.
    Não depende de checkpoints do HuggingFace, não faz decomposições espectrais em tempo de execução
    e não toca em nenhum arquivo de modelo externo.
    """
    def __init__(self, model_dir: str = "checkpoints/atlas_autonomous", device = None):
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json não encontrado em: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg_data = json.load(f)

        self.config = AtlasConfig(
            vocab_size=self.cfg_data["vocab_size"],
            hidden_size=self.cfg_data["hidden_size"],
            intermediate_size=self.cfg_data["intermediate_size"],
            num_hidden_layers=self.cfg_data["num_hidden_layers"],
            num_charts=self.cfg_data["num_charts"],
            layers_per_chart=self.cfg_data["layers_per_chart"],
            r_base=self.cfg_data["r_base"],
            r_corr=self.cfg_data["r_corr"]
        )

        hf_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.8-27B-FP8") if hasattr(AutoConfig, "from_pretrained") else None
        self.text_config = getattr(hf_cfg, "text_config", hf_cfg)
        self.text_config.intermediate_size = self.config.intermediate_size

        # Carregar componentes de entrada e saída (outside.safetensors)
        outside_path = os.path.join(model_dir, "outside.safetensors")
        with st.safe_open(outside_path, framework="pt") as f:
            self.embed_w = f.get_tensor("model.language_model.embed_tokens.weight")
            self.norm_w = f.get_tensor("model.language_model.norm.weight").to(device=self.device, dtype=torch.bfloat16)
            self.lm_head_w = f.get_tensor("lm_head.weight").to(device=self.device, dtype=torch.bfloat16)

        self.norm_mod = Qwen3_5RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(device=self.device, dtype=torch.bfloat16)
        self.norm_mod.weight.data.copy_(self.norm_w)
        self.rotary = Qwen3_5TextRotaryEmbedding(self.text_config).to(self.device)

        # Carregar estabilizadores persistidos
        self.stabilizers = {}
        stab_path = os.path.join(model_dir, "stabilizers.pt")
        if os.path.exists(stab_path):
            self.load_stabilizers(stab_path)

    def load_stabilizers(self, filepath: str):
        state = torch.load(filepath, map_location=self.device)
        layers = set()
        for k in state.keys():
            if k.startswith("stabilizer."):
                layers.add(int(k.split(".")[1]))

        for l in sorted(layers):
            stype = state[f"stabilizer.{l}.type"]
            W_down = state[f"stabilizer.{l}.W_down"].to(device=self.device, dtype=torch.bfloat16)
            W_up = state[f"stabilizer.{l}.W_up"].to(device=self.device, dtype=torch.bfloat16)
            if stype == "linear":
                stab = LinearResidualStabilizer(W_down, W_up)
            elif stype == "nonlinear":
                delta_alpha = state[f"stabilizer.{l}.delta_alpha"].item()
                stab = NonLinearResidualStabilizer(W_down, W_up, delta_alpha=delta_alpha)
            else:
                raise ValueError(f"Tipo desconhecido de estabilizador: {stype}")
            stab.freeze()
            self.stabilizers[l] = stab

    def forward_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = self.device
        num_seqs, seq_len = input_ids.shape

        pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
        pos_emb = self.rotary(torch.zeros(num_seqs, seq_len, self.config.hidden_size, device=device, dtype=torch.bfloat16), pos_ids)

        x = F.embedding(input_ids.cpu(), self.embed_w).to(device=device, dtype=torch.bfloat16)

        charts_dir = os.path.join(self.model_dir, "charts")

        for chart_idx in range(self.config.num_charts):
            chart_file = os.path.join(charts_dir, f"chart_{chart_idx:02d}.pt")
            chart_data = torch.load(chart_file, map_location=device)

            U_m = chart_data["bases"]["U_mix"].to(device=device, dtype=torch.bfloat16)
            U_d = chart_data["bases"]["U_down"].to(device=device, dtype=torch.bfloat16)
            V_j = chart_data["bases"]["V_joint"].to(device=device, dtype=torch.bfloat16)

            start_l = chart_idx * self.config.layers_per_chart
            end_l = start_l + self.config.layers_per_chart

            for l in range(start_l, end_l):
                l_data = chart_data["layers"][l]
                factors_fp8 = l_data["factors_fp8"]
                sub = l_data["submodules"]
                is_attn = l_data["is_attn"]

                layer_mod = Qwen3_5DecoderLayer(self.text_config, layer_idx=l).to(device=device, dtype=torch.bfloat16)

                # Carregar layernorms
                layer_mod.input_layernorm.weight.data.copy_(sub["input_ln"].to(device=device, dtype=torch.bfloat16))
                layer_mod.post_attention_layernorm.weight.data.copy_(sub["post_ln"].to(device=device, dtype=torch.bfloat16))

                # Desquantizar fatores FP8 bloco-128
                C_mix = dequant(factors_fp8["C_mix"].to(device), factors_fp8["C_mix_s"].to(device)).to(dtype=torch.bfloat16)
                C_gate = dequant(factors_fp8["C_gate"].to(device), factors_fp8["C_gate_s"].to(device)).to(dtype=torch.bfloat16)
                C_up = dequant(factors_fp8["C_up"].to(device), factors_fp8["C_up_s"].to(device)).to(dtype=torch.bfloat16)
                C_down = dequant(factors_fp8["C_down"].to(device), factors_fp8["C_down_s"].to(device)).to(dtype=torch.bfloat16)

                if is_attn:
                    layer_mod.self_attn.q_proj.weight.data.copy_(dequant(sub["q_w"].to(device), sub["q_s"].to(device)).to(dtype=torch.bfloat16))
                    layer_mod.self_attn.k_proj.weight.data.copy_(dequant(sub["k_w"].to(device), sub["k_s"].to(device)).to(dtype=torch.bfloat16))
                    layer_mod.self_attn.v_proj.weight.data.copy_(dequant(sub["v_w"].to(device), sub["v_s"].to(device)).to(dtype=torch.bfloat16))
                    layer_mod.self_attn.q_norm.weight.data.copy_(sub["q_norm"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.self_attn.k_norm.weight.data.copy_(sub["k_norm"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.self_attn.o_proj.weight.data.copy_(torch.matmul(U_m, C_mix))
                else:
                    layer_mod.linear_attn.in_proj_qkv.weight.data.copy_(dequant(sub["qkv_w"].to(device), sub["qkv_s"].to(device)).to(dtype=torch.bfloat16))
                    layer_mod.linear_attn.in_proj_z.weight.data.copy_(dequant(sub["z_w"].to(device), sub["z_s"].to(device)).to(dtype=torch.bfloat16))
                    layer_mod.linear_attn.in_proj_a.weight.data.copy_(sub["in_proj_a"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.in_proj_b.weight.data.copy_(sub["in_proj_b"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.conv1d.weight.data.copy_(sub["conv1d"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.dt_bias.data.copy_(sub["dt_bias"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.A_log.data.copy_(sub["A_log"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.norm.weight.data.copy_(sub["norm"].to(device=device, dtype=torch.bfloat16))
                    layer_mod.linear_attn.out_proj.weight.data.copy_(torch.matmul(U_m, C_mix))

                # Reconstruir pesos projetados do MLP
                layer_mod.mlp.gate_proj.weight.data.copy_(torch.matmul(C_gate, V_j.t()))
                layer_mod.mlp.up_proj.weight.data.copy_(torch.matmul(C_up, V_j.t()))
                layer_mod.mlp.down_proj.weight.data.copy_(torch.matmul(U_d, C_down))

                with torch.no_grad():
                    out = layer_mod(x, position_embeddings=pos_emb)
                    x = out[0] if isinstance(out, tuple) else out

                # Aplicar estabilizador residual persistido
                if l in self.stabilizers:
                    with torch.no_grad():
                        x = self.stabilizers[l](x)

                del layer_mod
                torch.cuda.empty_cache()

            del chart_data, U_m, U_d, V_j
            torch.cuda.empty_cache()

        with torch.no_grad():
            x_norm = self.norm_mod(x)
            logits = F.linear(x_norm, self.lm_head_w)

        return logits
