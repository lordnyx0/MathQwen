# -*- coding: utf-8 -*-
"""Modelo canônico de inferência e streaming do Atlas Assimétrico com estabilização residual."""
import os
import torch
import torch.nn.functional as F
import safetensors.torch as st
from .config import AtlasConfig
from .projection import compute_chart_atlas_bases, project_layer_weights
from .residual import LinearResidualStabilizer, NonLinearResidualStabilizer
from reference.loader import load_qwen_reference_components, load_layer_module, dequant

class AtlasStreamModel:
    def __init__(self, config: AtlasConfig = None, device=None):
        self.config = config or AtlasConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.comp = load_qwen_reference_components(device=self.device)
        self.stabilizers = {}
        self.bases_cache = {}

    def set_stabilizer(self, layer_idx: int, stabilizer):
        self.stabilizers[layer_idx] = stabilizer

    def freeze_backbone(self):
        """Assegura formalmente que todos os pesos do professor e normas estão congelados (requires_grad=False)."""
        self.comp["embed_w"].requires_grad = False
        self.comp["lm_head_w"].requires_grad = False
        for p in self.comp["norm_mod"].parameters():
            p.requires_grad = False
        for p in self.comp["rotary"].parameters():
            p.requires_grad = False

    def precompute_and_cache_chart_bases(self, snapshot_dir=None):
        """Pré-computa as bases Atlas (U_mix, U_down, V_joint) de todas as 16 cartas uma única vez."""
        if snapshot_dir is None:
            snapshot_dir = self.comp["snapshot_dir"]
        device = self.device
        for chart_idx in range(self.config.num_charts):
            start_l = chart_idx * self.config.layers_per_chart
            end_l = start_l + self.config.layers_per_chart
            chart_weights = {}
            for l in range(start_l, end_l):
                lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
                with st.safe_open(lp, framework="pt") as f:
                    is_attn = (l % 4 == 3)
                    if is_attn:
                        mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    else:
                        mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    chart_weights[l] = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}
            self.bases_cache[chart_idx] = compute_chart_atlas_bases(chart_weights, r_base=self.config.r_base, device=device)
        print(f"[OK] Cache de bases Atlas construído para todas as {self.config.num_charts} cartas!")

    def save_bases(self, filepath: str):
        """Persiste em disco as bases Atlas pré-computadas."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.bases_cache, filepath)
        print(f"[OK] Bases Atlas salvas com sucesso em: {filepath}")

    def load_bases(self, filepath: str):
        """Carrega do disco as bases Atlas pré-computadas."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de bases não encontrado: {filepath}")
        raw_cache = torch.load(filepath, map_location=self.device)
        self.bases_cache = {
            c_idx: {k: v.to(device=self.device, dtype=torch.bfloat16) for k, v in b.items()}
            for c_idx, b in raw_cache.items()
        }
        print(f"[OK] Bases Atlas carregadas de: {filepath} ({len(self.bases_cache)} cartas)")

    def save_stabilizers(self, filepath: str):
        """Persiste em disco os 64 estabilizadores residuais previamente calibrados."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {}
        for l, stab in self.stabilizers.items():
            if isinstance(stab, LinearResidualStabilizer):
                state[f"stabilizer.{l}.type"] = "linear"
                state[f"stabilizer.{l}.W_down"] = stab.W_down.data.cpu()
                state[f"stabilizer.{l}.W_up"] = stab.W_up.data.cpu()
            elif isinstance(stab, NonLinearResidualStabilizer):
                state[f"stabilizer.{l}.type"] = "nonlinear"
                state[f"stabilizer.{l}.W_down"] = stab.W_down.data.cpu()
                state[f"stabilizer.{l}.W_up"] = stab.W_up.data.cpu()
                state[f"stabilizer.{l}.delta_alpha"] = stab.delta_alpha.data.cpu()
        torch.save(state, filepath)
        print(f"[OK] {len(self.stabilizers)} estabilizadores residuais salvos em: {filepath}")

    def load_stabilizers(self, filepath: str):
        """Carrega do disco os estabilizadores residuais calibrados offline."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de estabilizadores não encontrado: {filepath}")
        state = torch.load(filepath, map_location=self.device)
        layers = set()
        for k in state.keys():
            if k.startswith("stabilizer."):
                l = int(k.split(".")[1])
                layers.add(l)

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

        print(f"[OK] {len(self.stabilizers)} estabilizadores carregados com sucesso de: {filepath}")

    def forward_tokens(self, input_ids: torch.Tensor, use_atlas=True):
        """Executa a inferência causal pura pelas 64 camadas."""
        device = self.device
        num_seqs, seq_len = input_ids.shape
        cfg = self.comp["cfg"]
        snapshot_dir = self.comp["snapshot_dir"]

        pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, num_seqs, -1)
        pos_emb = self.comp["rotary"](torch.zeros(num_seqs, seq_len, 5120, device=device, dtype=torch.bfloat16), pos_ids)

        x = F.embedding(input_ids.cpu(), self.comp["embed_w"]).to(device=device, dtype=torch.bfloat16)

        for chart_idx in range(self.config.num_charts):
            start_l = chart_idx * self.config.layers_per_chart
            end_l = start_l + self.config.layers_per_chart

            chart_weights = {}
            for l in range(start_l, end_l):
                lp = os.path.join(snapshot_dir, f"layers-{l}.safetensors")
                with st.safe_open(lp, framework="pt") as f:
                    is_attn = (l % 4 == 3)
                    if is_attn:
                        mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.self_attn.o_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    else:
                        mix_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.linear_attn.out_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    gate_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.gate_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    up_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.up_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    down_w = dequant(f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight"), f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight_scale_inv")).to(device=device, dtype=torch.bfloat16)
                    chart_weights[l] = {"mix": mix_w, "gate": gate_w, "up": up_w, "down": down_w, "is_attn": is_attn}

            if use_atlas:
                if chart_idx in self.bases_cache:
                    bases = self.bases_cache[chart_idx]
                else:
                    bases = compute_chart_atlas_bases(chart_weights, r_base=self.config.r_base, device=device)
                    self.bases_cache[chart_idx] = bases

            for l in range(start_l, end_l):
                layer_mod = load_layer_module(l, cfg, device, snapshot_dir=snapshot_dir)

                if use_atlas:
                    p_w = project_layer_weights(chart_weights[l], bases)
                    if chart_weights[l]["is_attn"]:
                        layer_mod.self_attn.o_proj.weight.data.copy_(p_w["mix"])
                    else:
                        layer_mod.linear_attn.out_proj.weight.data.copy_(p_w["mix"])
                    layer_mod.mlp.gate_proj.weight.data.copy_(p_w["gate"])
                    layer_mod.mlp.up_proj.weight.data.copy_(p_w["up"])
                    layer_mod.mlp.down_proj.weight.data.copy_(p_w["down"])

                with torch.no_grad():
                    out = layer_mod(x, position_embeddings=pos_emb)
                    x = out[0] if isinstance(out, tuple) else out

                # Aplicação direta do estabilizador persistido (sem fit na inferência!)
                if l in self.stabilizers:
                    with torch.no_grad():
                        x = self.stabilizers[l](x)

                del layer_mod
                torch.cuda.empty_cache()

        with torch.no_grad():
            x_norm = self.comp["norm_mod"](x)
            logits = F.linear(x_norm, self.comp["lm_head_w"])

        return logits
