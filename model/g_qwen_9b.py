"""Full PyTorch Model Implementation for G-Qwen 9B (Grassmannian Foliated Atlas).

Maintains all 64 physical layers of Qwen 3.8 27B within a strict 9B parameter budget
via 16 local topological charts:
- 16 ChartBases holding shared Stiefel frames U_g, V_g.
- 64 GQwenBlocks (48 Gated DeltaNet + 16 Gated Attention) with unique coordinates S_l.
- Tied Word Embeddings (152,064 x 5120).
- Pure linear 1x forward pass (no recurrent loops).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from model.config import GQwen9BConfig
from model.layers import ChartBases, GQwenBlock
from model.topological_modules import SheafChartNorm, StiefelChartBasis


class GQwen9BModel(nn.Module):
    """Core Transformer Backbone with 64 physical layers and Bipartite Grassmann Atlas."""

    def __init__(self, config: GQwen9BConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Macro-Charts for Attention (4 macro-charts over depth)
        self.num_attn_macro_charts = config.atlas.num_attn_macro_charts
        d_v_attn = config.num_attention_heads * config.attention_head_dim
        self.attn_macro_bases = nn.ModuleList([
            StiefelChartBasis(
                in_features=d_v_attn,
                out_features=config.hidden_size,
                rank=config.atlas.chart_rank_attn,
            )
            for _ in range(self.num_attn_macro_charts)
        ])

        # 16 Topological Chart Bases
        self.num_charts = config.atlas.num_charts
        self.layers_per_chart = config.atlas.layers_per_chart
        charts = []
        for g in range(self.num_charts):
            macro_idx = min(g // max(1, self.num_charts // self.num_attn_macro_charts), self.num_attn_macro_charts - 1)
            charts.append(ChartBases(config, attn_basis=self.attn_macro_bases[macro_idx]))
        self.charts = nn.ModuleList(charts)

        # 64 Distinct Layers
        self.layers = nn.ModuleList([
            GQwenBlock(
                config=config,
                chart_bases=self.charts[i // self.layers_per_chart],
                layer_type=config.layer_types[i],
            )
            for i in range(config.num_layers)
        ])

        # Final RMSNorm
        self.norm = SheafChartNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)

        # Pure causal 1x forward pass through all 64 layers
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        return h


class GQwen9BForCausalLM(nn.Module):
    """Full Causal Language Model with Tied LM Head and Parameter Accounting."""

    def __init__(self, config: Optional[GQwen9BConfig] = None):
        super().__init__()
        self.config = config or GQwen9BConfig()
        self.model = GQwen9BModel(self.config)

        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        if loss is not None:
            output["loss"] = loss

        return output

    def count_parameters(self) -> Dict[str, Any]:
        """Calculates exact parameter count broken down by charts, layers, and embeddings."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embed_params = self.model.embed_tokens.weight.numel()
        chart_bases_params = sum(p.numel() for p in self.model.charts.parameters())
        layers_params = sum(p.numel() for p in self.model.layers.parameters())

        return {
            "total_physical_parameters": total_params,
            "total_physical_parameters_in_billions": round(total_params / 1e9, 3),
            "trainable_parameters": trainable_params,
            "num_layers": self.config.num_layers,
            "num_charts": self.config.atlas.num_charts,
            "breakdown": {
                "embeddings": embed_params,
                "chart_bases_16": chart_bases_params,
                "layers_foliated_64": layers_params,
            },
        }
