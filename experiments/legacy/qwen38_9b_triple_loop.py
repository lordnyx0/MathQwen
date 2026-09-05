"""Full PyTorch Model Implementation for T^3-Qwen 9B (Topological Triple-Loop).

Fulfills the mission to faithfully represent Qwen 3.8 27B weights
with a strict 9B physical parameter budget via:
- 2 Prefix Layers (Physical 0, 1) -> Virtual 1, 2
- 20 Core Layers (Physical 2..21) looped 3x with Grassmann Adapters -> Virtual 3..62
- 2 Suffix Layers (Physical 22, 23) -> Virtual 63, 64
- Tied Word Embeddings (152,064 x 5120)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from model.config import Qwen38_9BTripleLoopConfig
from model.layers import HybridTransformerBlock
from model.topological_modules import (
    SheafConnectionNorm,
    BanachContractionGate,
)


class Qwen38_9BTripleLoopModel(nn.Module):
    """Core Transformer Backbone with Topological Triple Loop."""

    def __init__(self, config: Qwen38_9BTripleLoopConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Embeddings (tied with LM Head)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 1. Prefix physical layers (Layers 0, 1)
        self.prefix_layers = nn.ModuleList([
            HybridTransformerBlock(config, layer_type=config.layer_types[0], is_core=False),
            HybridTransformerBlock(config, layer_type=config.layer_types[1], is_core=False),
        ])

        # 2. Core physical layers (20 physical layers: idx 2..21)
        self.core_layers = nn.ModuleList([
            HybridTransformerBlock(config, layer_type=config.layer_types[i], is_core=True)
            for i in range(2, 22)
        ])

        # 3. Suffix physical layers (Layers 22, 23)
        self.suffix_layers = nn.ModuleList([
            HybridTransformerBlock(config, layer_type=config.layer_types[22], is_core=False),
            HybridTransformerBlock(config, layer_type=config.layer_types[23], is_core=False),
        ])

        # Banach Contraction and Context Injection Gate for the Triple Loop
        self.contraction_gate = BanachContractionGate(
            dim=config.hidden_size,
            alpha_init=config.topology.banach_alpha_init,
            beta_init=config.topology.banach_beta_init,
            num_loops=config.topology.num_loops,
        )

        # Final RMSNorm
        self.norm = SheafConnectionNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediate_loops: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # 1. Embedding lookup
        h = self.embed_tokens(input_ids)

        # 2. Prefix layers (Virtual layers 1, 2)
        for layer in self.prefix_layers:
            h = layer(h, loop_idx=None)

        h_prefix = h  # Anchored prefix context for Hamiltonian drive
        loop_states = []

        # 3. Recurrent Core Triple Loop (Virtual layers 3..62: 20 layers x 3 loops)
        for r in range(self.config.topology.num_loops):
            h_in_loop = h

            # Pass sequentially through the 20 physical core layers
            for layer in self.core_layers:
                h = layer(h, loop_idx=r)

            # Apply Banach Contraction Gate with prefix injection
            h = self.contraction_gate(
                h_current=h_in_loop,
                h_transformed=h,
                h_prefix=h_prefix,
                loop_idx=r,
            )

            if return_intermediate_loops:
                loop_states.append(h)

        # 4. Suffix layers (Virtual layers 63, 64)
        for layer in self.suffix_layers:
            h = layer(h, loop_idx=None)

        # 5. Final Normalization
        h = self.norm(h)

        return h, (loop_states if return_intermediate_loops else None)


class Qwen38_9BTripleLoopForCausalLM(nn.Module):
    """Full Causal Language Model with Tied LM Head and Parameter Accounting."""

    def __init__(self, config: Optional[Qwen38_9BTripleLoopConfig] = None):
        super().__init__()
        self.config = config or Qwen38_9BTripleLoopConfig()
        self.model = Qwen38_9BTripleLoopModel(self.config)

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
        return_intermediate_loops: bool = False,
    ) -> Dict[str, Any]:
        hidden_states, intermediate_loops = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_intermediate_loops=return_intermediate_loops,
        )

        # Compute logits via tied embeddings or independent head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction loss
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
        if intermediate_loops is not None:
            output["intermediate_loops"] = intermediate_loops

        return output

    def count_parameters(self) -> Dict[str, Any]:
        """Calculates precise physical parameter count vs virtual computational depth."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Breakdown by subsystem
        embed_params = self.model.embed_tokens.weight.numel()
        prefix_params = sum(p.numel() for p in self.model.prefix_layers.parameters())
        core_params = sum(p.numel() for p in self.model.core_layers.parameters())
        suffix_params = sum(p.numel() for p in self.model.suffix_layers.parameters())
        topological_params = sum(p.numel() for p in self.model.contraction_gate.parameters())

        return {
            "total_physical_parameters": total_params,
            "total_physical_parameters_in_billions": round(total_params / 1e9, 3),
            "trainable_parameters": trainable_params,
            "virtual_depth_layers": self.config.num_virtual_layers,
            "physical_depth_layers": self.config.num_physical_layers,
            "breakdown": {
                "embeddings": embed_params,
                "prefix_layers_2": prefix_params,
                "core_physical_layers_20": core_params,
                "suffix_layers_2": suffix_params,
                "contraction_gate": topological_params,
            },
        }
