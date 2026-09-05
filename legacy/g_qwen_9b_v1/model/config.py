"""Configuration for the G-Qwen 9B (Grassmannian Foliated Atlas) Architecture.

Faithfully represents Qwen 3.8 27B weights using a 9B parameter budget
grounded in differential geometry on Grassmann and Stiefel manifolds.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class AtlasConfig:
    """Hyperparameters for the Bipartite Grassmannian Atlas and Foliated Sub-bundles."""
    num_charts: int = 16
    layers_per_chart: int = 4
    chart_rank_ffn: int = 1792
    chart_rank_deltanet: int = 1024
    chart_rank_attn: int = 2048
    num_attn_macro_charts: int = 4
    residual_rank: int = 64
    use_stiefel_bases: bool = True
    use_sparse_residuals: bool = True


@dataclass
class GQwen9BConfig:
    """Architectural configuration for G-Qwen 9B model."""

    # Dimensional Parameters matching Qwen 3.8 27B
    vocab_size: int = 152064
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    max_position_embeddings: int = 262144
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    # Gated DeltaNet (Linear Attention) Sublayer Parameters (48 layers)
    deltanet_v_heads: int = 48
    deltanet_qk_heads: int = 16
    deltanet_head_dim: int = 128
    deltanet_conv_kernel_size: int = 4

    # Gated Attention (Softmax Attention) Sublayer Parameters (16 layers)
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    attention_head_dim: int = 256
    rope_dim: int = 64

    # Grassmannian Atlas Configuration
    atlas: AtlasConfig = field(default_factory=AtlasConfig)

    # Layer Types: Exactly 16 groups of [DeltaNet, DeltaNet, DeltaNet, Attention] = 64 layers
    layer_types: List[Literal["deltanet", "attention"]] = field(default_factory=lambda: [
        layer_type
        for _ in range(16)
        for layer_type in ["deltanet", "deltanet", "deltanet", "attention"]
    ])

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)
