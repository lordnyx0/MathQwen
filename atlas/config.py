# -*- coding: utf-8 -*-
"""Configuração canônica oficial do Atlas MathQwen."""
from dataclasses import dataclass

@dataclass
class AtlasConfig:
    model_name: str = "Qwen/Qwen3.8-27B-FP8"
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17920
    num_hidden_layers: int = 64
    num_charts: int = 16
    layers_per_chart: int = 4
    r_base: int = 2048          # Posto do subespaço compartilhado Atlas
    r_corr: int = 64            # Posto do estabilizador residual compacto
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6
