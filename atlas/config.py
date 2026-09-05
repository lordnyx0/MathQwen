# -*- coding: utf-8 -*-
"""Configuração canônica oficial do Atlas MathQwen."""
from dataclasses import dataclass

@dataclass
class AtlasConfig:
    model_name: str = "Qwen/Qwen3.8-27B-FP8"
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408  # d_ff oficial do Qwen3.8-27B
    num_hidden_layers: int = 64
    num_charts: int = 16
    layers_per_chart: int = 4
    r_base: int = 2048          # Posto do subespaço compartilhado Atlas
    r_corr: int = 64            # Posto do estabilizador residual compacto
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_qwen_config(cls, cfg, r_base: int = 2048, r_corr: int = 64):
        """Deriva a configuração dinamicamente a partir do AutoConfig oficial do HuggingFace."""
        text_cfg = getattr(cfg, "text_config", cfg)
        return cls(
            model_name=getattr(cfg, "_name_or_path", "Qwen/Qwen3.8-27B-FP8"),
            vocab_size=text_cfg.vocab_size,
            hidden_size=text_cfg.hidden_size,
            intermediate_size=text_cfg.intermediate_size,
            num_hidden_layers=text_cfg.num_hidden_layers,
            num_charts=16,
            layers_per_chart=text_cfg.num_hidden_layers // 16,
            r_base=r_base,
            r_corr=r_corr,
            tie_word_embeddings=getattr(text_cfg, "tie_word_embeddings", False),
            rms_norm_eps=getattr(text_cfg, "rms_norm_eps", 1e-6)
        )
