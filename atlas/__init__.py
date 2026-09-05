# -*- coding: utf-8 -*-
"""Pacote canônico oficial do Atlas Assimétrico e Estabilizadores Residuais MathQwen."""
from .config import AtlasConfig
from .projection import compute_chart_atlas_bases, project_layer_weights
from .residual import LinearResidualStabilizer, NonLinearResidualStabilizer, fit_svd_stabilizer, fit_gelu_warmstart
from .atlas_model import AtlasStreamModel

__all__ = [
    "AtlasConfig",
    "compute_chart_atlas_bases",
    "project_layer_weights",
    "LinearResidualStabilizer",
    "NonLinearResidualStabilizer",
    "fit_svd_stabilizer",
    "fit_gelu_warmstart",
    "AtlasStreamModel"
]
