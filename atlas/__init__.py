# -*- coding: utf-8 -*-
"""Pacote canônico oficial do Atlas Assimétrico e Estabilizadores Residuais MathQwen."""
from .config import AtlasConfig
from .projection import compute_chart_atlas_bases, project_layer_weights
from .residual import (
    LinearResidualStabilizer,
    NonLinearResidualStabilizer,
    fit_svd_stabilizer,
    fit_gelu_warmstart,
    freeze_backbone_and_isolate_stabilizer
)
from .atlas_model import AtlasStreamModel
from .calibration import calibrate_offline_stabilizers
from .export import export_autonomous_atlas_model
from .autonomous_model import AtlasAutonomousModel

__all__ = [
    "AtlasConfig",
    "compute_chart_atlas_bases",
    "project_layer_weights",
    "LinearResidualStabilizer",
    "NonLinearResidualStabilizer",
    "fit_svd_stabilizer",
    "fit_gelu_warmstart",
    "freeze_backbone_and_isolate_stabilizer",
    "AtlasStreamModel",
    "calibrate_offline_stabilizers",
    "export_autonomous_atlas_model",
    "AtlasAutonomousModel"
]
