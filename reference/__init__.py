# -*- coding: utf-8 -*-
"""Pacote canônico de referência: Qwen oficial."""
from .loader import load_qwen_reference_components, dequant, load_layer_module
from .causal_stream import run_official_causal_stream

__all__ = ["load_qwen_reference_components", "dequant", "load_layer_module", "run_official_causal_stream"]
