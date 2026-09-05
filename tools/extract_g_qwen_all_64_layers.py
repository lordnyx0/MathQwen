# -*- coding: utf-8 -*-
"""Compatibility shim forwarding to pipeline.phase1_extraction.extract_g_qwen_all_64_layers."""
import sys
import os

# Adiciona raiz ao sys.path se necessario
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pipeline.phase1_extraction.extract_g_qwen_all_64_layers import *

if __name__ == '__main__':
    import runpy
    runpy.run_module('pipeline.phase1_extraction.extract_g_qwen_all_64_layers', run_name='__main__')
