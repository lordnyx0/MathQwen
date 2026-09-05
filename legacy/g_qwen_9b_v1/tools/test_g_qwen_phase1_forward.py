# -*- coding: utf-8 -*-
"""Compatibility shim forwarding to experiments.test_g_qwen_phase1_forward."""
import sys
import os

# Adiciona raiz ao sys.path se necessario
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from experiments.test_g_qwen_phase1_forward import *

if __name__ == '__main__':
    import runpy
    runpy.run_module('experiments.test_g_qwen_phase1_forward', run_name='__main__')
