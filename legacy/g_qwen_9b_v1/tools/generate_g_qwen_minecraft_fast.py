# -*- coding: utf-8 -*-
"""Compatibility shim forwarding to inference.generate_g_qwen_minecraft_fast."""
import sys
import os

# Adiciona raiz ao sys.path se necessario
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from inference.generate_g_qwen_minecraft_fast import *

if __name__ == '__main__':
    import runpy
    runpy.run_module('inference.generate_g_qwen_minecraft_fast', run_name='__main__')
