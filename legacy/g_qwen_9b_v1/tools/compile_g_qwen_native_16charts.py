# -*- coding: utf-8 -*-
"""Compatibility shim forwarding to compilers.compile_g_qwen_native_16charts."""
import sys
import os

# Adiciona raiz ao sys.path se necessario
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from compilers.compile_g_qwen_native_16charts import *

if __name__ == '__main__':
    import runpy
    runpy.run_module('compilers.compile_g_qwen_native_16charts', run_name='__main__')
