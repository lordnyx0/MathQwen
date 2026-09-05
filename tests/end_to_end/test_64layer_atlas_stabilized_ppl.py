# -*- coding: utf-8 -*-
"""Alias para test_64layer_atlas_linear_stabilized_ppl.py para compatibilidade retroativa."""
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from test_64layer_atlas_linear_stabilized_ppl import run_64layer_end_to_end_benchmark

if __name__ == "__main__":
    run_64layer_end_to_end_benchmark()
