# -*- coding: utf-8 -*-
"""Sanity test for G-Qwen 9B Phase 3 (LoRA-Residual SVD)."""

import os
import glob
import torch
import safetensors.torch as st

def test_phase3():
    phase3_dir = os.path.abspath("models/g_qwen_9b_phase3")
    print("=" * 90)
    print("    TESTE DE SANIDADE: G-QWEN 9B (FASE 3 - LORA-RESIDUAL SVD)")
    print("=" * 90)

    # 1. Checar existencia de todas as 16 cartas
    charts = [os.path.join(phase3_dir, f"chart_{g}.safetensors") for g in range(16)]
    missing = [c for c in charts if not os.path.exists(c)]
    if missing:
        print(f"ERRO: Cartas ausentes: {missing}")
        return False

    total_size_mb = sum(os.path.getsize(c) for c in charts) / 1e6
    print(f"[OK] 16/16 Cartas presentes! Tamanho total: {total_size_mb:.1f} MB ({total_size_mb/1e3:.2f} GB)")

    # 2. Inspecionar tensores da Carta 0
    c0 = st.load_file(charts[0])
    assert "basis_mixer_U" in c0
    assert "basis_mixer_V" in c0
    assert "layer_0_mixer_S" in c0
    assert "layer_0_mixer_res_A" in c0
    assert "layer_0_mixer_res_B" in c0

    shape_A = c0["layer_0_mixer_res_A"].shape
    shape_B = c0["layer_0_mixer_res_B"].shape
    print(f"[OK] layer_0_mixer_res_A shape: {shape_A} (Esperado: [5120, 64])")
    print(f"[OK] layer_0_mixer_res_B shape: {shape_B} (Esperado: [6144, 64])")
    assert shape_A == (5120, 64)
    assert shape_B == (6144, 64)

    # 3. Teste de multiplicacao LoRA vetor-matriz
    x = torch.randn(1, 1, 6144, dtype=torch.float16)
    B = c0["layer_0_mixer_res_B"]
    A = c0["layer_0_mixer_res_A"]
    # (x @ B) @ A.T
    t = torch.matmul(x, B)
    y = torch.matmul(t, A.t())
    assert y.shape == (1, 1, 5120)
    print(f"[OK] Multiplicacao LoRA validada! Output shape: {y.shape}, norm: {y.norm().item():.3f}")

    print("=" * 90)
    print("TODOS OS TESTES DE SANIDADE DA FASE 3 PASSARAM COM SUCESSO!")
    print("=" * 90)
    return True

if __name__ == "__main__":
    test_phase3()
