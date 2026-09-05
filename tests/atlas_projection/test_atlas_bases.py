# -*- coding: utf-8 -*-
"""Teste unitário da ortogonalidade e álgebra das bases do Atlas Assimétrico."""
import sys
import os
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atlas.projection import compute_chart_atlas_bases, project_layer_weights

def test_atlas_bases_orthonormality():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("TESTE UNITÁRIO: ORTONORMALIDADE DAS BASES ATLAS (V_joint, U_mix, U_down)")
    print("=" * 80)

    # Simular 4 camadas de uma carta
    chart_weights = {}
    for l in range(4):
        chart_weights[l] = {
            "mix": torch.randn(5120, 5120, device=device, dtype=torch.bfloat16),
            "gate": torch.randn(17920, 5120, device=device, dtype=torch.bfloat16),
            "up": torch.randn(17920, 5120, device=device, dtype=torch.bfloat16),
            "down": torch.randn(5120, 17920, device=device, dtype=torch.bfloat16),
            "is_attn": (l == 3)
        }

    r_base = 2048
    bases = compute_chart_atlas_bases(chart_weights, r_base=r_base, device=device)

    # Verificar shapes
    assert bases["U_mix"].shape == (5120, r_base), f"Shape inesperado U_mix: {bases['U_mix'].shape}"
    assert bases["U_down"].shape == (5120, r_base), f"Shape inesperado U_down: {bases['U_down'].shape}"
    assert bases["V_joint"].shape == (5120, r_base), f"Shape inesperado V_joint: {bases['V_joint'].shape}"

    # Verificar ortonormalidade: U^T U = I_k
    eye_k = torch.eye(r_base, device=device, dtype=torch.float32)

    err_mat_u = torch.matmul(bases["U_mix"].float().t(), bases["U_mix"].float()) - eye_k
    err_mat_d = torch.matmul(bases["U_down"].float().t(), bases["U_down"].float()) - eye_k
    err_mat_v = torch.matmul(bases["V_joint"].float().t(), bases["V_joint"].float()) - eye_k

    max_u = err_mat_u.abs().max().item()
    max_d = err_mat_d.abs().max().item()
    max_v = err_mat_v.abs().max().item()

    frob_norm_u = torch.norm(err_mat_u).item() / r_base

    print(f"  -> Erro maximo pontual |U_mix^T U_mix - I|:   {max_u:.4e} (Frob norm normalizado: {frob_norm_u:.4e})")
    print(f"  -> Erro maximo pontual |U_down^T U_down - I|: {max_d:.4e}")
    print(f"  -> Erro maximo pontual |V_joint^T V_joint - I|: {max_v:.4e}")

    assert max_u < 1e-2, f"Erro pontual maximo de U_mix excedeu 0.01: {max_u}"
    assert max_d < 1e-2, f"Erro pontual maximo de U_down excedeu 0.01: {max_d}"
    assert max_v < 1e-2, f"Erro pontual maximo de V_joint excedeu 0.01: {max_v}"

    # Testar projeção de pesos
    p_w = project_layer_weights(chart_weights[0], bases)
    assert p_w["mix"].shape == (5120, 5120)
    assert p_w["gate"].shape == (17920, 5120)
    assert p_w["up"].shape == (17920, 5120)
    assert p_w["down"].shape == (5120, 17920)

    print("\n[OK] TESTE UNITÁRIO PASSOU: As bases do Atlas são rigorosamente ortonormais!")
    print("=" * 80)

if __name__ == "__main__":
    test_atlas_bases_orthonormality()
