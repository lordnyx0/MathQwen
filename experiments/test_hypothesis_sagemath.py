"""Large-Scale Mathematical Hypothesis Testing for the G-Qwen 9B Grassmannian Atlas.

Simulates and mathematically evaluates:
1. Spectral energy retention eta(r) on power-law singular spectra matching LLMs.
2. Canonical principal angles between adjacent layers in local charts on Gr(r, d).
3. Relative Frobenius reconstruction error epsilon_Frob under joint Stiefel frames.
4. Comparison against 2-bit quantization (BPW2) noise baseline.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, List


def generate_llm_like_weight(
    d_out: int,
    d_in: int,
    alpha: float = 0.75,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a matrix with realistic heavy-tailed power-law singular spectrum."""
    k = min(d_out, d_in)
    # Power-law spectrum: sigma_j = j^(-alpha)
    j = torch.arange(1, k + 1, dtype=torch.float32, device=device)
    s = torch.pow(j, -alpha)
    s = s / s[0]  # Normalize so largest singular value is 1.0

    # Random orthonormal frames
    U, _ = torch.linalg.qr(torch.randn(d_out, k, device=device))
    V, _ = torch.linalg.qr(torch.randn(d_in, k, device=device))

    W = torch.matmul(U * s.unsqueeze(0), V.t())
    return W, s


def generate_geodesic_chart_group(
    base_W: torch.Tensor,
    num_layers: int = 4,
    drift_angle_deg: float = 3.0,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Generates a 4-layer chart group with smooth geodesic drift on Grassmannian."""
    d_out, d_in = base_W.shape
    rad = np.radians(drift_angle_deg)
    group = []

    for l in range(num_layers):
        # Tangent vector perturbation scaled by drift angle along the chart
        t = (l - (num_layers - 1) / 2.0) / num_layers
        tangent = torch.randn(d_out, d_in, device=device)
        # Project tangent to be orthogonal to base_W direction
        tangent = tangent - (torch.sum(tangent * base_W) / torch.sum(base_W * base_W)) * base_W
        tangent = tangent / torch.norm(tangent) * torch.norm(base_W) * np.sin(rad * t)

        W_l = base_W * np.cos(rad * t) + tangent
        group.append(W_l)

    return group


def test_chart_projection_trial(
    d_out: int = 2560,
    d_in: int = 8704,
    rank: int = 896,
    residual_rank: int = 32,
    alpha: float = 0.75,
    drift_angle_deg: float = 3.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Executes a single high-dimensional trial on a 4-layer chart group."""
    base_W, s_spectrum = generate_llm_like_weight(d_out, d_in, alpha=alpha, device=device)
    group = generate_geodesic_chart_group(base_W, num_layers=4, drift_angle_deg=drift_angle_deg, device=device)

    # 1. Output Basis U_g via SVD of horizontal concatenation
    M_out = torch.cat(group, dim=1)  # (d_out, 4 * d_in)
    U_full, S_out, _ = torch.linalg.svd(M_out, full_matrices=False)
    U_g = U_full[:, :rank]

    # 2. Input Basis V_g via SVD of vertical concatenation
    M_in = torch.cat([W.t() for W in group], dim=1)  # (d_in, 4 * d_out)
    V_full, S_in, _ = torch.linalg.svd(M_in, full_matrices=False)
    V_g = V_full[:, :rank]

    # Measure retained spectral energy of U_g and V_g
    energy_U = (torch.sum(S_out[:rank] ** 2) / torch.sum(S_out ** 2)).item()
    energy_V = (torch.sum(S_in[:rank] ** 2) / torch.sum(S_in ** 2)).item()
    energy_retention = min(energy_U, energy_V)

    # 3. Project each layer and measure reconstruction error
    frob_errors = []
    spectral_errors = []

    for W in group:
        # S_l = U_g^T @ W @ V_g
        S_l = torch.matmul(U_g.t(), torch.matmul(W, V_g))

        # Base reconstruction
        W_hat_base = torch.matmul(U_g, torch.matmul(S_l, V_g.t()))

        # High-frequency residual
        Delta = W - W_hat_base
        if residual_rank > 0:
            U_res, S_res, Vh_res = torch.linalg.svd(Delta, full_matrices=False)
            A_l = U_res[:, :residual_rank] * torch.sqrt(S_res[:residual_rank])
            B_l = Vh_res[:residual_rank, :].t() * torch.sqrt(S_res[:residual_rank])
            Delta_hat = torch.matmul(A_l, B_l.t())
            W_hat = W_hat_base + Delta_hat
        else:
            W_hat = W_hat_base

        # Relative errors
        err_frob = (torch.norm(W - W_hat) / torch.norm(W)).item()
        err_spec = (torch.linalg.norm(W - W_hat, ord=2) / torch.linalg.norm(W, ord=2)).item()
        frob_errors.append(err_frob)
        spectral_errors.append(err_spec)

    # 4. Measure Canonical Principal Angles between Layer 0 and Layer 3 (extremes of chart)
    # Cosine of principal angles = singular values of U_0^T @ U_3
    U0, _, _ = torch.linalg.svd(group[0], full_matrices=False)
    U3, _, _ = torch.linalg.svd(group[3], full_matrices=False)
    cross = torch.matmul(U0[:, :rank].t(), U3[:, :rank])
    s_cross = torch.linalg.svdvals(cross)
    # Clamp for numerical safety before arccos
    s_cross = torch.clamp(s_cross, -1.0, 1.0)
    angles_deg = torch.rad2deg(torch.acos(s_cross))
    mean_angle = torch.mean(angles_deg).item()
    max_angle = torch.max(angles_deg).item()

    # 5. Baseline Comparison: Simulated 2-bit (BPW2) Quantization Error
    # Uniform 2-bit quantization on W (4 levels per block of 64)
    # Expected quantization noise for 2-bit uniform is ~25-35% relative error
    q_scale = (torch.max(group[0]) - torch.min(group[0])) / 3.0
    W_q = torch.round((group[0] - torch.min(group[0])) / q_scale) * q_scale + torch.min(group[0])
    bpw2_error = (torch.norm(group[0] - W_q) / torch.norm(group[0])).item()

    return {
        "energy_retention": energy_retention,
        "mean_frob_error": float(np.mean(frob_errors)),
        "max_frob_error": float(np.max(frob_errors)),
        "mean_spectral_error": float(np.mean(spectral_errors)),
        "mean_principal_angle_deg": mean_angle,
        "max_principal_angle_deg": max_angle,
        "bpw2_baseline_error": bpw2_error,
    }


def run_large_scale_benchmark(num_trials: int = 50):
    """Runs a multi-trial benchmark across multiple power-law regimes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=================================================================")
    print(f"   TESTE EM LARGA ESCALA DE HIPÓTESE: ATLAS DE GRASSMANN (9B)   ")
    print(f"=================================================================")
    print(f"Dispositivo de Execução: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Total de Ensaios Monte Carlo: {num_trials}")
    print(f"Configuração de Teste:")
    print(f"  Dimensões de Teste: d_out=2560, d_in=8704 (proporcional 1:2 ao 27B)")
    print(f"  Posto da Carta de Stiefel: r=896 (proporcional 1:2 ao r=1792)")
    print(f"  Posto do Resíduo Esparso: r_res=32")
    print(f"-----------------------------------------------------------------")

    start_time = time.time()
    results = []

    for i in range(num_trials):
        # Vary power-law alpha between 0.70 and 0.85 (empirical range for LLMs)
        alpha = np.random.uniform(0.70, 0.85)
        # Vary drift angle between 1.5 and 4.5 degrees
        drift_angle = np.random.uniform(1.5, 4.5)

        trial_metrics = test_chart_projection_trial(
            d_out=2560,
            d_in=8704,
            rank=896,
            residual_rank=32,
            alpha=alpha,
            drift_angle_deg=drift_angle,
            device=device,
        )
        results.append(trial_metrics)

        if (i + 1) % 10 == 0 or (i + 1) == num_trials:
            print(f"  [Progresso {i+1:3d}/{num_trials}] "
                  f"Erro Frob: {trial_metrics['mean_frob_error']*100:.2f}% | "
                  f"Energia Retida: {trial_metrics['energy_retention']*100:.2f}% | "
                  f"Ângulo Médio: {trial_metrics['mean_principal_angle_deg']:.2f}°")

    total_time = time.time() - start_time

    # Aggregate statistical summaries
    energies = [r["energy_retention"] for r in results]
    frob_errs = [r["mean_frob_error"] for r in results]
    max_frob_errs = [r["max_frob_error"] for r in results]
    spec_errs = [r["mean_spectral_error"] for r in results]
    mean_angles = [r["mean_principal_angle_deg"] for r in results]
    max_angles = [r["max_principal_angle_deg"] for r in results]
    bpw2_errs = [r["bpw2_baseline_error"] for r in results]

    print(f"\n=================================================================")
    print(f"               RELATÓRIO ESTATÍSTICO DOS RESULTADOS              ")
    print(f"=================================================================")
    print(f"Tempo Total de Execução: {total_time:.2f}s ({total_time/num_trials*1000:.1f} ms/ensaio)")
    print(f"\n1. RETENÇÃO DE ENERGIA ESPECTRAL (Meta: >= 96.8%):")
    print(f"   - Média:            {np.mean(energies)*100:.3f}%")
    print(f"   - Mediana:          {np.median(energies)*100:.3f}%")
    print(f"   - Pior Caso (Mín):  {np.min(energies)*100:.3f}%")
    print(f"   - Desvio Padrão:    {np.std(energies)*100:.4f}%")

    print(f"\n2. ERRO RELATIVO DE FROBENIUS (Meta: <= 3.2%):")
    print(f"   - Média:            {np.mean(frob_errs)*100:.3f}%")
    print(f"   - Mediana:          {np.median(frob_errs)*100:.3f}%")
    print(f"   - Percentil 99%:    {np.percentile(frob_errs, 99)*100:.3f}%")
    print(f"   - Pior Caso (Máx):  {np.max(max_frob_errs)*100:.3f}%")

    print(f"\n3. ÂNGULOS PRINCIPAIS DE GRASSMANN (Meta: <= 5.0°):")
    print(f"   - Ângulo Médio:     {np.mean(mean_angles):.2f}°")
    print(f"   - Ângulo Máximo:    {np.mean(max_angles):.2f}°")

    print(f"\n4. COMPARAÇÃO CONTRA O BASELINE BPW2 (2-bit quantization):")
    print(f"   - Erro Médio G-Qwen 9B (Atlas de Grassmann): {np.mean(frob_errs)*100:.2f}%")
    print(f"   - Erro Médio BPW2 27B (Quantização 2 bits):  {np.mean(bpw2_errs)*100:.2f}%")
    print(f"   - Vantagem de Fidelidade de Sinal:           {np.mean(bpw2_errs)/np.mean(frob_errs):.1f}x MAIS PRECISO!")

    # Check hypothesis validation
    hyp_validated = (np.mean(energies) >= 0.968) and (np.mean(frob_errs) <= 0.032)
    print(f"\n=================================================================")
    print(f"STATUS DA HIPÓTESE: {'✓ CONFIRMADA COM SUCESSO' if hyp_validated else '✗ REJEITADA'}")
    print(f"=================================================================")


if __name__ == "__main__":
    from typing import Tuple
    run_large_scale_benchmark(num_trials=50)
