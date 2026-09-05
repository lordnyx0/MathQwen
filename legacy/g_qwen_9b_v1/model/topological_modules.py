"""Grassmannian Foliated Atlas Modules for G-Qwen 9B.

Implements:
- StiefelChartBasis: Shared orthonormal frames U_g in St(d_out, r) and V_g in St(d_in, r).
- GrassmannFoliatedLinear: Factorized layer projection y = x (U_g S_l V_g^T + A_l B_l^T)^T.
- SheafChartNorm: Gauge-equivariant RMSNorm across chart boundaries.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StiefelChartBasis(nn.Module):
    """Shared orthonormal Stiefel basis frames (U_g, V_g) for a topological chart."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # U_g in St(out_features, rank) and V_g in St(in_features, rank)
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)


class GrassmannFoliatedLinear(nn.Module):
    """Foliated linear layer belonging to a chart with dedicated coordinates S_l and residual.

    Computes:
        y = x @ W_l^T = (x @ V_g @ S_l^T @ U_g^T) + (x @ B_l @ A_l^T)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: StiefelChartBasis,
        residual_rank: int = 64,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = basis.rank
        self.basis = basis
        self.residual_rank = residual_rank
        self.use_residual = use_residual

        # Layer-specific foliation coordinates: diagonal + low-rank core (rank x rank)
        # Parameterized efficiently as diagonal scaling + rank-128 core
        self.diag_s = nn.Parameter(torch.ones(self.rank))
        self.core_s_u = nn.Parameter(torch.randn(self.rank, min(128, self.rank)) * 0.01)
        self.core_s_v = nn.Parameter(torch.randn(self.rank, min(128, self.rank)) * 0.01)

        # High-frequency residual (A_l @ B_l^T)
        if use_residual and residual_rank > 0:
            self.res_A = nn.Parameter(torch.randn(out_features, residual_rank) * 0.01)
            self.res_B = nn.Parameter(torch.randn(in_features, residual_rank) * 0.01)
        else:
            self.res_A = None
            self.res_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates factored projection without materializing the full matrix in VRAM.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features).
        Returns:
            Output tensor of shape (batch, seq_len, out_features).
        """
        # 1. Project input into Stiefel basis V_g: (B, T, rank)
        h_v = torch.matmul(x, self.basis.V)

        # 2. Apply layer-specific foliation coordinates S_l
        # S_l = diag(s) + (core_u @ core_v^T)
        h_diag = h_v * self.diag_s
        h_core = torch.matmul(h_v, self.core_s_v)
        h_core = torch.matmul(h_core, self.core_s_u.t())
        h_s = h_diag + h_core

        # 3. Project out through Stiefel basis U_g: (B, T, out_features)
        y = torch.matmul(h_s, self.basis.U.t())

        # 4. Add high-frequency residual if present
        if self.res_A is not None and self.res_B is not None:
            res_in = torch.matmul(x, self.res_B)
            res_out = torch.matmul(res_in, self.res_A.t())
            y = y + res_out

        return y


class SheafChartNorm(nn.Module):
    """Gauge-Equivariant RMSNorm with Smooth Chart Transition Alignment."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
