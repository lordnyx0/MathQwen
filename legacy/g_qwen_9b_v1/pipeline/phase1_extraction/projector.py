"""Weight Projector Tool for G-Qwen 9B (Grassmannian Foliated Atlas).

Implements the multi-chart SVD projection and Eckart-Young-Mirsky optimal
subspace extraction as formulated in Document 03.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List


class GrassmannChartProjector:
    """Projects 64 dense layers of Qwen 3.8 27B into 16 Grassmann charts of G-Qwen 9B."""

    def __init__(self, rank: int = 1792, residual_rank: int = 64):
        self.rank = rank
        self.residual_rank = residual_rank

    def project_layer_group(
        self,
        weight_group: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Decomposes 4 weight matrices from a chart into shared Stiefel frames U, V, coordinates S_l, and residuals.

        Args:
            weight_group: List of 4 matrices [W_1, W_2, W_3, W_4], each (d_out, d_in).
        Returns:
            U_g: Shared output Stiefel frame (d_out, rank).
            V_g: Shared input Stiefel frame (d_in, rank).
            S_list: List of 4 coordinate matrices S_l (rank, rank).
            res_list: List of 4 tuples (A_l, B_l) for the high-frequency residuals.
        """
        assert len(weight_group) == 4, "A chart must contain exactly 4 layer weight matrices."

        d_out, d_in = weight_group[0].shape
        actual_rank = min(self.rank, d_out, d_in)

        # 1. Output Basis U_g via horizontal concatenation SVD
        M_out = torch.cat(weight_group, dim=1).float()  # (d_out, 4 * d_in)
        U_full, _, _ = torch.linalg.svd(M_out, full_matrices=False)
        U_g = U_full[:, :actual_rank]  # (d_out, rank)

        # 2. Input Basis V_g via vertical concatenation SVD
        M_in = torch.cat([W.t() for W in weight_group], dim=1).float()  # (d_in, 4 * d_out)
        V_full, _, _ = torch.linalg.svd(M_in, full_matrices=False)
        V_g = V_full[:, :actual_rank]  # (d_in, rank)

        # 3. Layer coordinates S_l and high-frequency residuals
        S_list = []
        res_list = []

        for W in weight_group:
            W_float = W.float()
            # Optimal projection: S_l = U_g^T @ W @ V_g
            S_l = torch.matmul(U_g.t(), torch.matmul(W_float, V_g))
            S_list.append(S_l.type_as(W))

            # Residual Delta_l = W - U_g @ S_l @ V_g^T
            Delta = W_float - torch.matmul(U_g, torch.matmul(S_l, V_g.t()))

            if self.residual_rank > 0:
                U_res, S_res, Vh_res = torch.linalg.svd(Delta, full_matrices=False)
                r_res = min(self.residual_rank, len(S_res))
                scale_res = torch.sqrt(S_res[:r_res])
                A_l = (U_res[:, :r_res] * scale_res).type_as(W)
                B_l = (Vh_res[:r_res, :].t() * scale_res).type_as(W)
                res_list.append((A_l, B_l))
            else:
                res_list.append((None, None))

        return U_g.type_as(weight_group[0]), V_g.type_as(weight_group[0]), S_list, res_list
