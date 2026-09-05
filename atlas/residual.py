# -*- coding: utf-8 -*-
"""Módulos de estabilização do residual stream (Linear e Não-Linear com warm-start)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearResidualStabilizer(nn.Module):
    def __init__(self, W_down: torch.Tensor, W_up: torch.Tensor):
        super().__init__()
        self.W_down = nn.Parameter(W_down.clone())
        self.W_up = nn.Parameter(W_up.clone())

    def freeze(self):
        self.W_down.requires_grad = False
        self.W_up.requires_grad = False

    def unfreeze(self):
        self.W_down.requires_grad = True
        self.W_up.requires_grad = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        corr = torch.matmul(torch.matmul(h, self.W_down), self.W_up)
        return h + corr

class NonLinearResidualStabilizer(nn.Module):
    def __init__(self, W_down: torch.Tensor, W_up: torch.Tensor, delta_alpha: float = 0.0):
        super().__init__()
        self.W_down = nn.Parameter(W_down.clone())
        self.W_up = nn.Parameter(W_up.clone())
        self.delta_alpha = nn.Parameter(torch.tensor([delta_alpha], dtype=torch.float32, device=W_down.device))

    def freeze(self):
        self.W_down.requires_grad = False
        self.W_up.requires_grad = False
        self.delta_alpha.requires_grad = False

    def unfreeze(self):
        self.W_down.requires_grad = True
        self.W_up.requires_grad = True
        self.delta_alpha.requires_grad = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = F.gelu(torch.matmul(h.float(), self.W_down.float())).to(dtype=h.dtype)
        corr = torch.matmul(z, self.W_up)
        alpha = (1.0 + self.delta_alpha).to(dtype=h.dtype)
        return alpha * h + corr

def freeze_backbone_and_isolate_stabilizer(layer_module: nn.Module, stabilizer: nn.Module = None):
    """Garante estruturalmente que os parâmetros do backbone permaneçam congelados (requires_grad=False)."""
    for param in layer_module.parameters():
        param.requires_grad = False
    if stabilizer is not None:
        for param in stabilizer.parameters():
            param.requires_grad = True

def fit_svd_stabilizer(X_calib: torch.Tensor, Y_calib: torch.Tensor, r_corr: int = 64, lambda_reg: float = 1e-3):
    E = Y_calib - X_calib
    N, d = X_calib.shape

    U, S, Vh = torch.linalg.svd(X_calib.float(), full_matrices=False)
    V = Vh.t()

    reg = lambda_reg * (S**2).mean()
    weights = S / (S**2 + reg)
    M = torch.matmul(E.float().t(), U * weights)

    Um, Sm, Vhm = torch.linalg.svd(M, full_matrices=False)
    Wr = torch.matmul(V, Vhm.t())

    k = min(r_corr, N)
    W_down = (Wr[:, :k] * torch.sqrt(Sm[:k])).to(dtype=torch.bfloat16)
    W_up = (Um[:, :k] * torch.sqrt(Sm[:k])).t().to(dtype=torch.bfloat16)

    return W_down, W_up

def fit_gelu_warmstart(X_calib: torch.Tensor, Y_calib: torch.Tensor, W_down: torch.Tensor, lambda_reg: float = 1e-3):
    Z_tr = F.gelu(torch.matmul(X_calib.float(), W_down.float()))
    R_tr = (Y_calib - X_calib).float()
    r = W_down.shape[1]

    ZTZ = torch.matmul(Z_tr.t(), Z_tr)
    reg_val = lambda_reg * (torch.trace(ZTZ) / r)
    reg_I = reg_val * torch.eye(r, device=X_calib.device, dtype=torch.float32)
    ZTR = torch.matmul(Z_tr.t(), R_tr)

    W_up = torch.linalg.solve(ZTZ + reg_I, ZTR).to(dtype=torch.bfloat16)
    return W_up
