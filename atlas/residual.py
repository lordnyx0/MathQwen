# -*- coding: utf-8 -*-
"""Módulos de estabilização do residual stream (Linear e Não-Linear com modulação de escala e warm-start)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearResidualStabilizer(nn.Module):
    def __init__(self, W_down: torch.Tensor, W_up: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.W_down = nn.Parameter(W_down.clone())
        self.W_up = nn.Parameter(W_up.clone())
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32, device=W_down.device))

    def freeze(self):
        self.W_down.requires_grad = False
        self.W_up.requires_grad = False
        self.alpha.requires_grad = False

    def unfreeze(self):
        self.W_down.requires_grad = True
        self.W_up.requires_grad = True
        self.alpha.requires_grad = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        corr = torch.matmul(torch.matmul(h, self.W_down), self.W_up)
        if self.alpha.item() == 1.0 and not self.alpha.requires_grad:
            return h + corr
        a = self.alpha.to(dtype=h.dtype)
        return a * h + corr

class NonLinearResidualStabilizer(nn.Module):
    def __init__(self, W_down: torch.Tensor, W_up: torch.Tensor, delta_alpha: float = 0.0, alpha: float = None):
        super().__init__()
        self.W_down = nn.Parameter(W_down.clone())
        self.W_up = nn.Parameter(W_up.clone())
        init_val = delta_alpha if alpha is None else (alpha - 1.0)
        self.delta_alpha = nn.Parameter(torch.tensor([init_val], dtype=torch.float32, device=W_down.device))

    def freeze(self):
        self.W_down.requires_grad = False
        self.W_up.requires_grad = False
        self.delta_alpha.requires_grad = False

    def unfreeze(self):
        self.W_down.requires_grad = True
        self.W_up.requires_grad = True
        self.delta_alpha.requires_grad = True

    @property
    def alpha(self):
        return 1.0 + self.delta_alpha

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = F.gelu(torch.matmul(h.float(), self.W_down.float())).to(dtype=h.dtype)
        corr = torch.matmul(z, self.W_up)
        a = (1.0 + self.delta_alpha).to(dtype=h.dtype)
        return a * h + corr

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

def fit_joint_alpha_stabilizer(
    X_calib: torch.Tensor,
    Y_calib: torch.Tensor,
    W_down: torch.Tensor,
    is_gelu: bool = False,
    lambda_reg: float = 1e-3,
    lambda_alpha: float = 1e-2
):
    """
    Resolve analiticamente o problema conjunto:
    min_{alpha, W_up} ||Y - alpha * X - Z @ W_up||_F^2 + lambda_alpha * (alpha - 1)^2 + lambda_reg * ||W_up||_F^2
    onde Z = GELU(X @ W_down) se is_gelu else (X @ W_down).
    """
    X_f = X_calib.float()
    Y_f = Y_calib.float()
    if is_gelu:
        Z = F.gelu(torch.matmul(X_f, W_down.float()))
    else:
        Z = torch.matmul(X_f, W_down.float())

    r = W_down.shape[1]
    ZTZ = torch.matmul(Z.t(), Z)
    reg_val = lambda_reg * (torch.trace(ZTZ) / r)
    reg_I = reg_val * torch.eye(r, device=X_calib.device, dtype=torch.float32)

    L = torch.linalg.cholesky(ZTZ + reg_I)
    W_up_Y = torch.cholesky_solve(torch.matmul(Z.t(), Y_f), L)
    W_up_X = torch.cholesky_solve(torch.matmul(Z.t(), X_f), L)

    tilde_Y = Y_f - torch.matmul(Z, W_up_Y)
    tilde_X = X_f - torch.matmul(Z, W_up_X)

    num = (tilde_X * tilde_Y).sum() + lambda_alpha
    den = (tilde_X * tilde_X).sum() + lambda_alpha
    alpha_opt = (num / den).item()

    W_up_opt = (W_up_Y - alpha_opt * W_up_X).to(dtype=torch.bfloat16)
    return alpha_opt, W_up_opt
