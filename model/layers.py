"""Layer implementations for G-Qwen 9B with Grassmannian Atlas Foliation.

Includes:
- ChartBases: Container of shared Stiefel basis frames for a chart.
- FoliatedSwiGLUFFN: SwiGLU FFN with chart-based Grassmann factorizations.
- FoliatedGatedDeltaNetLayer: Linear attention DeltaNet layer.
- FoliatedGatedAttentionLayer: GQA Attention layer with RoPE.
- GQwenBlock: Unified 64-layer block.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from model.config import GQwen9BConfig
from model.topological_modules import (
    StiefelChartBasis,
    GrassmannFoliatedLinear,
    SheafChartNorm,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = q[..., : cos.shape[-1]]
    q_pass = q[..., cos.shape[-1] :]
    k_rot = k[..., : cos.shape[-1]]
    k_pass = k[..., cos.shape[-1] :]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class ChartBases(nn.Module):
    """Shared Stiefel frames for a topological macro-chart in the Bipartite Atlas."""

    def __init__(self, config: GQwen9BConfig, attn_basis: Optional[StiefelChartBasis] = None):
        super().__init__()
        d = config.hidden_size
        d_ff = config.intermediate_size
        d_v_delta = config.deltanet_v_heads * config.deltanet_head_dim
        d_v_attn = config.num_attention_heads * config.attention_head_dim
        r_ffn = config.atlas.chart_rank_ffn
        r_delta = config.atlas.chart_rank_deltanet
        r_attn = config.atlas.chart_rank_attn

        # Homogeneous FFN bases: gate, up, down (shared across all 4 layers in chart)
        self.basis_ffn_gate = StiefelChartBasis(in_features=d, out_features=d_ff, rank=r_ffn)
        self.basis_ffn_up = StiefelChartBasis(in_features=d, out_features=d_ff, rank=r_ffn)
        self.basis_ffn_down = StiefelChartBasis(in_features=d_ff, out_features=d, rank=r_ffn)

        # Specialized DeltaNet output projection basis (dedicated to DeltaNet layers)
        self.basis_mixer_deltanet = StiefelChartBasis(in_features=d_v_delta, out_features=d, rank=r_delta)

        # High-capacity Attention output projection basis (from macro-chart or dedicated)
        if attn_basis is not None:
            self.basis_mixer_attn = attn_basis
        else:
            self.basis_mixer_attn = StiefelChartBasis(in_features=d_v_attn, out_features=d, rank=r_attn)


class FoliatedSwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network using Grassmannian Foliated Projections."""

    def __init__(self, config: GQwen9BConfig, chart_bases: ChartBases):
        super().__init__()
        res_rank = config.atlas.residual_rank
        use_res = config.atlas.use_sparse_residuals

        self.gate_proj = GrassmannFoliatedLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            basis=chart_bases.basis_ffn_gate,
            residual_rank=res_rank,
            use_residual=use_res,
        )
        self.up_proj = GrassmannFoliatedLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            basis=chart_bases.basis_ffn_up,
            residual_rank=res_rank,
            use_residual=use_res,
        )
        self.down_proj = GrassmannFoliatedLinear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            basis=chart_bases.basis_ffn_down,
            residual_rank=res_rank,
            use_residual=use_res,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FoliatedGatedDeltaNetLayer(nn.Module):
    """Gated DeltaNet Layer with chart-factorized output projection."""

    def __init__(self, config: GQwen9BConfig, chart_bases: ChartBases):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.v_heads = config.deltanet_v_heads
        self.qk_heads = config.deltanet_qk_heads
        self.head_dim = config.deltanet_head_dim
        self.d_v = self.v_heads * self.head_dim
        self.d_qk = self.qk_heads * self.head_dim
        self.kernel_size = config.deltanet_conv_kernel_size

        self.q_proj = nn.Linear(self.hidden_size, self.d_qk, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.d_qk, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.d_v, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.v_heads, bias=False)

        self.conv_qk = nn.Conv1d(
            in_channels=self.d_qk * 2,
            out_channels=self.d_qk * 2,
            kernel_size=self.kernel_size,
            groups=self.d_qk * 2,
            padding=self.kernel_size - 1,
            bias=True,
        )
        self.conv_v = nn.Conv1d(
            in_channels=self.d_v,
            out_channels=self.d_v,
            kernel_size=self.kernel_size,
            groups=self.d_v,
            padding=self.kernel_size - 1,
            bias=True,
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.d_v, bias=False)

        # DeltaNet Chart-factorized output projection (d_v -> hidden_size)
        self.out_proj = GrassmannFoliatedLinear(
            in_features=self.d_v,
            out_features=self.hidden_size,
            basis=chart_bases.basis_mixer_deltanet,
            residual_rank=config.atlas.residual_rank,
            use_residual=config.atlas.use_sparse_residuals,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))

        qk_cat = torch.cat([q, k], dim=-1).transpose(1, 2)
        qk_conv = self.conv_qk(qk_cat)[:, :, :seq_len].transpose(1, 2)
        q, k = torch.split(qk_conv, self.d_qk, dim=-1)

        v_conv = self.conv_v(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        v = F.silu(v_conv)

        k = F.normalize(k.view(batch, seq_len, self.qk_heads, self.head_dim), p=2, dim=-1)
        q = q.view(batch, seq_len, self.qk_heads, self.head_dim)

        ratio = self.v_heads // self.qk_heads
        k_exp = k.repeat_interleave(ratio, dim=2)
        q_exp = q.repeat_interleave(ratio, dim=2)
        v_exp = v.view(batch, seq_len, self.v_heads, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bshd,bthd->bhts", q_exp, k_exp) * scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal_mask, 0.0)

        out_v = torch.einsum("bhts,bthd->bshd", attn, v_exp)
        out_v = out_v.reshape(batch, seq_len, self.d_v)

        gate = F.silu(self.gate_proj(x))
        out_v = out_v * gate

        # Dimension alignment if d_v != hidden_size
        return self.out_proj(out_v)


class FoliatedGatedAttentionLayer(nn.Module):
    """Gated Softmax Attention (GQA) with RoPE and chart-factorized output projection."""

    def __init__(self, config: GQwen9BConfig, chart_bases: ChartBases):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.rope_dim = config.rope_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        d_v = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.out_proj = GrassmannFoliatedLinear(
            in_features=d_v,
            out_features=self.hidden_size,
            basis=chart_bases.basis_mixer_attn,
            residual_rank=config.atlas.residual_rank,
            use_residual=config.atlas.use_sparse_residuals,
        )

        self.rope_theta = config.rope_theta
        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_rope_embeddings(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb).unsqueeze(0).unsqueeze(2), torch.sin(emb).unsqueeze(0).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        gate = torch.sigmoid(self.gate_proj(x))

        cos, sin = self._get_rope_embeddings(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = attn_out * gate
        return self.out_proj(attn_out)


class GQwenBlock(nn.Module):
    """Single Transformer layer belonging to a Grassmann chart."""

    def __init__(
        self,
        config: GQwen9BConfig,
        chart_bases: ChartBases,
        layer_type: str = "deltanet",
    ):
        super().__init__()
        self.layer_type = layer_type

        self.input_norm = SheafChartNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = SheafChartNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_type == "deltanet":
            self.mixer = FoliatedGatedDeltaNetLayer(config, chart_bases)
        elif layer_type == "attention":
            self.mixer = FoliatedGatedAttentionLayer(config, chart_bases)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.ffn = FoliatedSwiGLUFFN(config, chart_bases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.input_norm(x))
        x = x + self.ffn(self.post_attention_norm(x))
        return x
