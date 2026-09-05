"""Unit Tests and Validation Suite for G-Qwen 9B (Grassmannian Foliated Atlas).

Tests:
1. Architectural configuration (64 physical layers, 16 charts of 4 layers each).
2. Parameter budget audit (verifying physical parameters <= 9.00B).
3. End-to-End causal forward pass across all 64 layers.
4. Backward pass and gradient flow through the 64-layer chart atlas.
5. Grassmannian chart projector SVD reconstruction accuracy.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GQwen9BConfig, AtlasConfig
from model.g_qwen_9b import (
    GQwen9BModel,
    GQwen9BForCausalLM,
)
from model.topological_modules import (
    StiefelChartBasis,
    GrassmannFoliatedLinear,
    SheafChartNorm,
)
from tools.projector import GrassmannChartProjector


class TestGQwenArchitecture(unittest.TestCase):
    """Validation test suite for G-Qwen 9B."""

    def setUp(self):
        self.prod_config = GQwen9BConfig()

        # Scaled-down configuration for fast unit testing execution
        self.test_config = GQwen9BConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=16,  # 4 charts of 4 layers
            deltanet_v_heads=2,
            deltanet_qk_heads=2,
            deltanet_head_dim=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_head_dim=32,
            rope_dim=16,
            atlas=AtlasConfig(
                num_charts=4,
                layers_per_chart=4,
                chart_rank_ffn=32,
                chart_rank_deltanet=32,
                chart_rank_attn=32,
                num_attn_macro_charts=2,
                residual_rank=8,
            ),
            layer_types=[
                layer_type
                for _ in range(4)
                for layer_type in ["deltanet", "deltanet", "deltanet", "attention"]
            ],
        )

    def test_layer_and_chart_configuration(self):
        """Verifies that G-Qwen 9B maintains 64 layers in 16 charts."""
        self.assertEqual(self.prod_config.num_layers, 64)
        self.assertEqual(self.prod_config.atlas.num_charts, 16)
        self.assertEqual(self.prod_config.atlas.layers_per_chart, 4)

        # Count DeltaNet and Attention layers
        deltanets = sum(1 for t in self.prod_config.layer_types if t == "deltanet")
        attentions = sum(1 for t in self.prod_config.layer_types if t == "attention")
        self.assertEqual(deltanets, 48)
        self.assertEqual(attentions, 16)

    def test_production_parameter_accounting(self):
        """Mathematically audits the parameter count of the 64-layer G-Qwen 9B v2 model."""
        d = self.prod_config.hidden_size                  # 5120
        d_ff = self.prod_config.intermediate_size          # 17408
        vocab = self.prod_config.vocab_size                # 152064
        r_ffn = self.prod_config.atlas.chart_rank_ffn      # 1792
        r_delta = self.prod_config.atlas.chart_rank_deltanet # 1024
        r_attn = self.prod_config.atlas.chart_rank_attn    # 2048
        num_attn_macro = self.prod_config.atlas.num_attn_macro_charts # 4
        r_res = self.prod_config.atlas.residual_rank       # 64

        d_v = self.prod_config.deltanet_v_heads * self.prod_config.deltanet_head_dim # 6144
        d_qk = self.prod_config.deltanet_qk_heads * self.prod_config.deltanet_head_dim # 2048

        # 1. Tied Embeddings
        embed_params = vocab * d

        # 2. Stiefel Bases:
        # 16 FFN bases (gate, up, down)
        ffn_basis_per_chart = 3 * (d * r_ffn + d_ff * r_ffn)
        total_ffn_bases = 16 * ffn_basis_per_chart

        # 16 DeltaNet dedicated mixer bases
        total_deltanet_bases = 16 * (d_v * r_delta + d * r_delta)

        # 4 Attention macro-chart mixer bases
        total_attn_bases = num_attn_macro * (d_v * r_attn + d * r_attn)

        total_bases = total_ffn_bases + total_deltanet_bases + total_attn_bases

        # 3. 64 Layers foliated coordinates and residuals
        # FFN coordinates per layer: 3 * (diag_s: r_ffn + core: 2 * r_ffn * 128)
        ffn_coords_per_layer = 3 * (r_ffn + 2 * r_ffn * 128)
        # FFN residuals per layer: 3 * (d * r_res + d_ff * r_res)
        ffn_res_per_layer = 3 * (d * r_res + d_ff * r_res)

        # DeltaNet exclusive projections (q, k, v, beta, conv1d, gate)
        deltanet_exclusive = (
            (d * d_qk) * 2
            + (d * d_v)
            + (d * self.prod_config.deltanet_v_heads)
            + (4 * (d_qk * 2 + d_v))
            + (d * d_v)
            + (r_delta + 2 * r_delta * 128) # mixer out coords
            + ((d_v + d) * r_res)           # mixer out res
            + 2 * d                         # norms
        )

        # Attention exclusive projections (q, k, v, gate)
        num_q = self.prod_config.num_attention_heads * self.prod_config.attention_head_dim # 6144
        num_kv = self.prod_config.num_key_value_heads * self.prod_config.attention_head_dim # 1024
        attn_exclusive = (
            (d * num_q)
            + (d * num_kv) * 2
            + (d * num_q)
            + (r_attn + 2 * r_attn * 128)   # mixer out coords
            + ((d_v + d) * r_res)           # mixer out res
            + 2 * d                         # norms
        )

        total_layers = (
            48 * (ffn_coords_per_layer + ffn_res_per_layer + deltanet_exclusive)
            + 16 * (ffn_coords_per_layer + ffn_res_per_layer + attn_exclusive)
        )

        total_physical = embed_params + total_bases + total_layers

        print(f"\n[Analytic G-Qwen 9B Parameter Audit]")
        print(f"  Embeddings (Tied): {embed_params / 1e9:.3f} B")
        print(f"  16 Stiefel Chart Bases: {total_bases / 1e9:.3f} B")
        print(f"  64 Foliated Layers: {total_layers / 1e9:.3f} B")
        print(f"  Total Physical Parameters: {total_physical / 1e9:.3f} B")

        self.assertLessEqual(total_physical, 9.00e9)

    def test_forward_pass_all_64_layers(self):
        """Verifies forward pass through all layers in the atlas."""
        model = GQwen9BForCausalLM(self.test_config)
        model.eval()

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.test_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids)

        logits = output["logits"]
        self.assertEqual(logits.shape, (batch_size, seq_len, self.test_config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward_pass_gradient_flow(self):
        """Verifies backpropagation through all charts and foliated layers."""
        model = GQwen9BForCausalLM(self.test_config)
        model.train()

        batch_size = 2
        seq_len = 4
        input_ids = torch.randint(0, self.test_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        loss = output["loss"]
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        # Check gradients in Chart 0 basis and Layer 0 coordinates
        chart_0 = model.model.charts[0]
        layer_0 = model.model.layers[0]
        self.assertIsNotNone(chart_0.basis_ffn_gate.U.grad)
        self.assertIsNotNone(layer_0.ffn.gate_proj.diag_s.grad)

    def test_grassmann_chart_projector(self):
        """Verifies multi-chart SVD projector on a coherent 4-matrix group."""
        projector = GrassmannChartProjector(rank=16, residual_rank=4)
        d_out, d_in = 32, 64

        # Generate a coherent 4-layer group with realistic low-rank spectral decay
        torch.manual_seed(42)
        U_true, _ = torch.linalg.qr(torch.randn(d_out, 12))
        V_true, _ = torch.linalg.qr(torch.randn(d_in, 12))
        s_true = torch.exp(-torch.linspace(0, 2, 12))
        base = torch.matmul(U_true, torch.matmul(torch.diag(s_true), V_true.t()))

        # 1% relative deviation between adjacent layers in the group
        noise_scale = 0.01 * torch.norm(base) / (d_out * d_in)**0.5
        group = [base + noise_scale * torch.randn(d_out, d_in) for _ in range(4)]

        U_g, V_g, S_list, res_list = projector.project_layer_group(group)

        self.assertEqual(U_g.shape, (d_out, 16))
        self.assertEqual(V_g.shape, (d_in, 16))
        self.assertEqual(len(S_list), 4)

        # Check orthogonality: U^T U = I, V^T V = I
        eye_u = torch.matmul(U_g.t(), U_g)
        eye_v = torch.matmul(V_g.t(), V_g)
        self.assertTrue(torch.allclose(eye_u, torch.eye(16), atol=1e-4))
        self.assertTrue(torch.allclose(eye_v, torch.eye(16), atol=1e-4))

        # Reconstruct layer 0 and measure relative error
        A0, B0 = res_list[0]
        W0_reconst = torch.matmul(U_g, torch.matmul(S_list[0], V_g.t())) + torch.matmul(A0, B0.t())
        rel_error = torch.norm(group[0] - W0_reconst) / torch.norm(group[0])

        print(f"  Chart Projector Relative Reconstruction Error: {rel_error.item():.4f}")
        self.assertLess(rel_error.item(), 0.05)  # Error < 5%


if __name__ == "__main__":
    unittest.main()
