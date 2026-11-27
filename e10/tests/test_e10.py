"""Unit tests for E10 Learned Low-Rank Transfer invariants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from e7.model import SimpleBehaviorReader
from e9.stage1_local_model import E9Stage1LocalPolicy
from e10 import EMBED_DIM, LATENT_DIM, LATENT_TOKENS
from e10.model import E10LowRankPolicy


def _make_dummy_policy(rank_k: int, transfer_mode: str) -> E10LowRankPolicy:
    torch.manual_seed(0)
    teacher = E9Stage1LocalPolicy(encoder_layers=1)
    reader = SimpleBehaviorReader(num_query_tokens=LATENT_TOKENS, embed_dim=EMBED_DIM, attn_dim=EMBED_DIM)
    g = torch.randn(LATENT_DIM, rank_k)
    u_k, _ = torch.linalg.qr(g, mode="reduced")
    z_mean = torch.randn(1, LATENT_TOKENS, EMBED_DIM) * 0.5
    return E10LowRankPolicy(
        teacher=teacher,
        reader=reader,
        basis_u_k=u_k,
        z_mean=z_mean,
        latent_scale_s_z=1.006,
        transfer_mode=transfer_mode,
        residual_layers=1,
    )


class TestE10Model(unittest.TestCase):
    def test_freeze_invariants_and_subspace_constraint(self) -> None:
        for mode, k in (("residual", 16), ("residual", 32), ("control", 32)):
            pol = _make_dummy_policy(k, mode)
            pol.train()
            self.assertFalse(pol.teacher.training)
            self.assertFalse(pol.reader.training)
            self.assertEqual(sum(p.numel() for p in pol.teacher.parameters() if p.requires_grad), 0)
            self.assertEqual(sum(p.numel() for p in pol.reader.parameters() if p.requires_grad), 0)
            self.assertFalse(pol.basis_u_k.requires_grad)

            B, M = 4, 5
            h_mem = torch.randn(B, M, LATENT_TOKENS, EMBED_DIM)
            b_mem = torch.randn(B, M, LATENT_TOKENS, EMBED_DIM)
            mpm = torch.zeros(B, M, dtype=torch.bool)
            q_rgb = torch.randn(B, 2, 2, EMBED_DIM)
            z_self = torch.randn(B, LATENT_TOKENS, EMBED_DIM)
            actions = torch.randn(B, 16, 10) * 0.1

            # Step 0 check for residual: exact identity z_hat == z_source
            z_c, h_q, _ = pol.read_cross_latent(h_mem, b_mem, mpm, q_rgb)
            z_hat, _ = pol.adapt_latent(z_c, h_q)
            if mode == "residual":
                self.assertLess(float((z_hat - z_c).abs().max().item()), 1e-6)

            # After random non-zero head weights, verify strict subspace containment
            with torch.no_grad():
                pol.transfer.coeff_head.weight.normal_(0.0, 0.05)
                pol.transfer.coeff_head.bias.normal_(0.0, 0.05)
            z_hat2, _ = pol.adapt_latent(z_c, h_q)
            base = z_c if mode == "residual" else pol.z_mean.expand_as(z_c)
            diff = (z_hat2 - base).reshape(B, LATENT_DIM)
            u = pol.basis_u_k
            proj = (diff @ u) @ u.T
            orth_err = float((diff - proj).abs().max().item())
            self.assertLess(orth_err, 1e-4, f"Mode {mode} K={k} leaked outside U_K: {orth_err}")

            # Gradient flow: teacher & reader get ZERO grads; transfer gets non-zero grads
            out = pol(h_mem, b_mem, mpm, q_rgb, z_self, actions)
            out["loss"].backward()
            self.assertFalse(any(p.grad is not None for p in pol.teacher.parameters()))
            self.assertFalse(any(p.grad is not None for p in pol.reader.parameters()))
            self.assertGreater(float(pol.transfer.coeff_head.weight.grad.norm().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
