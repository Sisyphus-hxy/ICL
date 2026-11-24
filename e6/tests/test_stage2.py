"""Unit tests for E6 Stage 2 (Latent Behavior Transfer, CLIP)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from e6 import ACTION_DIM_10D, ACTION_HORIZON, EMBED_DIM, NUM_QUERY_TOKENS
from e6.stage1_model import E6Stage1Policy
from e6.stage2_model import E6Stage2Policy
from e6.train_stage2 import get_cosine_schedule_with_min_lr


class TestE6Stage2(unittest.TestCase):
    def _build_dummy_stage2(self, s_z: float = 7.8) -> E6Stage2Policy:
        torch.manual_seed(0)
        teacher = E6Stage1Policy(use_attention_sink=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return E6Stage2Policy(teacher, latent_scale_s_z=s_z)

    def test_teacher_frozen_eval_and_zero_init_identity(self):
        policy = self._build_dummy_stage2(s_z=7.8)
        policy.train(True)
        # Teacher MUST remain frozen and in eval mode even after policy.train(True)
        self.assertFalse(policy.teacher.training)
        self.assertTrue(policy.residual_transformer.training)
        self.assertEqual(
            sum(p.numel() for p in policy.teacher.parameters() if p.requires_grad), 0
        )
        self.assertGreater(
            sum(p.numel() for p in policy.residual_transformer.parameters() if p.requires_grad),
            28_000_000,
        )

        # Zero-init identity: delta_z == 0, z_adapted == z_cross
        B = 4
        z_cross = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        h_q = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        z_adapted, delta_z = policy.adapt_latent(z_cross, h_q)
        torch.testing.assert_close(delta_z, torch.zeros_like(delta_z), atol=0.0, rtol=0.0)
        torch.testing.assert_close(z_adapted, z_cross, atol=0.0, rtol=0.0)

    def test_diffusion_loss_backprops_through_frozen_unet_to_residual(self):
        """CRITICAL: Verify Diffusion Loss alone (lambda_z=0, lambda_a=1) backpropagates
        finite non-zero gradients through the FROZEN Stage 1 UNet into ResidualTransformer,
        while zero Teacher parameters receive gradients."""
        policy = self._build_dummy_stage2(s_z=7.8)
        policy.train()

        B, P = 4, 8
        demo_mem = torch.randn(B, P, EMBED_DIM)
        pad_mask = torch.zeros(B, P, dtype=torch.bool)
        query_rgb = torch.randn(B, 2, 2, EMBED_DIM)
        z_self = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM) * 7.8
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_self = torch.zeros(B, dtype=torch.bool)

        opt = torch.optim.AdamW(
            policy.get_optim_groups(lr=1e-3, weight_decay=0.01),
            betas=(0.9, 0.95),
        )

        # Step 1: zero-init out_proj receives non-zero grad strictly from frozen UNet
        opt.zero_grad(set_to_none=True)
        out1 = policy(
            demo_mem, pad_mask, query_rgb, z_self, actions, is_self, lambda_z=0.0, lambda_a=1.0
        )
        out1["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        g_w = policy.residual_transformer.out_proj.weight.grad
        g_b = policy.residual_transformer.out_proj.bias.grad
        self.assertIsNotNone(g_w)
        self.assertIsNotNone(g_b)
        self.assertGreater(g_w.float().norm().item(), 0.0)
        self.assertGreater(g_b.float().norm().item(), 0.0)
        opt.step()

        # Step 2: after out_proj becomes non-zero, ALL ResidualTransformer parameters must
        # receive finite non-zero gradients purely through frozen UNet!
        opt.zero_grad(set_to_none=True)
        out2 = policy(
            demo_mem, pad_mask, query_rgb, z_self, actions, is_self, lambda_z=0.0, lambda_a=1.0
        )
        out2["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        for name, p in policy.residual_transformer.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing grad for {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad for {name}")
            self.assertGreater(p.grad.float().norm().item(), 0.0, f"Zero grad for {name}")

    def test_latent_loss_normalization_and_lr_schedule(self):
        policy = self._build_dummy_stage2(s_z=8.0)
        B = 4
        z_cross = torch.zeros(B, NUM_QUERY_TOKENS, EMBED_DIM)
        z_self = torch.full((B, NUM_QUERY_TOKENS, EMBED_DIM), 8.0)
        h_q = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        actions = torch.zeros(B, ACTION_HORIZON, ACTION_DIM_10D)
        # Since delta_z == 0 and (z_cross - z_self) / s_z == -1 everywhere -> L_z == 1.0 exactly
        out = policy.compute_loss_from_latents(
            z_cross, h_q, z_self, actions, lambda_z=0.5, lambda_a=0.0
        )
        self.assertAlmostEqual(float(out["latent_loss_norm"].item()), 1.0, places=5)
        self.assertAlmostEqual(float(out["loss"].item()), 0.5, places=5)

        # Verify cosine LR schedule with warmup=500 and min_lr=1e-5
        layer = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
        sched = get_cosine_schedule_with_min_lr(
            opt, num_warmup_steps=500, num_training_steps=42912, lr=1e-4, min_lr=1e-5
        )
        lrs = []
        for _ in range(42912):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        self.assertAlmostEqual(lrs[0], 0.0, places=9)
        self.assertAlmostEqual(lrs[500], 1e-4, places=7)
        self.assertAlmostEqual(lrs[-1], 1e-5, places=6)


if __name__ == "__main__":
    unittest.main()
