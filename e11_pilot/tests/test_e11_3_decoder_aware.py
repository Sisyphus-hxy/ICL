"""Unit tests for E11.3 (Decoder-Aware Transfer Optimization)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from e11_pilot.data import load_frozen_e10c_and_basis
from e11_pilot.e11_3_decoder_aware import (
    audit_decoder_aware_freeze_and_grad,
    build_initialized_transfer_model,
    compute_decoder_aware_loss,
)


class TestE113DecoderAware(unittest.TestCase):
    def test_inits_freeze_grad_and_loss_scale(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy, u64_t, meta = load_frozen_e10c_and_basis(device=device)
        s_z = float(meta["s_z"])
        e10c_params = sum(p.numel() for p in policy.transfer.parameters())

        num_train_t = int(policy.teacher.noise_scheduler.config.num_train_timesteps)
        t_edges = np.linspace(0, num_train_t, 5, dtype=int)

        torch.manual_seed(0)
        z_src = torch.randn(8, 3072, device=device)
        h_q = torch.randn(8, 4, 768, device=device)
        gt_a = torch.randn(8, 16, 10, device=device) * 0.2

        for grp in ("A", "B", "C"):
            mod, imeta = build_initialized_transfer_model(
                group=grp, s_z=s_z, dropout=0.15, seed=42, device=device
            )
            n_tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            self.assertEqual(n_tr, 28_562_752)
            self.assertEqual(n_tr, e10c_params)

            # Verify Scratch outputs exact c=0 while A/B output non-zero coefficients
            mod.eval()
            with torch.no_grad():
                c0 = mod(z_src.reshape(8, 4, 768), h_q)
            if grp == "C":
                self.assertAlmostEqual(float(c0.abs().max().item()), 0.0, places=7)
                self.assertIsNone(imeta["init_ckpt"])
            else:
                self.assertGreater(float(c0.norm(dim=-1).mean().item()), 1.0)
                self.assertIsNotNone(imeta["init_ckpt"])

            # Verify freeze invariant and non-zero through-decoder gradient
            audit = audit_decoder_aware_freeze_and_grad(
                model=mod,
                policy=policy,
                u64_t=u64_t,
                z_src_b=z_src,
                h_q_b=h_q,
                a_b=gt_a,
                lam=1e-5,
                mc_timesteps=4,
                t_edges=t_edges,
                use_bf16=True,
            )
            self.assertFalse(audit["teacher_any_grad"])
            self.assertFalse(audit["reader_any_grad"])
            self.assertFalse(audit["u64_requires_grad"])
            self.assertGreater(audit["coeff_head_grad_norm"], 0.0)
            self.assertGreater(audit["transfer_nonzero_grad_tensors"], 0)

        # Verify exact lambda * ||c||_2^2 scaling on Group A with identical RNG
        mod_a, _ = build_initialized_transfer_model(
            group="A", s_z=s_z, dropout=0.15, seed=42, device=device
        )
        mod_a.eval()
        g0 = torch.Generator(device=device).manual_seed(1234)
        tot0, diff0, reg0, c0 = compute_decoder_aware_loss(
            model=mod_a,
            teacher=policy.teacher,
            u64_t=u64_t,
            z_src_flat=z_src,
            h_q=h_q,
            gt_actions=gt_a,
            lam=0.0,
            mc_timesteps=4,
            t_edges=t_edges,
            diff_gen=g0,
            use_bf16=False,
        )
        g1 = torch.Generator(device=device).manual_seed(1234)
        tot1, diff1, reg1, c1 = compute_decoder_aware_loss(
            model=mod_a,
            teacher=policy.teacher,
            u64_t=u64_t,
            z_src_flat=z_src,
            h_q=h_q,
            gt_actions=gt_a,
            lam=1e-5,
            mc_timesteps=4,
            t_edges=t_edges,
            diff_gen=g1,
            use_bf16=False,
        )
        self.assertAlmostEqual(float(reg0.item()), 0.0, places=9)
        self.assertAlmostEqual(float(diff0.item()), float(diff1.item()), places=6)
        expected_reg = 1e-5 * float(c1.square().sum(dim=-1).mean().item())
        self.assertAlmostEqual(float(reg1.item()), expected_reg, places=7)
        self.assertAlmostEqual(float((tot1 - tot0).item()), expected_reg, places=6)


if __name__ == "__main__":
    unittest.main()
