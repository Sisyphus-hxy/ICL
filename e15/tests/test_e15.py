"""Unit tests for E15 mathematical identities, noise scale matching, subspace partitioning, and encoder freezing."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from e9.stage1_local_model import E9Stage1LocalPolicy
from e15.exp1_three_component import decompose_three_components, fit_three_component_model
from e15.residual_noise import LatentResidualNoiseInjector
from e15.train_e15 import _set_train_mode, compute_e15_relational_loss, compute_linear_cka_penalty


class TestE15MathAndInvariants(unittest.TestCase):
    def test_three_component_exact_identity_and_ols_vs_ridge_crossterm(self) -> None:
        rng = np.random.default_rng(7)
        N, D_a, D_z = 200, 16, 64
        A = rng.standard_normal((N, 1, D_a)).astype(np.float32)
        W_true = rng.standard_normal((D_a, D_z))
        # Nonlinear + orthogonal components
        Z = (
            A[:, 0, :] @ W_true
            + 0.4 * (A[:, 0, :] ** 2) @ (0.5 * W_true)
            + 0.6 * rng.standard_normal((N, D_z))
        ).astype(np.float32)

        dev = torch.device("cpu")
        # 1. OLS (alpha=0): exact zero cross-term on train & exact zero SVD truncation error
        m_ols = fit_three_component_model(A, Z, dev, forced_alpha=0.0, svd_threshold=1.0)
        d_ols = decompose_three_components(A, Z, m_ols)
        st = d_ols["stats"]
        self.assertLess(st["max_abs_identity_err"], 1e-11)
        self.assertAlmostEqual(st["frac_sum_check"], 1.0, places=9)
        self.assertLess(abs(st["frac_cross_2_lin_coax"]), 1e-10)
        self.assertLess(abs(st["frac_cross_par_perp"]), 1e-10)
        self.assertLess(st["frac_lin_trunc_svd_err"], 1e-12)

        # 2. Ridge (alpha=50.0): positive Ridge cross-term on train, exact sum check still 1.0!
        m_rid = fit_three_component_model(A, Z, dev, forced_alpha=50.0, svd_threshold=1.0)
        d_rid = decompose_three_components(A, Z, m_rid)
        st_r = d_rid["stats"]
        self.assertAlmostEqual(st_r["frac_sum_check"], 1.0, places=9)
        self.assertGreater(st_r["frac_cross_2_lin_coax"], 0.01)
        self.assertLess(abs(st_r["frac_cross_par_perp"]), 1e-10)

        # 3. SVD truncation (0.90): explicitly non-zero SVD truncation leakage!
        m_tr = fit_three_component_model(A, Z, dev, forced_alpha=1.0, svd_threshold=0.90)
        d_tr = decompose_three_components(A, Z, m_tr)
        self.assertGreater(d_tr["stats"]["frac_lin_trunc_svd_err"], 1e-4)
        self.assertAlmostEqual(d_tr["stats"]["frac_sum_check"], 1.0, places=9)

    def test_noise_injector_scale_invariance_and_random_norm_matching(self) -> None:
        rng = np.random.default_rng(11)
        bank = rng.standard_normal((500, 3072)).astype(np.float32) * 0.5
        R_M = 55.0
        dev = torch.device("cpu")

        inj_real = LatentResidualNoiseInjector(bank, R_M, dev, noise_type="real_cv", p_noisy=1.0, seed=42)
        inj_rand = LatentResidualNoiseInjector(bank, R_M, dev, noise_type="rand_matched", p_noisy=1.0, seed=42)

        b = torch.randn(64, 4, 768)
        eps_real, st_real = inj_real.sample_noise(b)
        eps_rand, st_rand = inj_rand.sample_noise(b)

        # Exact per-sample norm match between real_cv and rand_matched (with same seed -> same bank idx!)
        n_real = torch.linalg.norm(eps_real.reshape(64, -1), dim=-1)
        n_rand = torch.linalg.norm(eps_rand.reshape(64, -1), dim=-1)
        self.assertLess(float((n_real - n_rand).abs().max().item()), 1e-4)
        self.assertAlmostEqual(st_real["rel_eps_rms"], st_rand["rel_eps_rms"], places=4)

        # Scale invariance: scaling b by 3x must keep relative noise ratio identical!
        _, st_scaled = inj_real.sample_noise(b * 3.0)
        self.assertAlmostEqual(st_real["rel_eps_rms"], st_scaled["rel_eps_rms"], places=2)

    def test_op_subspace_loss_leaves_residual_subspace_grad_zero_without_decorr(self) -> None:
        b_t = torch.randn(32, 4, 768, requires_grad=True)
        a10 = torch.randn(32, 16, 10)
        loss, stats = compute_e15_relational_loss("op_e12_c", b_t, a10, op_dim=384, lambda_decorr=0.0)
        loss.backward()
        grad_op = b_t.grad[:, :, :384].abs().max().item()
        grad_res = b_t.grad[:, :, 384:].abs().max().item()
        self.assertGreater(grad_op, 1e-6)
        self.assertEqual(grad_res, 0.0)
        self.assertIn("cka_op_res", stats)

        # CKA sanity: identical matrices -> 1.0; orthogonal -> ~0.0
        x = torch.randn(64, 128)
        self.assertAlmostEqual(float(compute_linear_cka_penalty(x, x).item()), 1.0, places=5)

    def test_freeze_encoder_keeps_eval_and_zero_grad(self) -> None:
        pol = E9Stage1LocalPolicy(encoder_layers=1)
        for p in pol.local_encoder.parameters():
            p.requires_grad_(False)
        _set_train_mode(pol, freeze_encoder=True)
        self.assertFalse(pol.local_encoder.training)
        self.assertTrue(pol.action_decoder.training)
        groups = pol.get_optim_groups(lr=1e-4, weight_decay=1e-6)
        n_opt = sum(p.numel() for g in groups for p in g["params"])
        n_dec = sum(p.numel() for p in pol.action_decoder.parameters())
        self.assertEqual(n_opt, n_dec)


if __name__ == "__main__":
    unittest.main()
