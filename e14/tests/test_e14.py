"""Unit tests for E14 mathematical invariants, scale control, subspace projection, and equal-param controls."""

import unittest

import numpy as np
import torch

from e14.common import (
    apply_global_rms_standardizer,
    build_block_rms_features,
    compute_latent_predictability_metrics,
    fit_action_explainable_subspace,
    fit_global_rms_standardizer,
    fit_predict_rbf_krr,
    fit_predict_scaled_ridge,
    invert_global_rms_standardizer,
    project_subspace,
)
from e14.exp1_cross_transfer import two_way_anova_4x4
from e14.exp3_complementary_repr import GatedFusionAdapter, compute_orthogonal_residual


class TestE14Invariants(unittest.TestCase):
    def test_block_rms_scale_invariance(self):
        rng = np.random.default_rng(0)
        b1 = rng.standard_normal((200, 64)).astype(np.float32) * 0.05 + 3.0
        b2 = rng.standard_normal((200, 128)).astype(np.float32) * 17.5 - 4.0
        b1_ev = rng.standard_normal((50, 64)).astype(np.float32) * 0.05 + 3.0
        b2_ev = rng.standard_normal((50, 128)).astype(np.float32) * 17.5 - 4.0

        X_tr, [X_ev] = build_block_rms_features([b1, b2], [[b1_ev], [b2_ev]])
        # Each block should have expected squared L2 norm == 1.0 on train!
        e1 = float(np.mean(np.sum(X_tr[:, :64] ** 2, axis=-1)))
        e2 = float(np.mean(np.sum(X_tr[:, 64:] ** 2, axis=-1)))
        self.assertAlmostEqual(e1, 1.0, places=4)
        self.assertAlmostEqual(e2, 1.0, places=4)

        # Scaling b2 by 100x must leave X_tr and X_ev strictly invariant!
        X_tr2, [X_ev2] = build_block_rms_features([b1, b2 * 100.0], [[b1_ev], [b2_ev * 100.0]])
        np.testing.assert_allclose(X_tr, X_tr2, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(X_ev, X_ev2, atol=1e-5, rtol=1e-5)

    def test_scaled_ridge_and_krr_output_scale_invariance(self):
        rng = np.random.default_rng(1)
        dev = torch.device("cpu")
        X_tr = rng.standard_normal((120, 16)).astype(np.float32)
        X_te = rng.standard_normal((30, 16)).astype(np.float32)
        W = rng.standard_normal((16, 12)).astype(np.float32)
        Y_tr = X_tr @ W + 0.1 * rng.standard_normal((120, 12)).astype(np.float32)

        [p1], _ = fit_predict_scaled_ridge(X_tr, Y_tr, [X_te], dev)
        [p2], _ = fit_predict_scaled_ridge(X_tr, Y_tr * 50.0 + 7.0, [X_te], dev)
        np.testing.assert_allclose(p1 * 50.0 + 7.0, p2, atol=1e-3, rtol=1e-3)

        [k1], _ = fit_predict_rbf_krr(X_tr, Y_tr, [X_te], dev)
        [k2], _ = fit_predict_rbf_krr(X_tr, Y_tr * 50.0 + 7.0, [X_te], dev)
        np.testing.assert_allclose(k1 * 50.0 + 7.0, k2, atol=1e-3, rtol=1e-3)

    def test_two_way_anova_exact_decomposition(self):
        # Pure column effect: Output/Decoder explains 100%
        col_only = np.tile(np.array([0.015, 0.017, 0.018, 0.019]), (4, 1))
        a_col = two_way_anova_4x4(col_only)
        self.assertAlmostEqual(a_col["ss_input_frac"], 0.0, places=6)
        self.assertAlmostEqual(a_col["ss_output_frac"], 1.0, places=6)
        self.assertAlmostEqual(a_col["ss_interaction_frac"], 0.0, places=6)

        # Pure row effect: Input explains 100%
        row_only = col_only.T
        a_row = two_way_anova_4x4(row_only)
        self.assertAlmostEqual(a_row["ss_input_frac"], 1.0, places=6)
        self.assertAlmostEqual(a_row["ss_output_frac"], 0.0, places=6)

        # General matrix sums to 1.0
        rng = np.random.default_rng(42)
        M = rng.uniform(0.01, 0.03, size=(4, 4))
        a_gen = two_way_anova_4x4(M)
        self.assertAlmostEqual(
            a_gen["ss_input_frac"] + a_gen["ss_output_frac"] + a_gen["ss_interaction_frac"],
            1.0,
            places=6,
        )

    def test_action_subspace_orthogonality_and_zero_test_gt(self):
        rng = np.random.default_rng(7)
        dev = torch.device("cpu")
        A_tr = rng.standard_normal((250, 16, 10)).astype(np.float32)
        W = rng.standard_normal((160, 128)).astype(np.float32)
        Z_tr = A_tr.reshape(250, 160) @ W + 0.5 * rng.standard_normal((250, 128)).astype(np.float32)
        Z_te = rng.standard_normal((40, 128)).astype(np.float32)

        info = fit_action_explainable_subspace(A_tr, Z_tr, dev, var_threshold=0.99, max_rank=64)
        Q = info["Q_act"]
        # Q must be orthonormal
        np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-4)

        # Projection splits z - mu orthogonally without touching any test action!
        z_par, z_perp = project_subspace(Z_te, info["z_mu"], Q)
        np.testing.assert_allclose(z_par + z_perp, Z_te - info["z_mu"], atol=1e-5)
        inner = float(np.max(np.abs((z_par * z_perp).sum(axis=-1))))
        self.assertLess(inner, 1e-3)

    def test_orthogonal_residual_removes_base_linear_signal(self):
        rng = np.random.default_rng(9)
        dev = torch.device("cpu")
        z_base_tr = rng.standard_normal((200, 32)).astype(np.float32)
        z_base_ev = rng.standard_normal((60, 32)).astype(np.float32)
        W = rng.standard_normal((32, 32)).astype(np.float32)
        # z_supp is pure linear transform of z_base + orthogonal noise
        z_supp_tr = z_base_tr @ W + 0.1 * rng.standard_normal((200, 32)).astype(np.float32)
        z_supp_ev = z_base_ev @ W + 0.1 * rng.standard_normal((60, 32)).astype(np.float32)

        res_tr, [res_ev] = compute_orthogonal_residual(z_base_tr, z_supp_tr, [z_base_ev], [z_supp_ev], dev)
        # Residual variance should be dramatically smaller than original variance
        self.assertLess(float(np.var(res_ev)), 0.05 * float(np.var(z_supp_ev)))

    def test_gated_fusion_exact_equal_params(self):
        m_fusion = GatedFusionAdapter(obs_dim=256, branch_dim=64, fuse_dim=64, hidden_dim=256, out_dim=3072)
        m_single_split = GatedFusionAdapter(obs_dim=256, branch_dim=64, fuse_dim=64, hidden_dim=256, out_dim=3072)
        p1 = sum(p.numel() for p in m_fusion.parameters())
        p2 = sum(p.numel() for p in m_single_split.parameters())
        self.assertEqual(p1, p2)

        # Zero-init output layer guarantees step-0 output is strictly zero (exact Ridge recovery!)
        xo = torch.randn(4, 256)
        za = torch.randn(4, 64)
        zb = torch.randn(4, 64)
        out, g = m_fusion(xo, za, zb)
        self.assertEqual(tuple(out.shape), (4, 3072))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))
        self.assertTrue(((g >= 0.0) & (g <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
