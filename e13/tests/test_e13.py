"""Unit tests for E13 validation utilities (whitening, pseudoinverse, interference, zero-leakage)."""

import unittest

import numpy as np
import torch

from e5.actions import action_7d_to_10d
from e13.exp1_training_free import _fit_linear_probe_and_pinv_gpu, _fit_whitener_gpu
from e13.exp3_factor_composition import _attr_pairwise_dist


class TestE13Utilities(unittest.TestCase):
    def test_zca_whitener_invertibility(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(200, 32)).astype(np.float32) + 2.5
        w = _fit_whitener_gpu(Z, torch.device("cpu"), mode="zca", eps=0.05)
        Zt = torch.from_numpy(Z.astype(np.float64))
        Zw = (Zt - w["mu"]) @ w["W"]
        Z_rec = Zw @ w["W_inv"] + w["mu"]
        self.assertTrue(torch.allclose(Z_rec, Zt, atol=1e-5))

    def test_linear_probe_pseudoinverse_exact_linear_map(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(size=(300, 24)).astype(np.float32)
        W_true = rng.normal(size=(24, 6)).astype(np.float32)
        A = Z @ W_true
        pr = _fit_linear_probe_and_pinv_gpu(Z, A, torch.device("cpu"), pinv_gammas=(1e-6, 1e-4))
        # Editing along dA with pinv_euc should satisfy (dA @ pinv_euc) @ W_p ≈ dA
        dA = torch.randn(10, 6, dtype=torch.float64)
        dZ_euc = dA @ pr["pinv_euc"]
        dA_rec = dZ_euc @ pr["W_p"]
        self.assertLess(float(((dA_rec - dA) ** 2).mean().item()), 1e-3)
        dZ_cov = dA @ pr["pinv_cov"]
        dA_rec_c = dZ_cov @ pr["W_p"]
        self.assertLess(float(((dA_rec_c - dA) ** 2).mean().item()), 1e-3)

    def test_attr_pairwise_dist_zero_for_identical(self):
        rng = np.random.default_rng(2)
        a7 = rng.uniform(-0.4, 0.4, size=(8, 16, 7)).astype(np.float32)
        a7[:, :, 6] = np.where(a7[:, :, 6] >= 0, 1.0, -1.0)
        a10 = action_7d_to_10d(a7)
        d = _attr_pairwise_dist(a10, a10)
        for k, v in d.items():
            self.assertLess(float(v.max()), 1e-5, f"Non-zero self-dist for {k}")


if __name__ == "__main__":
    unittest.main()
