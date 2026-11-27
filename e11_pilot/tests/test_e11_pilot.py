"""Unit tests for E11-Pilot invariants:
  1. Frozen E9 Teacher, frozen E9 Reader, frozen E10-C, exact P0 U_64 orthonormality & match.
  2. Strict zero-leakage 4-way split (train_fit vs val_ckpt vs fold=val).
  3. Oracle coefficient optimization updates ONLY `c` (zero grad in frozen teacher) and reduces loss.
  4. Coefficient geometry & effective rank calculations.
"""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from e11_pilot import E5_CACHE_DIR, RANK_K
from e11_pilot.data import load_frozen_e10c_and_basis, split_train_episodes_for_learnability
from e11_pilot.oracle_opt import (
    compute_coeff_effective_rank,
    compute_coeff_geometry,
    optimize_coefficients_batch,
)


class TestE11PilotInvariants(unittest.TestCase):
    def test_4way_split_zero_leakage(self) -> None:
        episodes = json.loads((E5_CACHE_DIR / "episodes.json").read_text())
        tr_eps, va_eps = split_train_episodes_for_learnability(episodes, val_frac=0.10, seed=1101)
        val_fold_eps = {i for i, e in enumerate(episodes) if e["fold"] == "val"}
        self.assertTrue(len(tr_eps) > 10000)
        self.assertTrue(len(va_eps) > 500)
        self.assertTrue(tr_eps.isdisjoint(va_eps))
        self.assertTrue(tr_eps.isdisjoint(val_fold_eps))
        self.assertTrue(va_eps.isdisjoint(val_fold_eps))

    def test_coeff_geometry_and_erank(self) -> None:
        rng = np.random.default_rng(7)
        c_p0 = rng.standard_normal((128, RANK_K)).astype(np.float32) * 3.0
        c_zero = np.zeros_like(c_p0)
        g0 = compute_coeff_geometry(c_zero, c_p0, c_p0, s_z=1.0)
        self.assertAlmostEqual(g0["c_l2_mean"], 0.0, places=6)
        self.assertAlmostEqual(g0["cos_vs_c_p0"], 0.0, places=6)
        g1 = compute_coeff_geometry(c_p0, c_p0, c_p0, s_z=1.0)
        self.assertAlmostEqual(g1["cos_vs_c_p0"], 1.0, places=5)
        self.assertAlmostEqual(g1["c_norm_ratio_vs_p0"], 1.0, places=5)
        er = compute_coeff_effective_rank(c_p0)
        self.assertGreater(er["erank_energy"], 30.0)
        self.assertLessEqual(er["erank_energy"], float(RANK_K) + 1e-3)

    def test_freeze_and_oracle_step_gpu(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy, u64_t, meta = load_frozen_e10c_and_basis(device=device)
        self.assertLess(meta["basis_ortho_err"], 1e-4)
        self.assertLess(meta["basis_vs_e10c_max_abs_diff"], 1e-6)
        self.assertEqual(sum(p.numel() for p in policy.parameters() if p.requires_grad), 0)

        rng = np.random.default_rng(123)
        z_src = rng.standard_normal((16, 3072)).astype(np.float32) * 0.5
        gt_act = rng.uniform(-0.5, 0.5, size=(16, 16, 10)).astype(np.float32)
        c_opt, curves = optimize_coefficients_batch(
            policy.teacher,
            z_src,
            gt_act,
            u64_t,
            lam=1e-5,
            steps=15,
            lr=0.5,
            mc_timesteps=2,
            batch_size=16,
            seed=42,
            device=device,
            use_bf16=(device.type == "cuda"),
            log_steps=(0, 15),
        )
        self.assertEqual(c_opt.shape, (16, RANK_K))
        self.assertTrue(np.all(np.isfinite(c_opt)))
        self.assertGreater(float(np.linalg.norm(c_opt)), 0.0)
        # Ensure frozen teacher got ZERO gradients
        self.assertFalse(any(p.grad is not None for p in policy.teacher.parameters()))
        # Objective should drop from step 0 to step 15
        self.assertLess(curves[-1]["total_obj"], curves[0]["total_obj"])


if __name__ == "__main__":
    unittest.main()
