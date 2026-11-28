"""Unit tests for E17 modules (relation authenticity helpers, paired stats, pure-video models)."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from e17.exp1_relation_authenticity import _gini, _partial_spearman
from e17.exp2_emergent_factors import _dci_disentanglement_completeness, _roc_auc_binary
from e17.pretrain_e17 import E17PureVideoModel
from e17.stats_paired import paired_bootstrap_stats


class TestE17Core(unittest.TestCase):
    def test_gini_and_partial_spearman(self) -> None:
        self.assertAlmostEqual(_gini(np.ones(100)), 0.0, places=5)
        spike = np.zeros(100)
        spike[0] = 100.0
        self.assertGreater(_gini(spike), 0.95)

        rng = np.random.default_rng(0)
        z = rng.standard_normal(2000)
        x = z + 0.5 * rng.standard_normal(2000)
        y = z + 0.5 * rng.standard_normal(2000)
        # x and y are conditionally independent given z
        self.assertLess(abs(_partial_spearman(x, y, z)), 0.08)

    def test_dci_and_roc_auc(self) -> None:
        eye = np.eye(4)
        d, c = _dci_disentanglement_completeness(eye)
        self.assertAlmostEqual(d, 1.0, places=5)
        self.assertAlmostEqual(c, 1.0, places=5)

        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(_roc_auc_binary(scores, labels), 1.0, places=5)

    def test_paired_bootstrap_stats(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.uniform(0.02, 0.04, size=100)
        b = a - 0.005 + rng.normal(0.0, 0.001, size=100)
        st = paired_bootstrap_stats(a, b, lower_is_better=True, n_boot=2000, seed=1)
        self.assertTrue(st["significant_95ci"])
        self.assertLess(st["delta_B_minus_A"], -0.004)
        self.assertGreater(st["win_rate_B_over_A"], 0.95)
        self.assertLess(st["wilcoxon_p"], 1e-10)

    def test_e17_pure_video_forward_zero_actions(self) -> None:
        B = 8
        clip_rgb = torch.randn(B, 16, 2, 768)
        clip_mask = torch.zeros(B, 16, dtype=torch.bool)
        clip_mo = torch.randn(B, 16, 2, 64)
        ep_ids = torch.arange(B, dtype=torch.long)

        for var in ("e17_m1_joint_op", "e17_m2_compact_op", "e17_m3_dual_orth", "lam_vid_op_eqparam"):
            m = E17PureVideoModel(variant=var, predict_motion=True, top_k_donors=4)
            b_t = m.encode_local_behavior(clip_rgb, clip_mask)
            self.assertEqual(tuple(b_t.shape), (B, 4, 768))
            lam_l, _ = m.compute_lam_loss(b_t, clip_rgb, clip_mask, clip_mo)
            self.assertTrue(torch.isfinite(lam_l))
            if m.reuse_module is not None:
                z_op = b_t[:, :, :384]
                z_route = b_t[:, :, 192:384] if var == "e17_m3_dual_orth" else z_op
                reu_l, _ = m.reuse_module.forward_sharp_loss(
                    z_route,
                    z_op,
                    clip_rgb,
                    clip_mask,
                    ep_ids,
                    clip_motion=clip_mo,
                    encoder_fn=m.encode_local_behavior,
                    extract_pred_fn=lambda b: b[:, :, :384],
                    use_cycle=True,
                    use_equiv=True,
                )
                self.assertTrue(torch.isfinite(reu_l))


if __name__ == "__main__":
    unittest.main()
