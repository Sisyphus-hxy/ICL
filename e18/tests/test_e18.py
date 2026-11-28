"""Unit tests for E18 models, geometry relations, scaling controls, and 3D RGB-D flow math."""

import unittest

import numpy as np
import torch

from e18.cache_generalization import compute_rlbench_7d_actions, decode_rlbench_depth_rgb
from e18.geometry_relations import CVGeoMotionEncoder, compute_weighted_displacement_loss
from e18.models import CycleMimicModel, LAPAVQModel, URBEModel


class TestE18Core(unittest.TestCase):
    def test_parameter_matching_across_paradigms(self):
        m_urbe = URBEModel(compact_manifold=True, op_dim=384, use_reuse=True)
        m_lam = URBEModel(compact_manifold=True, op_dim=384, use_reuse=False)
        m_lapa = LAPAVQModel(codebook_size=256, code_dim=128, fdm_hidden=1920)
        m_cyc = CycleMimicModel(op_dim=384, fdm_hidden=1408, compact_bottleneck=True)

        # Identical LocalBehaviorEncoder across all paradigms (42.55M)
        enc_counts = [sum(p.numel() for p in m.local_encoder.parameters()) for m in (m_urbe, m_lam, m_lapa, m_cyc)]
        self.assertEqual(len(set(enc_counts)), 1)
        self.assertAlmostEqual(enc_counts[0] / 1e6, 42.55, delta=0.1)

        # Matched total pretrain parameter budget (~75M-88M)
        tot_counts = {
            "urbe": sum(p.numel() for p in m_urbe.parameters()) / 1e6,
            "compact_lam": sum(p.numel() for p in m_lam.parameters()) / 1e6,
            "lapa_vq": sum(p.numel() for p in m_lapa.parameters()) / 1e6,
            "cyclemimic": sum(p.numel() for p in m_cyc.parameters()) / 1e6,
        }
        for name, cnt in tot_counts.items():
            self.assertGreater(cnt, 70.0, f"{name} too small: {cnt:.2f}M")
            self.assertLess(cnt, 92.0, f"{name} too large: {cnt:.2f}M")

    def test_urbe_forward_and_scaling_controls(self):
        torch.manual_seed(0)
        B = 16
        rgb = torch.randn(B, 16, 2, 768)
        mask = torch.zeros(B, 16, dtype=torch.bool)
        ep_ids = torch.arange(B)
        # 2 tasks x 8 clips each -> 7 same-task diff-ep and 8 cross-task diff-ep
        task_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)

        # Test exact_cross_task_frac at fixed donor_pool_size=6 -> 50% = 3 same-task + 3 cross-task
        m = URBEModel(
            compact_manifold=True, op_dim=384, use_reuse=True,
            donor_pool_size=6, exact_cross_task_frac=0.5, top_k_donors=4,
        )
        b_t = m.encode_local_behavior(rgb, mask)
        self.assertEqual(tuple(b_t.shape), (B, 4, 768))
        l_lam, _ = m.compute_lam_loss(b_t, rgb, mask)
        l_reu, st = m.reuse_module.forward_loss(
            b_t[:, :, :384], b_t[:, :, :384], rgb, mask, ep_ids,
            task_ids=task_ids, encoder_fn=m.encode_local_behavior,
            extract_pred_fn=lambda b: b[:, :, :384], use_cycle=True, use_equiv=True,
        )
        self.assertAlmostEqual(float(st["n_valid_donors"].item()), 6.0, delta=0.01)
        (l_lam + 0.25 * l_reu).backward()

        # Test FIFO memory queue enqueue + reuse beyond batch size
        mq = URBEModel(compact_manifold=True, op_dim=384, use_reuse=True, memory_queue_size=32, top_k_donors=8)
        for _ in range(3):
            b_q = mq.encode_local_behavior(rgb, mask)
            lq, stq = mq.reuse_module.forward_loss(
                b_q[:, :, :384], b_q[:, :, :384], rgb, mask, ep_ids + 100 * _,
                use_cycle=False, use_equiv=False,
            )
            lq.backward()
        self.assertEqual(int(mq.reuse_module.queue_filled.item()), 32)
        self.assertGreater(float(stq["n_valid_donors"].item()), B)

    def test_lapa_vq_and_cyclemimic(self):
        torch.manual_seed(1)
        B = 8
        rgb = torch.randn(B, 16, 2, 768)
        mask = torch.zeros(B, 16, dtype=torch.bool)
        ep_ids = torch.arange(B)

        lapa = LAPAVQModel(codebook_size=32, code_dim=64, fdm_hidden=256)
        b_l = lapa.encode_local_behavior(rgb, mask)
        loss_l, st_l = lapa.compute_loss(b_l, rgb, mask)
        loss_l.backward()
        self.assertGreater(float(st_l["vq_perplexity"].item()), 1.0)

        cyc = CycleMimicModel(op_dim=384, fdm_hidden=256, compact_bottleneck=True)
        b_c = cyc.encode_local_behavior(rgb, mask)
        loss_c, st_c = cyc.compute_loss(b_c, rgb, mask, ep_ids, do_cycle=True)
        loss_c.backward()
        self.assertTrue(torch.isfinite(loss_c).item())
        self.assertIn("cyc_vis", st_c)

    def test_cv_geo_and_rlbench_depth_math(self):
        torch.manual_seed(2)
        B = 16
        cm = torch.randn(B, 16, 2, 64)
        enc = CVGeoMotionEncoder(in_dim=384, hidden_dim=128, out_dim=64)
        loss, st = enc.forward_loss(cm)
        loss.backward()
        u_cv, ag = enc.encode_canonical(cm)
        self.assertEqual(tuple(u_cv.shape), (B, 64))
        z = torch.randn(B, 4, 128, requires_grad=True)
        dl, _ = compute_weighted_displacement_loss(z, u_cv, pair_weights=torch.sigmoid(ag))
        dl.backward()

        # Verify 24-bit depth decode endpoints
        zeros = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        maxs = np.full((1, 2, 2, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(float(decode_rlbench_depth_rgb(zeros, 0.01, 4.5).mean()), 0.01, places=5)
        self.assertAlmostEqual(float(decode_rlbench_depth_rgb(maxs, 0.01, 4.5).mean()), 4.5, places=5)

        gp = np.zeros((10, 7), dtype=np.float32)
        gp[:, 6] = 1.0  # identity quat xyzw
        gp[:, 0] = np.linspace(0.0, 0.09, 10)
        go = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        a7 = compute_rlbench_7d_actions(gp, go)
        self.assertEqual(a7.shape, (10, 7))
        self.assertAlmostEqual(float(a7[0, 0]), 0.1, places=4)
        self.assertEqual(float(a7[0, 6]), 1.0)
        self.assertEqual(float(a7[5, 6]), -1.0)


if __name__ == "__main__":
    unittest.main()
