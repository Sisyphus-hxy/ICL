"""Unit tests for E16 self-supervised relational behavior learning modules."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from e16.extract_video_motion import GPUPyramidalFlowExtractor
from e16.pretrain_video_only import PureVideoLAMModel
from e16.video_relations import (
    LearnableReuseRelationModule,
    compute_vicreg_loss,
    compute_video_derived_displacement_loss,
    extract_video_relation_features,
)


class TestE16Modules(unittest.TestCase):
    def test_gpu_pyramidal_flow_physics(self) -> None:
        """Verify rightward shift produces positive u-flow and inward lower motion produces positive low_conv."""
        dev = torch.device("cpu")
        ext = GPUPyramidalFlowExtractor(dev)
        # Smooth multi-frequency texture (realistic spatial gradients for optical flow)
        ys = np.linspace(0, 4 * np.pi, 128)[:, None]
        xs = np.linspace(0, 4 * np.pi, 128)[None, :]
        tex = 0.5 + 0.25 * np.sin(xs) * np.cos(ys) + 0.20 * np.cos(2 * xs + ys)
        base = np.repeat((tex * 220 + 15).astype(np.uint8)[None, :, :, None], 3, axis=-1)
        # Shift image right by 2 pixels -> positive u-flow; left by 2 pixels -> negative u-flow
        shifted_r = np.roll(base, shift=2, axis=2)
        shifted_l = np.roll(base, shift=-2, axis=2)
        desc_r = ext.extract_pair_descriptors(base, shifted_r, convention="standard")
        desc_l = ext.extract_pair_descriptors(base, shifted_l, convention="standard")
        self.assertEqual(desc_r.shape, (1, 64))
        self.assertTrue(np.isfinite(desc_r).all())
        self.assertGreater(float(desc_r[0, 34]), 0.5)
        self.assertLess(float(desc_l[0, 34]), -0.5)

        # Inward lower-half motion -> positive low_conv; outward -> negative low_conv
        inward = base.copy()
        inward[:, 64:, :64, :] = np.roll(base[:, 64:, :64, :], shift=2, axis=2)
        inward[:, 64:, 64:, :] = np.roll(base[:, 64:, 64:, :], shift=-2, axis=2)
        outward = base.copy()
        outward[:, 64:, :64, :] = np.roll(base[:, 64:, :64, :], shift=-2, axis=2)
        outward[:, 64:, 64:, :] = np.roll(base[:, 64:, 64:, :], shift=2, axis=2)
        desc_in = ext.extract_pair_descriptors(base, inward, convention="standard")
        desc_out = ext.extract_pair_descriptors(base, outward, convention="standard")
        self.assertGreater(float(desc_in[0, 44]), 0.5)
        self.assertLess(float(desc_out[0, 44]), -0.5)

    def test_video_derived_displacement_loss(self) -> None:
        torch.manual_seed(0)
        B = 16
        z = torch.randn(B, 4, 384, requires_grad=True)
        clip_rgb = torch.randn(B, 16, 2, 768)
        clip_motion = torch.randn(B, 16, 2, 64)
        mask = torch.zeros(B, 16, dtype=torch.bool)
        for src in ("feat_delta", "feat_delta_decorr", "motion_flow", "random_rel"):
            u = extract_video_relation_features(src, clip_rgb, mask, clip_motion)
            loss, stats = compute_video_derived_displacement_loss(z, u)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("disp_dir_mse", stats)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_learnable_reuse_anti_scene_and_gradients(self) -> None:
        torch.manual_seed(1)
        B = 16
        clip_rgb = torch.randn(B, 16, 2, 768)
        clip_motion = torch.randn(B, 16, 2, 64)
        mask = torch.zeros(B, 16, dtype=torch.bool)
        ep_ids = torch.arange(B) // 2
        task_ids = torch.arange(B) // 4

        # Global & token-wise reuse modules
        for tok_wise in (False, True):
            for cmode in ("cross_demo_anti_scene", "strict_cross_task"):
                mod = LearnableReuseRelationModule(
                    z_rel_dim=4 * 384,
                    token_wise=tok_wise,
                    predict_motion=True,
                    coverage_mode=cmode,
                )
                b_t = torch.randn(B, 4, 768, requires_grad=True)
                loss, stats = mod.forward_loss(
                    b_t,
                    clip_rgb,
                    mask,
                    ep_ids,
                    task_ids=task_ids,
                    clip_motion=clip_motion,
                    op_dim=384,
                    use_cycle=False,
                    use_equiv=True,
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                # Check strict z_op restriction: z_res slice (384:) must have ZERO gradient!
                self.assertAlmostEqual(float(b_t.grad[:, :, 384:].abs().max().item()), 0.0, places=6)
                self.assertGreater(float(b_t.grad[:, :, :384].abs().max().item()), 0.0)

                # Verify strict_cross_task mask blocks all same-task pairs
                if cmode == "strict_cross_task":
                    vmask, _ = mod._build_donor_mask(clip_rgb[:, 0], ep_ids, task_ids=task_ids)
                    same_t = task_ids[:, None] == task_ids[None, :]
                    self.assertFalse(bool((vmask & same_t).any().item()))

    def test_pure_video_lam_zero_actions(self) -> None:
        torch.manual_seed(2)
        model = PureVideoLAMModel(variant="lam_learn_op", op_dim=384, predict_motion=True)
        B = 8
        clip_rgb = torch.randn(B, 16, 2, 768)
        clip_motion = torch.randn(B, 16, 2, 64)
        mask = torch.zeros(B, 16, dtype=torch.bool)
        ep_ids = torch.arange(B)
        b_t = model.encode_local_behavior(clip_rgb, mask)
        l_lam, _ = model.compute_lam_loss(b_t, clip_rgb, mask, clip_motion)
        l_rel, _ = model.reuse_module.forward_loss(
            b_t, clip_rgb, mask, ep_ids, clip_motion=clip_motion, encoder_fn=model.encode_local_behavior, op_dim=384
        )
        tot = l_lam + 0.25 * l_rel
        tot.backward()
        self.assertTrue(torch.isfinite(tot))
        # Verify VICReg stays finite
        l_vic, _, _ = compute_vicreg_loss(b_t.detach())
        self.assertTrue(torch.isfinite(l_vic))


if __name__ == "__main__":
    unittest.main()
