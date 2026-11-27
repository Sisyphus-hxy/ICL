"""Unit tests for E12 action attribute decomposition and relational utilities."""

import unittest

import numpy as np
import torch

from e5.actions import action_7d_to_10d
from e12.action_attributes import (
    compute_attribute_similarity_matrix,
    discretize_behavior_factors,
    extract_action_attribute_vectors,
    extract_action_delta_features,
)


class TestActionAttributes(unittest.TestCase):
    def test_extract_shapes_and_invariants(self):
        rng = np.random.default_rng(0)
        a7 = rng.uniform(-0.5, 0.5, size=(16, 16, 7)).astype(np.float32)
        a7[:, :, 6] = np.where(a7[:, :, 6] >= 0, 1.0, -1.0)
        a10 = action_7d_to_10d(a7)

        attrs = extract_action_attribute_vectors(torch.from_numpy(a10))
        self.assertEqual(attrs["global_raw7d"].shape, (16, 112))
        self.assertEqual(attrs["global_balanced7d"].shape, (16, 112))
        self.assertEqual(attrs["global_10d"].shape, (16, 160))
        self.assertEqual(attrs["translation"].shape, (16, 48))
        self.assertEqual(attrs["rotation"].shape, (16, 48))
        self.assertEqual(attrs["gripper"].shape, (16, 16))
        self.assertEqual(attrs["magnitude"].shape, (16, 37))

        for name in (
            "global_raw7d",
            "global_balanced7d",
            "global_10d",
            "translation",
            "rotation",
            "magnitude",
            "temporal",
            "gripper",
        ):
            S = compute_attribute_similarity_matrix(attrs, name)
            self.assertEqual(S.shape, (16, 16))
            self.assertTrue(torch.isfinite(S).all())
            self.assertTrue(torch.allclose(S, S.T, atol=1e-5))

    def test_magnitude_direction_invariance_and_temporal_scale_invariance(self):
        # Sample 0: moves +x with speed profile v; Sample 1: moves +y with same v;
        # Sample 2: moves +x with 2*v
        t = np.linspace(0.1, 0.5, 16, dtype=np.float32)
        a7 = np.zeros((3, 16, 7), dtype=np.float32)
        a7[0, :, 0] = t
        a7[1, :, 1] = t
        a7[2, :, 0] = 2.0 * t
        a7[:, :, 6] = -1.0
        a10 = torch.from_numpy(action_7d_to_10d(a7))
        attrs = extract_action_attribute_vectors(a10)

        # Sample 0 and 1 have identical magnitude despite orthogonal translation!
        self.assertTrue(torch.allclose(attrs["magnitude"][0], attrs["magnitude"][1], atol=1e-5))
        # Sample 0 and 2 have identical temporal shape despite 2x speed!
        self.assertTrue(torch.allclose(attrs["temporal"][0], attrs["temporal"][2], atol=1e-5))
        # Sample 0 and 1 have orthogonal translation
        S_tr = compute_attribute_similarity_matrix(attrs, "translation")
        self.assertAlmostEqual(float(S_tr[0, 1].item()), 0.0, places=4)
        self.assertAlmostEqual(float(S_tr[0, 2].item()), 1.0, places=4)

    def test_discretize_and_delta(self):
        a7 = np.zeros((4, 16, 7), dtype=np.float32)
        a7[0, :, 0] = 0.2
        a7[0, :, 6] = -1.0
        a7[1, :, 2] = -0.3
        a7[1, :, 6] = 1.0
        a7[2, :, 1] = 0.25
        a7[2, :8, 6] = -1.0
        a7[2, 8:, 6] = 1.0
        a7[3, :, 6] = 1.0
        a7[3, 8:, 6] = -1.0
        f = discretize_behavior_factors(a7)
        self.assertEqual(f["dir_label"].tolist(), ["+x", "-z", "+y", "still"])
        self.assertEqual(f["grip_label"].tolist(), ["open", "closed", "closing", "opening"])

        da = extract_action_delta_features(torch.from_numpy(a7[:2]), torch.from_numpy(a7[2:]))
        self.assertEqual(da["delta_balanced7d"].shape, (2, 112))

    def test_relational_losses_grad_and_finiteness(self):
        from e12.losses import compute_relational_loss

        rng = np.random.default_rng(7)
        B = 16
        a7 = rng.uniform(-0.6, 0.6, size=(B, 16, 7)).astype(np.float32)
        a7[:, :, 6] = np.where(a7[:, :, 6] >= 0, 1.0, -1.0)
        a10 = torch.from_numpy(action_7d_to_10d(a7))
        a7_t = torch.from_numpy(a7)

        for variant in ("control", "e12_a", "e12_b", "e12_c"):
            b_t = torch.randn(B, 4, 64, requires_grad=True)
            loss, stats = compute_relational_loss(variant, b_t, a10, a7_t)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(b_t.grad)
            self.assertTrue(torch.isfinite(b_t.grad).all())
            if variant != "control":
                self.assertGreater(float(b_t.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
