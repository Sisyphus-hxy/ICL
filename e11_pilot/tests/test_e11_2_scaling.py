"""Unit tests for E11.2 (28.6M E10-C-aligned Transfer Network Scaling)."""

from __future__ import annotations

import unittest

import torch

from e11_pilot.data import load_frozen_e10c_and_basis
from e11_pilot.e11_2_scaling import (
    build_e10_aligned_28m_transfer,
    get_transfer_optim_groups,
)


class TestE112Scaling(unittest.TestCase):
    def test_28m_architecture_matches_e10c_exactly(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy, u64_t, meta = load_frozen_e10c_and_basis(device=device)
        e10c_params = sum(p.numel() for p in policy.transfer.parameters())

        model = build_e10_aligned_28m_transfer(dropout=0.15, latent_scale_s_z=meta["s_z"]).to(device)
        built_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(built_params, 28_562_752)
        self.assertEqual(built_params, e10c_params)
        self.assertEqual(sum(p.numel() for p in policy.parameters() if p.requires_grad), 0)

        # Zero-init -> exact c = 0 at step 0
        z_src = torch.randn(4, 4, 768, device=device)
        h_q = torch.randn(4, 4, 768, device=device)
        c0 = model(z_src, h_q)
        self.assertEqual(tuple(c0.shape), (4, 64))
        self.assertAlmostEqual(float(c0.abs().max().item()), 0.0, places=7)

        # Verify optim groups cover 100% of trainable params
        groups = get_transfer_optim_groups(model, lr=1e-4, weight_decay=0.01)
        grouped_cnt = sum(p.numel() for g in groups for p in g["params"])
        self.assertEqual(grouped_cnt, built_params)


if __name__ == "__main__":
    unittest.main()
