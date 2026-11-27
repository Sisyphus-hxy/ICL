"""Unit tests for E9 Reader Inference-only Diagnosis readouts and metrics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from e7.model import SimpleBehaviorReader
from e9.reader_diagnosis import (
    UNIQUE_READER_IDS,
    apply_reader_readout,
    compute_base_attention_geometry,
    compute_trajectory_peak_jumps,
    label_episode_behavioral_phases,
)


class TestReaderDiagnosisReadouts(unittest.TestCase):
    def test_all_readouts_exact_properties(self) -> None:
        torch.manual_seed(0)
        # Constructed logits with non-adjacent top-2 (indices 0 and 5) vs strong adjacent pair (2, 3)
        # Row 0: s = [3.0, -5.0, 2.5, 2.4, -5.0, 2.6] -> softmax:
        #   idx 0 has highest single peak (3.0), idx 5 is 2nd highest (2.6) -> non-adjacent!
        #   Adjacent pair (2, 3) has e^2.5 + e^2.4 = 12.18 + 11.02 = 23.20 > e^3.0 + e^-5 = 20.09!
        #   So Adjacent Top-2 MUST pick (2, 3), while Global Top-1 picks 0, and Global Top-2 picks {0, 5}!
        logits = torch.tensor(
            [
                [3.0, -5.0, 2.5, 2.4, -5.0, 2.6],
                [1.0, 2.0, 3.0, 4.0, -10.0, -10.0],
            ],
            dtype=torch.float32,
        )
        pad = torch.tensor(
            [
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
            ],
            dtype=torch.bool,
        )

        # 1. Global Soft == Temp 1.0
        w_gs = apply_reader_readout(logits, "global_soft", mem_pad_mask=pad)
        w_t1 = apply_reader_readout(logits, "temp_tau_1p00", mem_pad_mask=pad)
        self.assertTrue(torch.allclose(w_gs, w_t1, atol=0.0, rtol=0.0))
        self.assertTrue(torch.all(w_gs[1, 4:] == 0.0))

        # 2. Global Top-1
        w_top1 = apply_reader_readout(logits, "global_top1", mem_pad_mask=pad)
        self.assertEqual(int(w_top1[0].argmax()), 0)
        self.assertAlmostEqual(float(w_top1[0, 0]), 1.0, places=6)
        self.assertEqual(int(w_top1[1].argmax()), 3)
        self.assertAlmostEqual(float(w_top1[1, 3]), 1.0, places=6)

        # 3. Global Top-2 (non-adjacent allowed)
        w_top2 = apply_reader_readout(logits, "global_top2", mem_pad_mask=pad)
        nz0 = torch.nonzero(w_top2[0] > 1e-6, as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(nz0, [0, 5])
        expected_w0 = w_gs[0, 0] / (w_gs[0, 0] + w_gs[0, 5])
        self.assertAlmostEqual(float(w_top2[0, 0]), float(expected_w0), places=6)
        self.assertAlmostEqual(float(w_top2[0].sum()), 1.0, places=6)

        # 4. Adjacent Top-2 (max adjacent weight sum -> picks pair (2, 3)!)
        w_adj2 = apply_reader_readout(logits, "adjacent_top2", mem_pad_mask=pad)
        nz_adj2 = torch.nonzero(w_adj2[0] > 1e-6, as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(nz_adj2, [2, 3])
        exp_a2 = w_gs[0, 2] / (w_gs[0, 2] + w_gs[0, 3])
        self.assertAlmostEqual(float(w_adj2[0, 2]), float(exp_a2), places=6)
        nz_adj2_r1 = torch.nonzero(w_adj2[1] > 1e-6, as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(nz_adj2_r1, [2, 3])
        self.assertTrue(torch.all(w_adj2[1, 4:] == 0.0))

        # 5. Adjacent Top-3 (max contiguous 3-slot window -> picks (1, 2, 3) or (2, 3, 4))
        w_adj3 = apply_reader_readout(logits, "adjacent_top3", mem_pad_mask=pad)
        nz_adj3_r1 = torch.nonzero(w_adj3[1] > 1e-6, as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(nz_adj3_r1, [1, 2, 3])
        exp_a3 = w_gs[1, 1:4] / w_gs[1, 1:4].sum()
        self.assertTrue(torch.allclose(w_adj3[1, 1:4], exp_a3, atol=1e-6))
        self.assertTrue(torch.all(w_adj3[1, 4:] == 0.0))

        # 6. Temperature sharpness monotonicity
        w_t025 = apply_reader_readout(logits, "temp_tau_0p25", mem_pad_mask=pad)
        w_t050 = apply_reader_readout(logits, "temp_tau_0p50", mem_pad_mask=pad)
        self.assertGreater(float(w_t025[0].max()), float(w_t050[0].max()))
        self.assertGreater(float(w_t050[0].max()), float(w_gs[0].max()))

    def test_short_sequence_edge_cases(self) -> None:
        # Valid lengths 1 and 2 inside padded width 4
        logits = torch.tensor(
            [
                [1.5, 0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        pad = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        )
        for mode in UNIQUE_READER_IDS:
            w = apply_reader_readout(logits, mode, mem_pad_mask=pad)
            self.assertTrue(torch.all(w[pad] == 0.0), f"pad leak in {mode}")
            self.assertTrue(torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-6), f"sum!=1 in {mode}")
            self.assertAlmostEqual(float(w[0, 0]), 1.0, places=6, msg=f"len=1 failed in {mode}")

    def test_matches_simple_behavior_reader_forward(self) -> None:
        torch.manual_seed(7)
        reader = SimpleBehaviorReader(num_query_tokens=4, embed_dim=32, attn_dim=32).eval()
        h_q = torch.randn(3, 4, 32)
        h_mem = torch.randn(3, 7, 4, 32)
        b_mem = torch.randn(3, 7, 4, 32)
        pad = torch.zeros(3, 7, dtype=torch.bool)
        pad[1, 5:] = True
        z_ref, w_ref = reader(h_q, h_mem, b_mem, mem_pad_mask=pad)

        q = reader.encode_query(h_q)
        k = reader.encode_keys(h_mem)
        logits = torch.einsum("bd,bmd->bm", q.float(), k.float()) * reader.scale
        w_new = apply_reader_readout(logits, "global_soft", mem_pad_mask=pad)
        z_new = torch.einsum("bm,bmkd->bkd", w_new, b_mem)
        self.assertTrue(torch.allclose(w_ref, w_new, atol=1e-6))
        self.assertTrue(torch.allclose(z_ref, z_new, atol=1e-6))

    def test_phase_labels_and_peak_jumps(self) -> None:
        # Open [0..39], Closed [40..89], Open [90..127]
        grip = np.full(128, -1.0, dtype=np.float32)
        grip[40:90] = 1.0
        anchors = np.arange(0, 128, 16, dtype=np.int32)
        ph = label_episode_behavioral_phases(grip, anchors)
        self.assertEqual(ph["flips"], [40, 90])
        self.assertEqual(ph["semantic_phases"][0], "approach")
        self.assertEqual(ph["semantic_phases"][2], "grasp_transition")   # [32..48) contains 40
        self.assertEqual(ph["semantic_phases"][3], "transport_carry")    # [48..64) closed
        self.assertEqual(ph["semantic_phases"][5], "release_transition") # [80..96) contains 90
        self.assertEqual(ph["semantic_phases"][7], "post_release_transit")

        # Geometry check
        g = compute_base_attention_geometry(np.array([0.5, 0.05, 0.05, 0.4]))
        self.assertEqual(g["top1_idx"], 0)
        self.assertEqual(g["top2_idx"], 3)
        self.assertEqual(g["top2_adj"], 0)
        self.assertEqual(g["top2_dist"], 3)

        # Peak jump check: peaks [0, 1, 5, 3] -> diffs [+1, +4, -2]
        recs = [
            {"traj_id": "t0", "task": "a", "anchor_idx": i, "peak_j": p, "exp_j": float(p),
             "ref_j": i, "peak_err_slots": abs(p - i), "D2_mse10_masked": 0.01, "D4_mse10_masked": 0.01}
            for i, p in enumerate([0, 1, 5, 3])
        ]
        agg, _ = compute_trajectory_peak_jumps(recs)
        self.assertAlmostEqual(agg["mean_abs_peak_step"], (1 + 4 + 2) / 3.0, places=6)
        self.assertAlmostEqual(agg["peak_jump_ge3_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(agg["backward_jump_rate"], 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
