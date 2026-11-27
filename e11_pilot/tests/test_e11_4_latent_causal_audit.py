"""Unit tests for E11.4 (Latent Causal Audit)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from e11_pilot.data import load_frozen_e10c_and_basis
from e11_pilot.e11_4_latent_causal_audit import (
    aggregate_condition_metrics,
    build_cross_task_random_permutation,
    compose_p_plus_q,
    decode_and_collect,
    project_p_and_q,
    sample_source_swap_episodes,
    trajectory_cluster_bootstrap_contrast,
    verify_projector_and_interventions,
)
from e11_pilot.oracle_opt import decode_and_score_with_phases


class TestE114LatentCausalAudit(unittest.TestCase):
    def test_projector_and_intervention_invariants(self) -> None:
        rng = np.random.default_rng(0)
        G = rng.standard_normal((3072, 64))
        u64, _ = np.linalg.qr(G, mode="reduced")

        z_q = rng.standard_normal((32, 3072)).astype(np.float32)
        z_src = rng.standard_normal((32, 3072)).astype(np.float32)
        z_other = rng.standard_normal((32, 3072)).astype(np.float32)
        c_hat = rng.standard_normal((32, 64)).astype(np.float32)

        z_b = compose_p_plus_q(z_q, z_src, u64)
        z_c = z_q
        z_d = compose_p_plus_q(z_q, z_other, u64)

        # Learned transfer fixed-c invariance
        z_hat_orig = (z_src.astype(np.float64) + c_hat.astype(np.float64) @ u64.T).astype(np.float32)
        z_l_b = compose_p_plus_q(z_hat_orig, z_src, u64)
        z_l_d = compose_p_plus_q(z_hat_orig, z_other, u64)
        self.assertLess(float(np.max(np.abs(z_l_b - z_hat_orig))), 1e-5)

        audit = verify_projector_and_interventions(
            u64, z_q, z_src, z_other, z_b, z_c, z_d, z_learned_b=z_l_b, z_learned_d=z_l_d
        )
        self.assertLess(audit["u64_gram_max_err"], 1e-12)
        self.assertLess(audit["p_diff_B_vs_C_max"], 1e-5)
        self.assertLess(audit["p_diff_B_vs_D_max"], 1e-5)
        self.assertLess(audit["learned_p_diff_B_vs_D_max"], 1e-5)
        self.assertLess(audit["pq_orthogonality_cos_max"], 1e-5)
        self.assertGreater(audit["q_swap_l2_diff_mean"], 10.0)

        # Verify P^2 = P and Q^2 = Q
        pz, qz = project_p_and_q(z_q, u64)
        ppz, qpz = project_p_and_q(pz, u64)
        self.assertLess(float(np.max(np.abs(ppz - pz))), 1e-5)
        self.assertLess(float(np.max(np.abs(qpz))), 1e-5)

    def test_source_swap_and_cross_task_permutation(self) -> None:
        demos = []
        idx = 0
        for t in ("t0", "t1", "t2"):
            for lay in range(5):
                demos.append({"task": t, "procedure": f"p_{t}", "layout": lay, "fold": "val", "ep": idx})
                idx += 1
        pairs = [
            {"task": "t0", "procedure": "p_t0", "q_ep": 0, "s_ep": 1, "q_layout": 0, "s_layout": 1},
            {"task": "t1", "procedure": "p_t1", "q_ep": 5, "s_ep": 6, "q_layout": 0, "s_layout": 1},
            {"task": "t2", "procedure": "p_t2", "q_ep": 10, "s_ep": 11, "q_layout": 0, "s_layout": 1},
        ]
        s_other, s_lays, meta = sample_source_swap_episodes(
            pairs, demos, fold_filter="val", layout_key="layout", has_procedure=True, seed=123
        )
        self.assertEqual(meta["same_task_rate"], 1.0)
        self.assertEqual(meta["same_procedure_rate"], 1.0)
        self.assertEqual(meta["strictly_3rd_layout_rate"], 1.0)
        for i, p in enumerate(pairs):
            self.assertNotIn(int(s_other[i]), (p["q_ep"], p["s_ep"]))
            self.assertNotIn(int(s_lays[i]), (p["q_layout"], p["s_layout"]))
            self.assertEqual(demos[int(s_other[i])]["task"], p["task"])

        perm = build_cross_task_random_permutation(pairs * 10, seed=99)
        self.assertEqual(len(set(perm.tolist())), 30)
        p30 = pairs * 10
        for i in range(30):
            self.assertNotEqual(p30[int(perm[i])]["task"], p30[i]["task"])

    def test_trajectory_cluster_bootstrap_coverage(self) -> None:
        rng = np.random.default_rng(7)
        # 50 trajectories, 10 samples each, true constant shift +0.02 + trajectory noise
        G = 50
        traj_ids = np.repeat(np.arange(G), 10)
        v16 = np.full(G * 10, 16.0)
        base = rng.uniform(0.01, 0.05, size=G * 10) * v16
        traj_shift = np.repeat(0.02 + rng.normal(0.0, 0.004, size=G), 10) * v16
        sq_x = base + traj_shift
        sq_y = base
        res = trajectory_cluster_bootstrap_contrast(
            sq_x, sq_y, v16, traj_ids, np.arange(G * 10), gap_denom=0.10, n_boot=2000, seed=42
        )
        self.assertEqual(res["n_trajectories"], G)
        self.assertTrue(res["ci95_excludes_zero"])
        self.assertGreater(res["ci95_lo"], 0.015)
        self.assertLess(res["ci95_hi"], 0.025)
        self.assertAlmostEqual(res["pct_of_transfer_gap"], 100.0 * res["diff"] / 0.10, places=5)

    def test_decode_and_collect_matches_oracle_opt(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy, u64_t, meta = load_frozen_e10c_and_basis(device=device)
        u64_f64 = u64_t.detach().cpu().numpy().astype(np.float64)
        s_z = float(meta["s_z"])

        rng = np.random.default_rng(11)
        B = 8
        z_q = rng.standard_normal((B, 3072)).astype(np.float32) * 0.5
        z_src = rng.standard_normal((B, 3072)).astype(np.float32) * 0.5
        z_hat = compose_p_plus_q(z_q, z_src, u64_f64)
        gt = rng.standard_normal((B, 16, 10)).astype(np.float32) * 0.2
        v16 = np.ones((B, 16), dtype=bool)
        v16[0, 12:] = False
        noise = torch.randn((B, 16, 10), generator=torch.Generator().manual_seed(77))
        tasks = ["t0"] * 4 + ["t1"] * 4
        phases = ["approach", "grasp_transition", "transport_carry", "release_transition"] * 2

        ov_ref, ph_ref = decode_and_score_with_phases(
            policy,
            z_hat,
            z_q,
            z_src,
            gt,
            v16,
            noise,
            tasks,
            phases,
            s_z=s_z,
            device=device,
            use_bf16=True,
            batch_size=4,
        )
        col = decode_and_collect(
            policy, z_hat, gt, v16, noise, device=device, use_bf16=True, batch_size=4
        )
        ov_new, ph_new = aggregate_condition_metrics(
            col, z_hat, z_q, z_src, u64_f64, tasks, phases, s_z=s_z
        )
        for k in ("ddim_10d_mse", "ddim_pos_mse", "ddim_rot6d_mse", "ddim_grip_mse", "ddim_7d_mae", "grip_acc16", "latent_lz"):
            self.assertAlmostEqual(ov_new[k], ov_ref[k], places=6, msg=f"Mismatch on {k}")
        for ph in ("approach", "grasp_transition", "transport_carry", "release_transition"):
            self.assertAlmostEqual(
                ph_new[ph]["ddim_10d_mse"], ph_ref[ph]["ddim_10d_mse"], places=6, msg=f"Mismatch on phase {ph}"
            )


if __name__ == "__main__":
    unittest.main()
