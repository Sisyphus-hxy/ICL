"""Unit tests for E7: Behavior Memory Retrieval + Latent Adaptation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from e6.stage1_model import E6Stage1Policy
from e7 import ACTION_DIM_10D, ACTION_HORIZON, EMBED_DIM, FIXED_LATENT_SCALE_S_Z, NUM_QUERY_TOKENS
from e7.model import (
    E7Policy,
    SimpleBehaviorReader,
    compute_attention_stats,
    compute_behavior_anchor_steps,
    load_e7_policy,
)
from e7.train import get_cosine_schedule_with_min_ratio


class TestE7(unittest.TestCase):
    def _build_dummy_e7(self, s_z: float = FIXED_LATENT_SCALE_S_Z) -> E7Policy:
        torch.manual_seed(0)
        teacher = E6Stage1Policy(use_attention_sink=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return E7Policy(teacher, latent_scale_s_z=s_z)

    def test_stride16_anchor_steps_and_tail_windows(self):
        # 192-step Demo -> exact 12 behavior latents [0, 16, ..., 176]
        s192 = compute_behavior_anchor_steps(192, stride=16)
        self.assertEqual(len(s192), 12)
        np.testing.assert_array_equal(s192, np.arange(0, 192, 16))

        # 195-step Demo -> 13 behavior latents including tail window at 192
        s195 = compute_behavior_anchor_steps(195, stride=16)
        self.assertEqual(len(s195), 13)
        self.assertEqual(int(s195[-1]), 192)

        # Short Demo (<16 steps) -> 1 behavior latent at 0
        s7 = compute_behavior_anchor_steps(7, stride=16)
        np.testing.assert_array_equal(s7, np.array([0], dtype=np.int32))

    def test_simple_reader_shared_weights_no_v_proj_and_pad_masking(self):
        torch.manual_seed(1)
        reader = SimpleBehaviorReader()

        # Verify ONLY Q/K parameters exist in reader (no V proj, no out proj, no skip)
        param_names = set(dict(reader.named_parameters()).keys())
        self.assertEqual(
            param_names,
            {
                "q_norm.weight",
                "q_norm.bias",
                "k_norm.weight",
                "k_norm.bias",
                "q_proj.weight",
                "q_proj.bias",
                "k_proj.weight",
                "k_proj.bias",
            },
        )

        B, M = 3, 7
        h_q = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        h_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        pad_mask = torch.zeros(B, M, dtype=torch.bool)
        pad_mask[0, 5:] = True
        pad_mask[1, 3:] = True

        # Corrupt padded b_mem and h_mem entries with huge values -> output must be invariant!
        h_mem_corrupt = h_mem.clone()
        b_mem_corrupt = b_mem.clone()
        h_mem_corrupt[pad_mask] = 1e4
        b_mem_corrupt[pad_mask] = -1e4

        z_c1, w1 = reader(h_q, h_mem, b_mem, mem_pad_mask=pad_mask)
        z_c2, w2 = reader(h_q, h_mem_corrupt, b_mem_corrupt, mem_pad_mask=pad_mask)

        self.assertEqual(w1.shape, (B, M))
        self.assertEqual(z_c1.shape, (B, NUM_QUERY_TOKENS, EMBED_DIM))
        torch.testing.assert_close(w1, w2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(z_c1, z_c2, atol=1e-6, rtol=1e-6)

        # Padded weights are strictly 0 and valid weights sum to 1
        self.assertEqual(float(w1[pad_mask].abs().max().item()), 0.0)
        torch.testing.assert_close(w1.sum(dim=-1), torch.ones(B), atol=1e-6, rtol=1e-6)

        # Entire (4, 768) latent is exact convex combination sum_i w_i * b_i
        manual_zc = sum(w1[:, i, None, None] * b_mem[:, i] for i in range(M))
        torch.testing.assert_close(z_c1, manual_zc, atol=1e-6, rtol=1e-6)

        # Check attention diagnostics on uniform vs peaked distributions
        w_unif = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0]])
        pm_unif = torch.tensor([[False, False, False, False, True]])
        st_u = compute_attention_stats(w_unif, pm_unif)
        self.assertAlmostEqual(float(st_u["norm_entropy"].item()), 1.0, places=5)
        self.assertAlmostEqual(float(st_u["eff_pos_exp"].item()), 4.0, places=5)
        self.assertAlmostEqual(float(st_u["eff_pos_ipr"].item()), 4.0, places=5)
        self.assertAlmostEqual(float(st_u["max_weight"].item()), 0.25, places=6)

        w_one = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        st_o = compute_attention_stats(w_one, pm_unif)
        self.assertAlmostEqual(float(st_o["norm_entropy"].item()), 0.0, places=5)
        self.assertAlmostEqual(float(st_o["eff_pos_exp"].item()), 1.0, places=5)
        self.assertAlmostEqual(float(st_o["eff_pos_ipr"].item()), 1.0, places=5)
        self.assertAlmostEqual(float(st_o["max_weight"].item()), 1.0, places=6)

    def test_zero_init_delta_and_frozen_teacher_grad_flow(self):
        policy = self._build_dummy_e7()
        policy.train(True)
        self.assertFalse(policy.teacher.training)
        self.assertTrue(policy.reader.training)
        self.assertTrue(policy.residual_transformer.training)

        teacher_snap = {k: v.detach().clone() for k, v in policy.teacher.state_dict().items()}

        B, M = 4, 6
        h_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        mem_mask = torch.zeros(B, M, dtype=torch.bool)
        mem_mask[:, -1] = True
        query_rgb = torch.randn(B, 2, 2, EMBED_DIM)
        z_self = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        # At step 0, zero-init out_proj -> delta_z == 0 identically
        policy.eval()
        z_c, h_q, _ = policy.read_cross_latent(h_mem, b_mem, mem_mask, query_rgb)
        z_q, delta_z = policy.adapt_latent(z_c, h_q)
        torch.testing.assert_close(delta_z, torch.zeros_like(delta_z), atol=0.0, rtol=0.0)
        torch.testing.assert_close(z_q, z_c, atol=0.0, rtol=0.0)

        # Pure L_diff flows through frozen UNet into BOTH SimpleReader Q/K and ResidualTransformer
        policy.train(True)
        opt = torch.optim.AdamW(
            policy.get_optim_groups(reader_lr=1e-4, residual_lr=1e-4, weight_decay=0.01),
            betas=(0.9, 0.95),
        )
        opt.zero_grad(set_to_none=True)
        out1 = policy(
            h_mem, b_mem, mem_mask, query_rgb, z_self, actions,
            lambda_a=1.0, lambda_z=0.0,
        )
        out1["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertGreater(policy.reader.q_proj.weight.grad.float().norm().item(), 0.0)
        self.assertGreater(policy.reader.k_proj.weight.grad.float().norm().item(), 0.0)
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )
        opt.step()

        # Full E7 loss: L = L_diff + 0.5 * L_z
        opt.zero_grad(set_to_none=True)
        out2 = policy(
            h_mem, b_mem, mem_mask, query_rgb, z_self, actions,
            lambda_a=1.0, lambda_z=0.5,
        )
        expected = out2["diff_loss"] + 0.5 * out2["latent_loss_norm"]
        torch.testing.assert_close(out2["loss"].detach(), expected, atol=1e-6, rtol=1e-6)
        out2["loss"].backward()
        opt.step()

        # Frozen Stage 1 Teacher must remain bit-exact
        for k, v in policy.teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Teacher param changed: {k}")

    def test_video_only_behavior_memory_extraction_and_ckpt_roundtrip(self):
        policy = self._build_dummy_e7()
        policy.eval()

        P, M = 10, 12
        demo_1hz = torch.randn(P, 2, EMBED_DIM)
        anchor_q = torch.randn(M, 2, 2, EMBED_DIM)
        h_mem, b_mem = policy.extract_behavior_memory_from_single_demo(demo_1hz, anchor_q)
        self.assertEqual(h_mem.shape, (M, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(b_mem.shape, (M, NUM_QUERY_TOKENS, EMBED_DIM))

        # Verify b_mem matches running Teacher encode_z_self on the FULL Demo for each entry
        z_ref, _ = policy.teacher.encode_z_self(
            demo_1hz.unsqueeze(0).expand(M, P, 2, EMBED_DIM), None, anchor_q
        )
        torch.testing.assert_close(b_mem, z_ref, atol=1e-5, rtol=1e-5)

        # Scheduler & checkpoint roundtrip
        opt = torch.optim.AdamW(policy.get_optim_groups(reader_lr=1e-4, residual_lr=1e-4))
        sched = get_cosine_schedule_with_min_ratio(
            opt, num_warmup_steps=500, num_training_steps=42912, min_lr_ratio=0.1
        )
        lrs = []
        for _ in range(42912):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        self.assertAlmostEqual(lrs[0], 0.0, places=10)
        self.assertAlmostEqual(lrs[500], 1e-4, places=7)
        self.assertAlmostEqual(lrs[-1], 1e-5, places=6)

        tmp_root = Path(__file__).resolve().parents[1] / "outputs"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=tmp_root) as tmp:
            s1_path = Path(tmp) / "s1.pt"
            e7_path = Path(tmp) / "e7.pt"
            torch.save({"model": policy.teacher.state_dict(), "config": {}}, s1_path)
            torch.save(
                {
                    "stage": "e7_behavior_memory_retrieval",
                    "epoch": 1,
                    "latent_scale_s_z": FIXED_LATENT_SCALE_S_Z,
                    "reader": policy.reader.state_dict(),
                    "residual_transformer": policy.residual_transformer.state_dict(),
                    "config": {"stage1_ckpt": str(s1_path)},
                },
                e7_path,
            )
            reloaded, _ = load_e7_policy(e7_path, device="cpu")
            q = torch.randn(2, 2, 2, EMBED_DIM)
            z0 = torch.randn(2, ACTION_HORIZON, ACTION_DIM_10D)
            hm_b = h_mem.unsqueeze(0).expand(2, M, NUM_QUERY_TOKENS, EMBED_DIM)
            bm_b = b_mem.unsqueeze(0).expand(2, M, NUM_QUERY_TOKENS, EMBED_DIM)
            pm_b = torch.zeros(2, M, dtype=torch.bool)
            o1 = policy.predict_action_from_behavior_memory(hm_b, bm_b, pm_b, q, init_noise=z0)
            o2 = reloaded.predict_action_from_behavior_memory(hm_b, bm_b, pm_b, q, init_noise=z0)
            torch.testing.assert_close(o1["action_10d"], o2["action_10d"], atol=1e-6, rtol=1e-6)

    def test_direct_prediction_mode_no_skip_grad_flow_and_roundtrip(self):
        torch.manual_seed(7)
        teacher = E6Stage1Policy(use_attention_sink=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        res_policy = E7Policy(teacher, prediction_mode="residual")
        mean_bias = torch.randn(1, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        dir_policy = E7Policy(
            teacher,
            prediction_mode="direct",
            direct_latent_bias=mean_bias,
            direct_out_proj_init_std=0.002,
        )

        # Identical trainable parameter counts (buffer is non-trainable)
        self.assertEqual(
            res_policy.parameter_summary()["total_trainable"],
            dir_policy.parameter_summary()["total_trainable"],
        )
        self.assertFalse(dir_policy.direct_latent_bias.requires_grad)

        B, M = 3, 5
        h_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        mem_mask = torch.zeros(B, M, dtype=torch.bool)
        mem_mask[:, -1] = True
        query_rgb = torch.randn(B, 2, 2, EMBED_DIM)
        z_self = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        # Verify NO skip connection from z_c: if out_proj == 0, z_q == direct_latent_bias identically!
        dir_zero = E7Policy(
            teacher,
            prediction_mode="direct",
            direct_latent_bias=mean_bias,
            direct_out_proj_init_std=0.0,
        )
        dir_zero.eval()
        z_c, h_q, _ = dir_zero.read_cross_latent(h_mem, b_mem, mem_mask, query_rgb)
        z_q_zero, _ = dir_zero.adapt_latent(z_c, h_q)
        torch.testing.assert_close(
            z_q_zero, mean_bias.expand(B, -1, -1), atol=1e-6, rtol=1e-6
        )
        # Must NOT equal z_c
        self.assertGreater((z_q_zero - z_c).abs().mean().item(), 1.0)

        # Verify invalid delta_scale raises in direct mode
        with self.assertRaises(ValueError):
            dir_policy.adapt_latent(z_c, h_q, delta_scale=0.5)

        # Pure L_diff flows through frozen UNet into BOTH out_proj and SimpleReader Q/K at step 0
        teacher_snap = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
        dir_policy.train(True)
        opt = torch.optim.AdamW(
            dir_policy.get_optim_groups(reader_lr=1e-4, residual_lr=1e-4, weight_decay=0.01),
            betas=(0.9, 0.95),
        )
        opt.zero_grad(set_to_none=True)
        out1 = dir_policy(
            h_mem, b_mem, mem_mask, query_rgb, z_self, actions,
            lambda_a=1.0, lambda_z=0.0,
        )
        out1["loss"].backward()
        self.assertTrue(all(p.grad is None for p in dir_policy.teacher.parameters()))
        self.assertGreater(dir_policy.reader.q_proj.weight.grad.float().norm().item(), 0.0)
        self.assertGreater(dir_policy.reader.k_proj.weight.grad.float().norm().item(), 0.0)
        self.assertGreater(
            dir_policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )
        opt.step()

        # Teacher remains bit-exact frozen, direct_latent_bias remains unchanged
        for k, v in dir_policy.teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Teacher param changed: {k}")
        torch.testing.assert_close(dir_policy.direct_latent_bias, mean_bias, atol=0.0, rtol=0.0)

        # Checkpoint roundtrip preserves prediction_mode='direct' and direct_latent_bias
        dir_policy.eval()
        tmp_root = Path(__file__).resolve().parents[1] / "outputs"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=tmp_root) as tmp:
            s1_path = Path(tmp) / "s1.pt"
            e7_path = Path(tmp) / "e7_direct.pt"
            torch.save({"model": teacher.state_dict(), "config": {}}, s1_path)
            torch.save(
                {
                    "stage": "e7_behavior_memory_retrieval",
                    "prediction_mode": "direct",
                    "direct_latent_bias": dir_policy.direct_latent_bias.detach().cpu(),
                    "epoch": 1,
                    "latent_scale_s_z": FIXED_LATENT_SCALE_S_Z,
                    "reader": dir_policy.reader.state_dict(),
                    "residual_transformer": dir_policy.residual_transformer.state_dict(),
                    "config": {
                        "stage1_ckpt": str(s1_path),
                        "prediction_mode": "direct",
                        "direct_out_proj_init_std": 0.002,
                    },
                },
                e7_path,
            )
            reloaded, _ = load_e7_policy(e7_path, device="cpu")
            self.assertEqual(reloaded.prediction_mode, "direct")
            torch.testing.assert_close(
                reloaded.direct_latent_bias, dir_policy.direct_latent_bias, atol=0.0, rtol=0.0
            )
            z0 = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
            o1 = dir_policy.predict_action_from_behavior_memory(
                h_mem, b_mem, mem_mask, query_rgb, init_noise=z0
            )
            o2 = reloaded.predict_action_from_behavior_memory(
                h_mem, b_mem, mem_mask, query_rgb, init_noise=z0
            )
            torch.testing.assert_close(o1["action_10d"], o2["action_10d"], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(o1["z_adapted"], o2["z_adapted"], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
