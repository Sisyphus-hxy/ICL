"""Unit tests for E6 Stage 2.1 (Joint Cross Reader Q/K & Latent Delta Training)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from e6 import ACTION_DIM_10D, ACTION_HORIZON, EMBED_DIM, NUM_QUERY_TOKENS
from e6.stage1_model import E6Stage1Policy
from e6.stage2_1_model import (
    FIXED_LATENT_SCALE_S_Z,
    E6Stage2_1Policy,
    load_stage2_1_policy,
)
from e6.train_stage2_1 import get_cosine_schedule_with_min_ratio, get_scheduled_lambda_z


class TestE6Stage2_1(unittest.TestCase):
    def _build_dummy_stage2_1(
        self, s_z: float = FIXED_LATENT_SCALE_S_Z, scope: str = "both"
    ) -> E6Stage2_1Policy:
        torch.manual_seed(0)
        teacher = E6Stage1Policy(use_attention_sink=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return E6Stage2_1Policy(teacher, latent_scale_s_z=s_z, reader_qk_scope=scope)

    def test_independent_instances_and_exact_qk_unfreezing(self):
        policy = self._build_dummy_stage2_1(scope="both")
        policy.train(True)

        # Teacher & Cross Reader stay in eval(); ResidualTransformer in train()
        self.assertFalse(policy.teacher.training)
        self.assertFalse(policy.cross_reader.training)
        self.assertTrue(policy.residual_transformer.training)

        # Zero shared tensor storage between Teacher and Cross Reader
        t_ptrs = {p.data_ptr() for p in policy.teacher.parameters()}
        s_ptrs = {p.data_ptr() for p in policy.cross_reader.parameters()}
        self.assertEqual(len(t_ptrs & s_ptrs), 0)

        # Teacher has zero trainable params
        self.assertEqual(
            sum(p.numel() for p in policy.teacher.parameters() if p.requires_grad), 0
        )

        # Cross Reader: ONLY 12 attention modules' q_proj_weight, k_proj_weight, in_proj_bias
        audit = policy.reader_qk_audit
        self.assertEqual(audit["num_target_attention_modules"], 12)
        self.assertEqual(len(audit["trainable_param_names"]), 36)  # 12 * (q_w, k_w, in_proj_bias)
        expected_qk = 12 * (2 * EMBED_DIM * EMBED_DIM + 2 * EMBED_DIM)
        self.assertEqual(audit["trainable_effective_qk_scalars"], expected_qk)

        for name, p in policy.cross_reader.named_parameters():
            if p.requires_grad:
                self.assertTrue(
                    name.endswith(".q_proj_weight")
                    or name.endswith(".k_proj_weight")
                    or name.endswith(".in_proj_bias"),
                    f"Unexpected trainable param: {name}",
                )
            else:
                self.assertFalse(name.endswith(".q_proj_weight"))
                self.assertFalse(name.endswith(".k_proj_weight"))

    def test_zero_init_identity_and_exact_step0_equivalence(self):
        policy = self._build_dummy_stage2_1()
        policy.eval()

        B, P = 4, 10
        demo_mem = torch.randn(B, P, EMBED_DIM)
        pad_mask = torch.zeros(B, P, dtype=torch.bool)
        pad_mask[:, -2:] = True
        query_rgb = torch.randn(B, 2, 2, EMBED_DIM)

        z_t, h_q = policy.read_teacher_latent_from_memory(demo_mem, pad_mask, query_rgb)
        z_s, _ = policy.read_cross_latent_from_memory(demo_mem, pad_mask, query_rgb)
        z_q, delta_z = policy.adapt_latent(z_s, h_q)

        torch.testing.assert_close(z_s, z_t, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(delta_z, torch.zeros_like(delta_z), atol=0.0, rtol=0.0)
        torch.testing.assert_close(z_q, z_s, atol=0.0, rtol=0.0)

    def test_frozen_weights_stay_bit_exact_and_unet_grad_flows_to_both(self):
        policy = self._build_dummy_stage2_1()
        policy.train()

        # Snapshot all frozen weights AND Value slices of in_proj_bias before optimization
        teacher_snap = {k: v.detach().clone() for k, v in policy.teacher.state_dict().items()}
        frozen_reader_snap = {
            k: v.detach().clone()
            for k, v in policy.cross_reader.named_parameters()
            if not v.requires_grad
        }
        v_bias_snap = {
            k: v.detach()[2 * EMBED_DIM :].clone()
            for k, v in policy.cross_reader.named_parameters()
            if v.requires_grad and k.endswith(".in_proj_bias")
        }
        qk_before = {
            k: v.detach().clone()
            for k, v in policy.cross_reader.named_parameters()
            if v.requires_grad and (k.endswith(".q_proj_weight") or k.endswith(".k_proj_weight"))
        }

        opt = torch.optim.AdamW(
            policy.get_optim_groups(reader_lr=3e-4, residual_lr=1e-3, weight_decay=0.01),
            betas=(0.9, 0.95),
        )

        B, P = 4, 8
        demo_mem = torch.randn(B, P, EMBED_DIM)
        demo_mask = torch.zeros(B, P, dtype=torch.bool)
        query_mem = torch.randn(B, P, EMBED_DIM)
        query_mask = torch.zeros(B, P, dtype=torch.bool)
        query_rgb = torch.randn(B, 2, 2, EMBED_DIM)
        z_self, _ = policy.read_teacher_latent_from_memory(query_mem, query_mask, query_rgb)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_self = torch.zeros(B, dtype=torch.bool)

        # Step 1: Pure Diffusion Loss (lambda_z=0, lambda_anchor=0) MUST flow through frozen UNet
        # into BOTH Cross Reader Q/K AND ResidualTransformer out_proj!
        opt.zero_grad(set_to_none=True)
        out1 = policy(
            demo_mem,
            demo_mask,
            query_mem,
            query_mask,
            query_rgb,
            z_self,
            actions,
            is_self,
            lambda_a=1.0,
            lambda_z=0.0,
            lambda_anchor=0.0,
        )
        out1["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )
        for name, p in policy.cross_reader.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing grad for {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad for {name}")
                self.assertGreater(p.grad.float().norm().item(), 0.0, f"Zero grad for {name}")
                if name.endswith(".in_proj_bias"):
                    self.assertEqual(
                        float(p.grad[2 * EMBED_DIM :].abs().max().item()),
                        0.0,
                        f"Non-zero V-bias grad in {name}",
                    )
            else:
                self.assertIsNone(p.grad, f"Frozen param received grad: {name}")
        opt.step()

        # Steps 2-4: Full joint loss (1.0 * L_diff + 0.5 * L_z + 0.1 * L_anchor)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = policy(
                demo_mem,
                demo_mask,
                query_mem,
                query_mask,
                query_rgb,
                z_self,
                actions,
                is_self,
                lambda_a=1.0,
                lambda_z=0.5,
                lambda_anchor=0.1,
            )
            out["loss"].backward()
            opt.step()

        # Verify ALL frozen Teacher params, frozen Cross Reader params, AND V-bias slices are BIT-EXACT
        for k, v in policy.teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Teacher changed: {k}")
        for k, p in policy.cross_reader.named_parameters():
            if not p.requires_grad:
                self.assertTrue(
                    torch.equal(p.detach(), frozen_reader_snap[k]),
                    f"Frozen Cross Reader param changed: {k}",
                )
            elif k.endswith(".in_proj_bias"):
                self.assertTrue(
                    torch.equal(p.detach()[2 * EMBED_DIM :], v_bias_snap[k]),
                    f"V-bias slice changed in {k}",
                )
            else:
                self.assertFalse(
                    torch.equal(p.detach(), qk_before[k]),
                    f"Trainable Q/K weight failed to update: {k}",
                )

    def test_dual_lr_cosine_scheduler_and_checkpoint_roundtrip(self):
        policy = self._build_dummy_stage2_1()
        opt = torch.optim.AdamW(
            policy.get_optim_groups(reader_lr=3e-6, residual_lr=1e-4, weight_decay=0.01),
            betas=(0.9, 0.95),
        )
        sched = get_cosine_schedule_with_min_ratio(
            opt, num_warmup_steps=500, num_training_steps=42912, min_lr_ratio=0.1
        )
        r_lrs, res_lrs = [], []
        for _ in range(42912):
            lrs = sched.get_last_lr()
            r_lrs.append(lrs[0])
            res_lrs.append(lrs[2])
            sched.step()

        self.assertAlmostEqual(r_lrs[0], 0.0, places=11)
        self.assertAlmostEqual(r_lrs[500], 3e-6, places=9)
        self.assertAlmostEqual(r_lrs[-1], 3e-7, places=8)
        self.assertAlmostEqual(res_lrs[0], 0.0, places=10)
        self.assertAlmostEqual(res_lrs[500], 1e-4, places=7)
        self.assertAlmostEqual(res_lrs[-1], 1e-5, places=6)

        # Round-trip checkpoint save/load
        with TemporaryDirectory() as tmp:
            s1_path = Path(tmp) / "s1.pt"
            s21_path = Path(tmp) / "s21.pt"
            torch.save({"model": policy.teacher.state_dict(), "config": {}}, s1_path)
            torch.save(
                {
                    "stage": "e6_stage2_1_joint_reader_delta",
                    "epoch": 1,
                    "latent_scale_s_z": FIXED_LATENT_SCALE_S_Z,
                    "cross_reader": policy.cross_reader.state_dict(),
                    "residual_transformer": policy.residual_transformer.state_dict(),
                    "config": {
                        "stage1_ckpt": str(s1_path),
                        "reader_qk_scope": "both",
                    },
                },
                s21_path,
            )
            reloaded, _ = load_stage2_1_policy(s21_path, device="cpu")
            B, P = 2, 6
            dm = torch.randn(B, P, EMBED_DIM)
            pm = torch.zeros(B, P, dtype=torch.bool)
            q = torch.randn(B, 2, 2, EMBED_DIM)
            z0 = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
            o1 = policy.predict_action_from_memory(dm, pm, q, init_noise=z0)
            o2 = reloaded.predict_action_from_memory(dm, pm, q, init_noise=z0)
            torch.testing.assert_close(o1["action_10d"], o2["action_10d"], atol=1e-6, rtol=1e-6)

    def test_lambda_z_schedule_and_action_only_loss(self):
        # Constant schedule (Stage 2.1 baseline & Action-only)
        self.assertAlmostEqual(
            get_scheduled_lambda_z(0, total_steps=42912, schedule="constant", lambda_z=0.0),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            get_scheduled_lambda_z(42911, total_steps=42912, schedule="constant", lambda_z=0.5),
            0.5,
            places=12,
        )

        # Hold-linear schedule: first 20% hold 0.5, remaining 80% linearly decay to 0.1
        total = 42912
        hold = int(round(0.2 * total))  # 8582
        self.assertEqual(hold, 8582)
        self.assertAlmostEqual(
            get_scheduled_lambda_z(0, total_steps=total, schedule="hold_linear"),
            0.5,
            places=12,
        )
        self.assertAlmostEqual(
            get_scheduled_lambda_z(hold - 1, total_steps=total, schedule="hold_linear"),
            0.5,
            places=12,
        )
        self.assertAlmostEqual(
            get_scheduled_lambda_z(hold, total_steps=total, schedule="hold_linear"),
            0.5,
            places=12,
        )
        mid = hold + (total - 1 - hold) // 2
        self.assertAlmostEqual(
            get_scheduled_lambda_z(mid, total_steps=total, schedule="hold_linear"),
            0.3,
            places=4,
        )
        self.assertAlmostEqual(
            get_scheduled_lambda_z(total - 1, total_steps=total, schedule="hold_linear"),
            0.1,
            places=12,
        )

        # Action-only forward: L = L_diff + 0.1 * L_anchor (lambda_z = 0.0)
        policy = self._build_dummy_stage2_1()
        policy.train()
        B, P = 4, 8
        dm = torch.randn(B, P, EMBED_DIM)
        pm = torch.zeros(B, P, dtype=torch.bool)
        qm = torch.randn(B, P, EMBED_DIM)
        qpm = torch.zeros(B, P, dtype=torch.bool)
        q = torch.randn(B, 2, 2, EMBED_DIM)
        zs, _ = policy.read_teacher_latent_from_memory(qm, qpm, q)
        act = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_s = torch.zeros(B, dtype=torch.bool)
        out = policy(
            dm, pm, qm, qpm, q, zs, act, is_s,
            lambda_a=1.0, lambda_z=0.0, lambda_anchor=0.1,
        )
        expected = out["diff_loss"] + 0.1 * out["anchor_loss_norm"]
        torch.testing.assert_close(out["loss"].detach(), expected, atol=1e-6, rtol=1e-6)
        self.assertGreater(float(out["latent_loss_norm"].item()), 0.0)
        out["loss"].backward()
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )

    def test_pure_latent_alignment_skips_unet_and_updates_both_modules(self):
        policy = self._build_dummy_stage2_1()
        policy.train()

        # Guard UNet / diffusion_loss_from_z_self to guarantee it is NEVER called when lambda_a=0
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("UNet diffusion_loss_from_z_self must NOT be called when lambda_a=0.0")

        policy.teacher.diffusion_loss_from_z_self = _fail_if_called
        policy.teacher.action_decoder.forward = _fail_if_called

        B, P = 4, 8
        dm = torch.randn(B, P, EMBED_DIM)
        pm = torch.zeros(B, P, dtype=torch.bool)
        qm = torch.randn(B, P, EMBED_DIM)
        qpm = torch.zeros(B, P, dtype=torch.bool)
        q = torch.randn(B, 2, 2, EMBED_DIM)
        zs, _ = policy.read_teacher_latent_from_memory(qm, qpm, q)
        act = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_s = torch.zeros(B, dtype=torch.bool)

        out = policy(
            dm, pm, qm, qpm, q, zs, act, is_s,
            lambda_a=0.0, lambda_z=0.5, lambda_anchor=0.1,
        )
        self.assertEqual(float(out["diff_loss"].item()), 0.0)
        expected = 0.5 * out["latent_loss_norm"] + 0.1 * out["anchor_loss_norm"]
        torch.testing.assert_close(out["loss"].detach(), expected, atol=1e-6, rtol=1e-6)
        out["loss"].backward()

        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )
        for name, p in policy.cross_reader.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing grad for {name}")
                self.assertGreater(p.grad.float().norm().item(), 0.0, f"Zero grad for {name}")
                if name.endswith(".in_proj_bias"):
                    self.assertEqual(float(p.grad[2 * EMBED_DIM :].abs().max().item()), 0.0)
            else:
                self.assertIsNone(p.grad, f"Frozen param received grad: {name}")

    def test_stage2_3_id_and_refine_losses_stop_grad_and_multistep_inference(self):
        policy = self._build_dummy_stage2_1()
        policy.train()

        # Take 1 small optimization step so out_proj is non-zero before testing L_id / L_refine
        opt = torch.optim.AdamW(
            policy.get_optim_groups(reader_lr=3e-4, residual_lr=1e-3, weight_decay=0.01),
            betas=(0.9, 0.95),
        )
        B, P = 4, 8
        dm = torch.randn(B, P, EMBED_DIM)
        pm = torch.zeros(B, P, dtype=torch.bool)
        qm = torch.randn(B, P, EMBED_DIM)
        qpm = torch.zeros(B, P, dtype=torch.bool)
        q = torch.randn(B, 2, 2, EMBED_DIM)
        zs, _ = policy.read_teacher_latent_from_memory(qm, qpm, q)
        act = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_s = torch.zeros(B, dtype=torch.bool)

        opt.zero_grad(set_to_none=True)
        out0 = policy(
            dm, pm, qm, qpm, q, zs, act, is_s,
            lambda_a=1.0, lambda_z=0.5, lambda_anchor=0.1,
            lambda_id=1.0, lambda_refine=0.2,
        )
        out0["loss"].backward()
        opt.step()

        # 1. Pure L_id: gradients flow ONLY into residual_transformer (zero on cross_reader & teacher)
        policy.zero_grad(set_to_none=True)
        out_id = policy(
            dm, pm, qm, qpm, q, zs, act, is_s,
            lambda_a=0.0, lambda_z=0.0, lambda_anchor=0.0,
            lambda_id=1.0, lambda_refine=0.0,
        )
        self.assertGreater(float(out_id["id_loss_norm"].item()), 0.0)
        self.assertEqual(float(out_id["refine_loss_norm"].item()), 0.0)
        torch.testing.assert_close(out_id["loss"].detach(), out_id["id_loss_norm"], atol=1e-6, rtol=1e-6)
        out_id["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.cross_reader.parameters()))
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )

        # 2. Pure L_refine: z1 is detached so cross_reader gets ZERO grad, while 2nd G forward has grad!
        policy.zero_grad(set_to_none=True)
        out_ref = policy(
            dm, pm, qm, qpm, q, zs, act, is_s,
            lambda_a=0.0, lambda_z=0.0, lambda_anchor=0.0,
            lambda_id=0.0, lambda_refine=0.2,
        )
        self.assertGreater(float(out_ref["refine_loss_norm"].item()), 0.0)
        self.assertEqual(float(out_ref["id_loss_norm"].item()), 0.0)
        torch.testing.assert_close(
            out_ref["loss"].detach(), 0.2 * out_ref["refine_loss_norm"], atol=1e-6, rtol=1e-6
        )
        out_ref["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.cross_reader.parameters()))
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )

        # 3. Exact combined Stage 2.3 loss value + frozen invariants
        teacher_snap = {k: v.detach().clone() for k, v in policy.teacher.state_dict().items()}
        v_bias_snap = {
            k: v.detach()[2 * EMBED_DIM :].clone()
            for k, v in policy.cross_reader.named_parameters()
            if v.requires_grad and k.endswith(".in_proj_bias")
        }
        opt.zero_grad(set_to_none=True)
        out_f = policy(
            dm, pm, qm, qpm, q, zs, act, is_s,
            lambda_a=1.0, lambda_z=0.5, lambda_anchor=0.1,
            lambda_id=1.0, lambda_refine=0.2,
        )
        expected_f = (
            out_f["diff_loss"]
            + 0.5 * out_f["latent_loss_norm"]
            + 0.1 * out_f["anchor_loss_norm"]
            + 1.0 * out_f["id_loss_norm"]
            + 0.2 * out_f["refine_loss_norm"]
        )
        torch.testing.assert_close(out_f["loss"].detach(), expected_f, atol=1e-5, rtol=1e-5)
        out_f["loss"].backward()
        opt.step()

        for k, v in policy.teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Teacher changed: {k}")
        for k, p in policy.cross_reader.named_parameters():
            if p.requires_grad and k.endswith(".in_proj_bias"):
                self.assertTrue(
                    torch.equal(p.detach()[2 * EMBED_DIM :], v_bias_snap[k]),
                    f"V-bias slice changed in {k}",
                )

        # 4. Multi-step inference equivalence (0-step, 1-step, 2-step)
        policy.eval()
        z0 = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        z_c, h_q = policy.read_cross_latent_from_memory(dm, pm, q)
        z1_manual, _ = policy.adapt_latent(z_c, h_q)
        z2_manual, _ = policy.adapt_latent(z1_manual, h_q)

        o0 = policy.predict_action_from_memory(dm, pm, q, apply_delta=False, init_noise=z0)
        o1 = policy.predict_action_from_memory(dm, pm, q, apply_delta=True, init_noise=z0)
        o2 = policy.predict_action_from_memory(dm, pm, q, num_delta_steps=2, init_noise=z0)
        o_s0 = policy.predict_action_from_memory(dm, pm, q, apply_delta=True, delta_scale=0.0, init_noise=z0)
        o_s15 = policy.predict_action_from_memory(dm, pm, q, apply_delta=True, delta_scale=1.5, init_noise=z0)

        torch.testing.assert_close(o0["z_adapted"], z_c, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(o_s0["z_adapted"], z_c, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(o1["z_adapted"], z1_manual, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(o2["z_adapted"], z2_manual, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            o_s15["z_adapted"], z_c + 1.5 * (z1_manual - z_c), atol=1e-6, rtol=1e-6
        )
        self.assertEqual(o0["num_delta_steps"], 0)
        self.assertEqual(o1["num_delta_steps"], 1)
        self.assertEqual(o2["num_delta_steps"], 2)
        self.assertEqual(o_s15["delta_scale"], 1.5)


if __name__ == "__main__":
    unittest.main()
