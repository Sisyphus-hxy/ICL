"""Unit tests for Experiment A (Reader Temperature Sweep) and Experiment B (Direct + Exact Self Identity)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from e7.model import SimpleBehaviorReader, compute_attention_stats
from e9 import ACTION_DIM_10D, ACTION_HORIZON, EMBED_DIM, NUM_BEHAVIOR_TOKENS, NUM_CAMERAS
from e9.direct_identity_model import build_e9_direct_identity_policy
from e9.reader_temperature import TemperatureScaledReader, set_policy_reader_temperature
from e9.stage1_local_model import E9Stage1LocalPolicy
from e9.stage2_direct_model import load_e9_stage2_direct_policy
from e9.train_direct_identity import measure_gradient_contributions


class TestExpAAndExpB(unittest.TestCase):
    def test_exp_a_temperature_scaling_exactness_and_monotonicity(self):
        torch.manual_seed(7)
        base = SimpleBehaviorReader()
        base.eval()

        B, M = 4, 10
        h_q = torch.randn(B, 4, EMBED_DIM)
        h_mem = torch.randn(B, M, 4, EMBED_DIM)
        b_mem = torch.randn(B, M, 4, EMBED_DIM)
        pad = torch.zeros(B, M, dtype=torch.bool)
        pad[0, 7:] = True

        # tau = 1.0 matches original SimpleBehaviorReader bit-for-bit
        z_orig, w_orig = base(h_q, h_mem, b_mem, mem_pad_mask=pad)
        wrapped = TemperatureScaledReader(base, temperature=1.0)
        z_t1, w_t1 = wrapped(h_q, h_mem, b_mem, mem_pad_mask=pad)
        torch.testing.assert_close(z_t1, z_orig, atol=0.0, rtol=0.0)
        torch.testing.assert_close(w_t1, w_orig, atol=0.0, rtol=0.0)

        # Manual formula check for tau = 0.25 and tau = 1.5, plus argmax invariance & entropy monotonicity
        logits = wrapped.compute_logits(h_q, h_mem, mem_pad_mask=pad)
        argmax_ref = w_orig.argmax(dim=-1)
        prev_ent = -1.0
        for tau in (0.25, 0.5, 0.75, 1.0, 1.5):
            wrapped.temperature = tau
            z_t, w_t = wrapped(h_q, h_mem, b_mem, mem_pad_mask=pad)
            w_manual = torch.softmax(logits / tau, dim=-1)
            z_manual = torch.einsum("bm,bmkd->bkd", w_manual, b_mem)
            torch.testing.assert_close(w_t, w_manual, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(z_t, z_manual, atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(w_t.argmax(dim=-1), argmax_ref))
            self.assertTrue(torch.all(w_t[0, 7:] == 0.0))
            ent = float(compute_attention_stats(w_t, pad)["entropy"].mean().item())
            self.assertGreater(ent, prev_ent)
            prev_ent = ent

    def test_exp_b_identity_loss_isolation_grad_decomposition_and_reload(self):
        torch.manual_seed(11)
        s1 = E9Stage1LocalPolicy(encoder_layers=2)
        for p in s1.parameters():
            p.requires_grad_(False)
        s1.eval()

        s_z = 1.006133
        bias = torch.randn(1, NUM_BEHAVIOR_TOKENS, EMBED_DIM) * 0.2
        policy = build_e9_direct_identity_policy(
            s1,
            latent_scale_s_z=s_z,
            direct_latent_bias=bias,
            direct_out_proj_init_std=0.002,
            residual_dropout=0.0,  # deterministic comparison of exact formula
        )

        # Parameter counts identical to E9 Stage 2 Direct
        ps = policy.parameter_summary()
        self.assertEqual(ps["simple_reader_trainable"], 4_732_416)
        self.assertEqual(ps["residual_transformer_trainable"], 28_956_672)
        self.assertEqual(ps["total_trainable"], 33_689_088)
        self.assertEqual(ps["stage1_teacher_trainable"], 0)

        B, M = 3, 6
        h_mem = torch.randn(B, M, 4, EMBED_DIM)
        b_mem = torch.randn(B, M, 4, EMBED_DIM)
        pad = torch.zeros(B, M, dtype=torch.bool)
        pad[1, 4:] = True
        q_rgb = torch.randn(B, 2, NUM_CAMERAS, EMBED_DIM)
        z_self = torch.randn(B, 4, EMBED_DIM)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_self = torch.tensor([False, False, True])

        # 1. Pure L_id must NOT touch Reader or Teacher/UNet
        policy.train()
        out_id = policy(
            h_mem, b_mem, pad, q_rgb, z_self, actions, is_self,
            lambda_a=0.0, lambda_z=0.0, lambda_id=0.1,
        )
        out_id["loss"].backward()
        self.assertGreater(policy.residual_transformer.out_proj.weight.grad.norm().item(), 0.0)
        self.assertFalse(any(p.grad is not None for p in policy.reader.parameters()))
        self.assertFalse(any(p.grad is not None for p in policy.teacher.parameters()))
        policy.zero_grad(set_to_none=True)

        # 2. Combined loss formula & gradient decomposition exactness
        out_all = policy(
            h_mem, b_mem, pad, q_rgb, z_self, actions, is_self,
            lambda_a=1.0, lambda_z=0.5, lambda_id=0.1, return_loss_terms=True,
        )
        torch.testing.assert_close(
            out_all["loss"],
            out_all["diff_loss"] + 0.5 * out_all["latent_loss_norm"] + 0.1 * out_all["id_loss_norm"],
            atol=1e-5,
            rtol=1e-5,
        )
        gp = measure_gradient_contributions(policy, out_all, use_ddp=False)
        out_all["loss"].backward()
        transfer_params = [p for p in policy.residual_transformer.parameters() if p.requires_grad]
        actual_total_gn = torch.cat([p.grad.detach().float().reshape(-1) for p in transfer_params]).norm().item()
        self.assertAlmostEqual(gp["transfer_gn_total"], actual_total_gn, places=4)
        self.assertGreater(gp["transfer_gn_cross"], 0.0)
        self.assertGreater(gp["transfer_gn_id"], 0.0)
        self.assertGreater(gp["reader_gn_cross"], 0.0)
        policy.zero_grad(set_to_none=True)

        # 3. Checkpoint reload via standard `load_e9_stage2_direct_policy` + temperature helper
        policy.eval()
        z0 = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        pred_before = policy.predict_action_from_behavior_memory(
            h_mem, b_mem, pad, q_rgb, apply_delta=True, init_noise=z0
        )
        with TemporaryDirectory() as tmp:
            s1_path = Path(tmp) / "stage1.pt"
            s2_path = Path(tmp) / "best.pt"
            torch.save(
                {"stage": "e9_stage1_local_behavior_encoder", "model": s1.state_dict(), "config": {"encoder_layers": 2}},
                s1_path,
            )
            torch.save(
                {
                    "stage": "e9_exp_b_direct_exact_self_identity",
                    "prediction_mode": "direct",
                    "epoch": 1,
                    "latent_scale_s_z": s_z,
                    "direct_latent_bias": policy.direct_latent_bias.detach().cpu(),
                    "reader": policy.reader.state_dict(),
                    "residual_transformer": policy.residual_transformer.state_dict(),
                    "config": {
                        "stage1_ckpt": str(s1_path),
                        "latent_scale_s_z": s_z,
                        "reader_attn_dim": EMBED_DIM,
                        "residual_layers": 3,
                        "residual_dropout": 0.0,
                        "direct_out_proj_init_std": 0.002,
                    },
                },
                s2_path,
            )
            reloaded, _ = load_e9_stage2_direct_policy(s2_path, device="cpu")
            pred_after = reloaded.predict_action_from_behavior_memory(
                h_mem, b_mem, pad, q_rgb, apply_delta=True, init_noise=z0
            )
            torch.testing.assert_close(pred_before["z_adapted"], pred_after["z_adapted"], atol=1e-5, rtol=1e-5)
            set_policy_reader_temperature(reloaded, 0.5)
            pred_t05 = reloaded.predict_action_from_behavior_memory(
                h_mem, b_mem, pad, q_rgb, apply_delta=True, init_noise=z0
            )
            self.assertFalse(torch.allclose(pred_after["attn_weights"], pred_t05["attn_weights"]))


if __name__ == "__main__":
    unittest.main()
