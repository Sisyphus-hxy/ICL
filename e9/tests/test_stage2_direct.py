"""Unit tests for E9 Stage 2 — Local Behavior Space + Direct Transfer (CLIP)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from e9 import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    EMBED_DIM,
    NUM_BEHAVIOR_TOKENS,
    NUM_CAMERAS,
)
from e9.stage1_local_model import E9Stage1LocalPolicy
from e9.stage2_direct_model import (
    build_e9_stage2_direct_policy,
    load_e9_stage2_direct_policy,
)


class TestE9Stage2Direct(unittest.TestCase):
    def test_direct_architecture_no_residual_skip_and_frozen_teacher(self):
        """Verify E9 Stage 2 strictly reuses E7-Direct architecture:
        - Exact parameter counts (SimpleBehaviorReader=4,732,416, Transfer=28,956,672, total=33,689,088)
        - prediction_mode == 'direct' (NO residual skip from z_c, NO Source State Conditioning)
        - Frozen Stage 1 Local teacher receives ZERO gradients
        - Pure L_diff flows through frozen UNet into BOTH Reader Q/K and Transfer out_proj
        - Checkpoint save/reload roundtrip is exact
        """
        torch.manual_seed(0)
        s1 = E9Stage1LocalPolicy(encoder_layers=2)
        for p in s1.parameters():
            p.requires_grad_(False)
        s1.eval()

        s_z = 1.00617
        bias = torch.randn(1, NUM_BEHAVIOR_TOKENS, EMBED_DIM) * 0.3
        policy = build_e9_stage2_direct_policy(
            s1,
            latent_scale_s_z=s_z,
            direct_latent_bias=bias,
            direct_out_proj_init_std=0.002,
        )

        self.assertEqual(policy.prediction_mode, "direct")
        ps = policy.parameter_summary()
        self.assertEqual(ps["simple_reader_trainable"], 4_732_416)
        self.assertEqual(ps["residual_transformer_trainable"], 28_956_672)
        self.assertEqual(ps["total_trainable"], 33_689_088)
        self.assertEqual(ps["stage1_teacher_trainable"], 0)

        # Verify NO residual skip: with zeroed out_proj, z_q == direct_latent_bias regardless of z_c
        with torch.no_grad():
            orig_w = policy.residual_transformer.out_proj.weight.clone()
            policy.residual_transformer.out_proj.weight.zero_()
            z_c1 = torch.randn(2, 4, EMBED_DIM) * 5.0
            h_q = torch.randn(2, 4, EMBED_DIM)
            z_q0, _ = policy.adapt_latent(z_c1, h_q)
            torch.testing.assert_close(z_q0, bias.expand(2, -1, -1), atol=1e-6, rtol=1e-6)
            policy.residual_transformer.out_proj.weight.copy_(orig_w)

        # Verify pure L_diff gradient flows through frozen UNet into Reader Q/K and Transfer out_proj
        B, M = 3, 6
        h_mem = torch.randn(B, M, 4, EMBED_DIM)
        b_mem = torch.randn(B, M, 4, EMBED_DIM)
        mem_pad = torch.zeros(B, M, dtype=torch.bool)
        mem_pad[0, 4:] = True
        q_rgb = torch.randn(B, 2, NUM_CAMERAS, EMBED_DIM)
        z_self = torch.randn(B, 4, EMBED_DIM)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_self = torch.tensor([False, False, True])

        policy.train()
        out = policy(
            h_mem, b_mem, mem_pad, q_rgb, z_self, actions, is_self, lambda_a=1.0, lambda_z=0.0
        )
        out["loss"].backward()

        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.norm().item(), 0.0
        )
        self.assertGreater(policy.reader.q_proj.weight.grad.norm().item(), 0.0)
        self.assertGreater(policy.reader.k_proj.weight.grad.norm().item(), 0.0)
        self.assertFalse(any(p.grad is not None for p in policy.teacher.parameters()))
        policy.zero_grad(set_to_none=True)

        # Verify combined L = L_diff + 0.5 * L_z loss formula
        out_both = policy(
            h_mem, b_mem, mem_pad, q_rgb, z_self, actions, is_self, lambda_a=1.0, lambda_z=0.5
        )
        torch.testing.assert_close(
            out_both["loss"],
            out_both["diff_loss"] + 0.5 * out_both["latent_loss_norm"],
            atol=1e-5,
            rtol=1e-5,
        )

        # Verify checkpoint save & reload roundtrip
        policy.eval()
        gen = torch.Generator().manual_seed(99)
        z0 = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D, generator=gen)
        pred_before = policy.predict_action_from_behavior_memory(
            h_mem, b_mem, mem_pad, q_rgb, apply_delta=True, init_noise=z0
        )

        with TemporaryDirectory() as tmp:
            s1_path = Path(tmp) / "stage1.pt"
            s2_path = Path(tmp) / "stage2.pt"
            torch.save(
                {
                    "stage": "e9_stage1_local_behavior_encoder",
                    "model": s1.state_dict(),
                    "config": {"encoder_layers": 2},
                },
                s1_path,
            )
            torch.save(
                {
                    "stage": "e9_stage2_local_direct_transfer",
                    "prediction_mode": "direct",
                    "epoch": 3,
                    "latent_scale_s_z": s_z,
                    "direct_latent_bias": policy.direct_latent_bias.detach().cpu(),
                    "reader": policy.reader.state_dict(),
                    "residual_transformer": policy.residual_transformer.state_dict(),
                    "config": {
                        "stage1_ckpt": str(s1_path),
                        "latent_scale_s_z": s_z,
                        "reader_attn_dim": EMBED_DIM,
                        "residual_layers": 3,
                        "residual_dropout": 0.1,
                        "direct_out_proj_init_std": 0.002,
                    },
                },
                s2_path,
            )
            reloaded, _ = load_e9_stage2_direct_policy(s2_path, device="cpu")
            pred_after = reloaded.predict_action_from_behavior_memory(
                h_mem, b_mem, mem_pad, q_rgb, apply_delta=True, init_noise=z0
            )
            torch.testing.assert_close(
                pred_before["z_adapted"], pred_after["z_adapted"], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                pred_before["action_10d"], pred_after["action_10d"], atol=1e-5, rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
