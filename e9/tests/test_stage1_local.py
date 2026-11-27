"""Unit tests for E9 Stage 1 Local Behavior Encoder (CLIP)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from e5.actions import ActionNormalizer
from e7.model import E7Policy
from e9 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    EMBED_DIM,
    EXEC_HORIZON,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_CAMERAS,
    Z_SELF_COND_DIM,
)
from e9.data import LocalClipFeatureBank
from e9.stage1_local_model import (
    E9Stage1LocalPolicy,
    LocalBehaviorEncoder,
    load_frozen_e9_stage1_teacher,
)


class TestE9Stage1Local(unittest.TestCase):
    def test_local_encoder_padding_invariance_and_dual_camera_temporal_sensitivity(self):
        """Verify:
        1. Output shape is (B, 4, 768).
        2. Tail-window padding mask makes output & gradients strictly invariant to padded frames,
           including length-1 tail window (t = T - 1).
        3. Both cameras and temporal ordering materially affect `b_t`.
        """
        torch.manual_seed(0)
        enc = LocalBehaviorEncoder(
            embed_dim=EMBED_DIM,
            num_layers=2,
            dropout=0.0,
        )
        enc.eval()

        B, L = 3, LOCAL_WINDOW_LEN
        clip_a = torch.randn(B, L, NUM_CAMERAS, EMBED_DIM)
        pad = torch.zeros(B, L, dtype=torch.bool)
        pad[0, 10:] = True  # 10 valid frames, 6 padded
        pad[1, 1:] = True   # extreme length-1 tail window (1 valid frame, 15 padded)
        # sample 2: full 16 valid frames

        # Perturb ONLY padded frames with huge garbage values
        clip_b = clip_a.clone()
        clip_b[0, 10:] = 1e4 * torch.randn_like(clip_b[0, 10:])
        clip_b[1, 1:] = -1e4 * torch.randn_like(clip_b[1, 1:])

        out_a = enc(clip_a, clip_pad_mask=pad)
        out_b = enc(clip_b, clip_pad_mask=pad)
        self.assertEqual(tuple(out_a.shape), (B, NUM_BEHAVIOR_TOKENS, EMBED_DIM))
        self.assertTrue(torch.isfinite(out_a).all())
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)

        # Verify exact zero gradient on padded frames (use random projection since
        # out_norm has weight=1 at init so unweighted .sum() over D is identically ~0)
        clip_grad = clip_a.clone().requires_grad_(True)
        out_g = enc(clip_grad, clip_pad_mask=pad)
        proj = torch.randn_like(out_g)
        (out_g * proj).sum().backward()
        self.assertIsNotNone(clip_grad.grad)
        self.assertEqual(clip_grad.grad[0, 10:].abs().max().item(), 0.0)
        self.assertEqual(clip_grad.grad[1, 1:].abs().max().item(), 0.0)
        self.assertGreater(clip_grad.grad[0, :10].abs().max().item(), 1e-4)
        self.assertGreater(clip_grad.grad[1, :1].abs().max().item(), 1e-4)

        # Dual-camera sensitivity: perturbing camera 0 or camera 1 changes b_t
        clip_cam0 = clip_a.clone()
        clip_cam0[:, 0, 0] += torch.randn_like(clip_cam0[:, 0, 0]) * 5.0
        clip_cam1 = clip_a.clone()
        clip_cam1[:, 0, 1] += torch.randn_like(clip_cam1[:, 0, 1]) * 5.0
        with torch.no_grad():
            out_c0 = enc(clip_cam0, clip_pad_mask=pad)
            out_c1 = enc(clip_cam1, clip_pad_mask=pad)
        self.assertGreater((out_c0 - out_a).abs().max().item(), 1e-2)
        self.assertGreater((out_c1 - out_a).abs().max().item(), 1e-2)

        # Temporal ordering sensitivity: reversing frame order changes b_t
        clip_rev = torch.flip(clip_a[2:3], dims=[1])
        with torch.no_grad():
            out_orig = enc(clip_a[2:3], clip_pad_mask=None)
            out_rev = enc(clip_rev, clip_pad_mask=None)
        self.assertGreater((out_orig - out_rev).abs().max().item(), 1e-2)

        # Reject all-padded clip
        all_pad = torch.ones(1, L, dtype=torch.bool)
        with self.assertRaises(ValueError):
            enc(clip_a[:1], clip_pad_mask=all_pad)

    def test_policy_unet_loss_grad_reload_and_e7_direct_compatibility(self):
        """Verify UNet conditioning shape, pure diffusion loss, gradient flow,
        optimizer parameter grouping, frozen reload, and full E7-Direct compatibility."""
        torch.manual_seed(1)
        policy = E9Stage1LocalPolicy(encoder_layers=2)
        self.assertEqual(policy.global_cond_dim, Z_SELF_COND_DIM)
        self.assertEqual(policy.global_cond_dim, 3072)

        # UNet cond_encoder matches E6 Stage 1 identically: 128 + 3072 = 3200
        w0 = policy.action_decoder.down_modules[0][0].cond_encoder[1].weight
        self.assertEqual(tuple(w0.shape), (512, 128 + 3072))

        # Optimizer grouping covers 100% of trainable params without overlap
        groups = policy.get_optim_groups(lr=1e-4, weight_decay=1e-6)
        self.assertEqual(len(groups), 2)
        grouped_n = sum(len(g["params"]) for g in groups)
        trainable_n = sum(1 for p in policy.parameters() if p.requires_grad)
        self.assertEqual(grouped_n, trainable_n)

        B = 4
        clip = torch.randn(B, LOCAL_WINDOW_LEN, NUM_CAMERAS, EMBED_DIM)
        pad = torch.zeros(B, LOCAL_WINDOW_LEN, dtype=torch.bool)
        pad[0, 12:] = True
        pad[1, 1:] = True
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        policy.train()
        out = policy(clip, pad, actions)
        self.assertSetEqual(set(out.keys()), {"loss", "diff_loss", "b_rms"})
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

        for name, mod in (
            ("local_encoder", policy.local_encoder),
            ("action_decoder", policy.action_decoder),
        ):
            grads = [p.grad for p in mod.parameters() if p.requires_grad]
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in grads),
                f"Missing or non-finite grad in {name}",
            )
            self.assertGreater(sum(g.float().norm().item() for g in grads), 0.0)

        # Prediction + frozen Teacher reload roundtrip
        policy.eval()
        gen = torch.Generator().manual_seed(7)
        pred = policy.predict_action(clip, pad, generator=gen)
        self.assertEqual(tuple(pred["action_10d"].shape), (B, ACTION_HORIZON, ACTION_DIM_10D))
        self.assertEqual(tuple(pred["action_7d"].shape), (B, ACTION_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["exec_action_7d"].shape), (B, EXEC_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["b_t"].shape), (B, NUM_BEHAVIOR_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(pred["z_self"].shape), (B, NUM_BEHAVIOR_TOKENS, EMBED_DIM))

        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best.pt"
            torch.save(
                {
                    "stage": "e9_stage1_local_behavior_encoder",
                    "epoch": 1,
                    "model": policy.state_dict(),
                    "config": {
                        "embed_dim": EMBED_DIM,
                        "window_len": LOCAL_WINDOW_LEN,
                        "num_behavior_tokens": NUM_BEHAVIOR_TOKENS,
                        "encoder_layers": 2,
                    },
                },
                ckpt,
            )
            teacher = load_frozen_e9_stage1_teacher(ckpt, device="cpu")
            self.assertEqual(
                sum(p.numel() for p in teacher.parameters() if p.requires_grad), 0
            )
            with torch.inference_mode():
                b_reloaded = teacher.encode_local_behavior(clip, pad)
            torch.testing.assert_close(b_reloaded, pred["b_t"], atol=1e-5, rtol=1e-5)

            # E7-Direct compatibility: Behavior Memory + SimpleBehaviorReader + Direct Transfer + UNet
            T_demo = 37  # 3 stride-16 anchors: [0, 16, 32] with tail window len=5
            demo_frames = torch.randn(T_demo, NUM_CAMERAS, EMBED_DIM)
            h_mem, b_mem, anchors = teacher.extract_behavior_memory_from_demo_features(
                demo_frames, stride=16
            )
            self.assertEqual(anchors.tolist(), [0, 16, 32])
            self.assertEqual(tuple(h_mem.shape), (3, 4, EMBED_DIM))
            self.assertEqual(tuple(b_mem.shape), (3, 4, EMBED_DIM))

            e7 = E7Policy(teacher=teacher, prediction_mode="direct")
            e7.eval()
            q_rgb = torch.randn(2, 2, 2, EMBED_DIM)
            e7_out = e7.predict_action_from_behavior_memory(
                h_mem.unsqueeze(0).expand(2, -1, -1, -1),
                b_mem.unsqueeze(0).expand(2, -1, -1, -1),
                mem_pad_mask=None,
                query_rgb_feats=q_rgb,
                apply_delta=True,
                delta_scale=1.0,
            )
            self.assertEqual(
                tuple(e7_out["action_10d"].shape), (2, ACTION_HORIZON, ACTION_DIM_10D)
            )
            self.assertEqual(tuple(e7_out["z_adapted"].shape), (2, 4, EMBED_DIM))
            self.assertTrue(torch.isfinite(e7_out["action_10d"]).all())

    def test_local_clip_feature_bank_tail_mask_and_repeat_last(self):
        """Verify synthetic LocalClipFeatureBank constructs exact tail padding masks,
        zeros padded frames, and preserves E6 repeat-last 10D actions."""
        with TemporaryDirectory() as tmp:
            e5_dir = Path(tmp) / "e5"
            dense_dir = Path(tmp) / "dense"
            e5_dir.mkdir()
            dense_dir.mkdir()

            # 2 train episodes (lengths 25, 20), 2 val episodes from different tasks (lengths 22, 18)
            ep_lens = np.array([25, 20, 22, 18], dtype=np.int32)
            episodes = [
                {"id": "t0/demo_0", "task": "task_a", "fold": "train", "frames": 25},
                {"id": "t1/demo_0", "task": "task_b", "fold": "train", "frames": 20},
                {"id": "t0/demo_1", "task": "task_a", "fold": "val", "frames": 22},
                {"id": "t1/demo_1", "task": "task_b", "fold": "val", "frames": 18},
            ]
            offsets = np.zeros(5, dtype=np.int64)
            offsets[1:] = np.cumsum(ep_lens)
            total = int(offsets[-1])

            rng = np.random.default_rng(7)
            feats = rng.standard_normal((total, NUM_CAMERAS, EMBED_DIM)).astype(np.float16)
            actions = rng.standard_normal((total, ACTION_DIM_10D)).astype(np.float32)

            # Query starts including a tail start near end of ep 0 (frame 20 -> 5 valid frames)
            # and extreme tail start (frame 24 -> 1 valid frame)
            q_lists = [
                np.array([0, 1, 19, 20, 23, 24], dtype=np.int32),
                np.array([0, 1, 5, 6], dtype=np.int32),
                np.array([0, 1, 15, 16], dtype=np.int32),
                np.array([0, 1, 10, 11], dtype=np.int32),
            ]
            q_off = np.zeros(5, dtype=np.int64)
            for i, arr in enumerate(q_lists):
                q_off[i + 1] = q_off[i] + len(arr)
            q_nums = np.concatenate(q_lists)

            (e5_dir / "cache_meta.json").write_text(json.dumps({"version": "test"}))
            (e5_dir / "episodes.json").write_text(json.dumps(episodes))
            np.save(e5_dir / "actions_10d.npy", actions)
            np.save(e5_dir / "action_offsets.npy", offsets)
            np.save(e5_dir / "query_ep_offsets.npy", q_off)
            np.save(e5_dir / "query_frame_nums.npy", q_nums)
            norm = ActionNormalizer()
            torch.save({"scale": norm.scale, "offset": norm.offset}, e5_dir / "action_normalizer.pt")

            (dense_dir / "dense_cache_meta.json").write_text(json.dumps({"version": "test_dense"}))
            np.save(dense_dir / "frame_features.npy", feats)
            np.save(dense_dir / "frame_offsets.npy", offsets)
            np.save(dense_dir / "ep_lengths.npy", ep_lens)

            bank = LocalClipFeatureBank(
                e5_cache_dir=e5_dir,
                dense_cache_dir=dense_dir,
                device=torch.device("cpu"),
                resident_on_device=False,
                seed=42,
            )

            # Gather ep 0 at frame 20 (5 valid, 11 padded) and frame 24 (1 valid, 15 padded)
            c_rgb, c_pad, act = bank.gather_batch(np.array([0, 0]), np.array([20, 24]))
            self.assertEqual(tuple(c_rgb.shape), (2, LOCAL_WINDOW_LEN, NUM_CAMERAS, EMBED_DIM))
            self.assertEqual(tuple(c_pad.shape), (2, LOCAL_WINDOW_LEN))
            self.assertEqual(tuple(act.shape), (2, ACTION_HORIZON, ACTION_DIM_10D))

            # Sample 0: frames 20..24 valid (5), 5..15 padded
            self.assertFalse(c_pad[0, :5].any().item())
            self.assertTrue(c_pad[0, 5:].all().item())
            self.assertEqual(c_rgb[0, 5:].abs().max().item(), 0.0)
            # Sample 1: frame 24 valid (1), 1..15 padded
            self.assertFalse(c_pad[1, 0].item())
            self.assertTrue(c_pad[1, 1:].all().item())
            self.assertEqual(c_rgb[1, 1:].abs().max().item(), 0.0)
            # Actions use repeat-last padding
            for s in range(5, ACTION_HORIZON):
                torch.testing.assert_close(act[0, s], act[0, 4])
            for s in range(1, ACTION_HORIZON):
                torch.testing.assert_close(act[1, s], act[1, 0])


if __name__ == "__main__":
    unittest.main()
