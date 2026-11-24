"""Unit tests for E6 Stage 1 (Self-conditioned Behavior Latent Learning, CLIP)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from e6 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    DEFAULT_INIT_CKPT,
    E5_CACHE_DIR,
    EMBED_DIM,
    EXEC_HORIZON,
    NUM_QUERY_TOKENS,
    Z_SELF_COND_DIM,
)
from e6.data import SelfPairFeatureBank
from e6.stage1_model import (
    BehaviorReferenceReader,
    E6Stage1Policy,
    QueryReadLayer,
    load_frozen_stage1_teacher,
)


class TestE6Stage1(unittest.TestCase):
    def test_no_query_residual_bypass_strict_bottleneck(self):
        """Prove mathematically that `h_q` cannot bypass Demo Memory via any residual/skip path:
        when Demo Memory (M_d + prompt_pos_emb) is constant across time, `z_self` is strictly
        invariant to `h_q` and `d(z_self)/d(h_q) == 0`."""
        torch.manual_seed(0)
        reader = BehaviorReferenceReader(
            embed_dim=EMBED_DIM,
            num_layers=6,
            use_attention_sink=False,
        )
        reader.eval()
        self.assertIsInstance(reader.prompt_with_obs_decoder.layers[0], QueryReadLayer)
        self.assertIsNone(reader.attention_sink_tokens)

        B, P = 2, 10
        # Zero out prompt_pos_emb so constant M_d produces identical K/V across all P tokens.
        with torch.no_grad():
            reader.prompt_pos_emb.zero_()
        const_token = torch.randn(B, 1, EMBED_DIM)
        M_d_const = const_token.expand(B, P, EMBED_DIM).clone()

        h_q_1 = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM, requires_grad=True)
        h_q_2 = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)

        z1, _ = reader(M_d_const, h_q_1)
        z2, _ = reader(M_d_const, h_q_2)
        torch.testing.assert_close(z1, z2, atol=1e-5, rtol=1e-5)

        z1.sum().backward()
        self.assertIsNotNone(h_q_1.grad)
        self.assertLess(h_q_1.grad.abs().max().item(), 1e-5)

        # Contrast: with non-constant temporal M_d, different h_q MUST produce different z_self
        # and receive non-zero gradient so h_q can learn where to attend in Demo Memory!
        M_d_varying = torch.randn(B, P, EMBED_DIM)
        h_q_3 = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM, requires_grad=True)
        h_q_4 = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        z3, attn3 = reader(M_d_varying, h_q_3, need_weights=True)
        z4, _ = reader(M_d_varying, h_q_4)
        self.assertEqual(tuple(z3.shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(attn3.shape), (6, B, 8, NUM_QUERY_TOKENS, P))
        self.assertGreater((z3 - z4).abs().max().item(), 1e-2)
        z3.sum().backward()
        self.assertGreater(h_q_3.grad.abs().max().item(), 1e-4)

    def test_policy_architecture_unet_cond_and_gradient_flow(self):
        torch.manual_seed(1)
        policy = E6Stage1Policy(use_attention_sink=False)
        self.assertEqual(policy.global_cond_dim, Z_SELF_COND_DIM)
        self.assertEqual(policy.global_cond_dim, 3072)
        self.assertFalse(hasattr(policy, "residual_transformer"))
        self.assertFalse(hasattr(policy, "latent_action_head"))

        # Check UNet first cond_encoder takes 128 (diffusion step) + 3072 (z_self only) = 3200
        w0 = policy.action_decoder.down_modules[0][0].cond_encoder[1].weight
        self.assertEqual(tuple(w0.shape), (512, 128 + 3072))

        B, P = 4, 12
        demo = torch.randn(B, P, 2, EMBED_DIM)
        pad = torch.zeros(B, P, dtype=torch.bool)
        pad[0, 8:] = True
        query = torch.randn(B, 2, 2, EMBED_DIM)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        policy.train()
        out = policy.compute_loss(demo, pad, query, actions)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

        for name, mod in (
            ("prompt_pooler", policy.prompt_pooler),
            ("reference_reader", policy.reference_reader),
            ("action_decoder", policy.action_decoder),
        ):
            grads = [p.grad for p in mod.parameters() if p.requires_grad]
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in grads),
                f"Missing or non-finite grad in {name}",
            )
            self.assertGreater(
                sum(g.float().norm().item() for g in grads),
                0.0,
                f"Zero grad in {name}",
            )

        # Verify DDIM prediction & frozen Teacher reload round-trip
        policy.eval()
        gen = torch.Generator().manual_seed(7)
        pred = policy.predict_action(demo, pad, query, generator=gen)
        self.assertEqual(tuple(pred["action_10d"].shape), (B, ACTION_HORIZON, ACTION_DIM_10D))
        self.assertEqual(tuple(pred["action_7d"].shape), (B, ACTION_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["exec_action_7d"].shape), (B, EXEC_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["z_self"].shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))

        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "stage1_test.pt"
            torch.save(
                {
                    "stage": "e6_stage1_self_latent",
                    "model": policy.state_dict(),
                    "config": {"use_attention_sink": False},
                },
                ckpt,
            )
            teacher = load_frozen_stage1_teacher(ckpt, device="cpu")
            self.assertEqual(
                sum(p.numel() for p in teacher.parameters() if p.requires_grad), 0
            )
            with torch.inference_mode():
                z_teacher, _ = teacher.encode_z_self(demo, pad, query)
            torch.testing.assert_close(z_teacher, pred["z_self"], atol=1e-5, rtol=1e-5)

    def test_bpp_rgb_checkpoint_partial_compatibility_loading(self):
        if not DEFAULT_INIT_CKPT.is_file():
            self.skipTest(f"Init checkpoint not found: {DEFAULT_INIT_CKPT}")
        policy = E6Stage1Policy(use_attention_sink=False)
        report = policy.load_compatible_from_checkpoint(DEFAULT_INIT_CKPT)

        # PromptPooler 100% loaded; ReferenceReader 100% loaded (sinks unused);
        # ActionDecoder 136/148 loaded, exactly 12 cond_encoder.1.weight incompatible.
        self.assertEqual(report["per_module"]["prompt_pooler"]["reinit_tensors"], 0)
        self.assertEqual(report["per_module"]["reference_reader"]["reinit_tensors"], 0)
        self.assertEqual(report["per_module"]["action_decoder"]["loaded_tensors"], 136)
        self.assertEqual(report["per_module"]["action_decoder"]["reinit_tensors"], 12)
        self.assertEqual(len(report["shape_incompatible_tensors"]), 12)
        self.assertTrue(
            all(
                item["key"].endswith("cond_encoder.1.weight")
                for item in report["shape_incompatible_tensors"]
            )
        )
        self.assertEqual(report["missing_in_source"], [])
        self.assertEqual(len(report["unused_source_tensors"]), 1)
        self.assertEqual(
            report["unused_source_tensors"][0]["key"],
            "reference_reader.attention_sink_tokens",
        )

    def test_self_pair_feature_bank_cache_reuse(self):
        if not (E5_CACHE_DIR / "cache_meta.json").is_file():
            self.skipTest(f"E5 cache not found: {E5_CACHE_DIR}")
        bank = SelfPairFeatureBank(
            cache_dir=E5_CACHE_DIR,
            device=torch.device("cpu"),
            resident_on_device=False,
            seed=42,
        )
        self.assertEqual(bank.num_train_samples(), 1_373_035)
        self.assertEqual(bank.num_val_samples(), 6_248)
        self.assertGreater(len(bank.val_timestep_pairs), 1_000)

        demos, queries, frames, q_rows = bank.build_train_epoch_schedule(1, max_samples=32)
        # 100% Self-pair invariant
        self.assertTrue(np.array_equal(demos, queries))
        d_rgb, pad, q_rgb, act = bank.gather_batch(demos[:8], queries[:8], frames[:8], q_rows[:8])
        self.assertEqual(d_rgb.shape[0], 8)
        self.assertEqual(d_rgb.shape[2:], (2, EMBED_DIM))
        self.assertEqual(tuple(q_rgb.shape), (8, 2, 2, EMBED_DIM))
        self.assertEqual(tuple(act.shape), (8, ACTION_HORIZON, ACTION_DIM_10D))
        self.assertTrue(torch.isfinite(d_rgb).all())
        self.assertTrue(torch.isfinite(q_rgb).all())
        self.assertTrue(torch.isfinite(act).all())

        # Verify every val timestep pair shares the exact same episode with distinct frames
        pa = bank.val_timestep_pairs[:, 0]
        pb = bank.val_timestep_pairs[:, 1]
        self.assertTrue(np.array_equal(bank.val_episodes[pa], bank.val_episodes[pb]))
        self.assertTrue(np.all(bank.val_frames[pb] > bank.val_frames[pa]))


if __name__ == "__main__":
    unittest.main()
