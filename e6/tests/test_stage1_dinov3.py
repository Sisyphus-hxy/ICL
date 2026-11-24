"""Unit and CPU integration tests for E6 Stage 1 (DINOv3 Spatial 4x4, RGB-only)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dinov3 import DEFAULT_CACHE_DIR as DINOV3_CACHE_DIR
from dinov3 import DEFAULT_CLIP_CACHE_DIR, NUM_SPATIAL_TOKENS
from e6 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    EMBED_DIM,
    EXEC_HORIZON,
    NUM_QUERY_TOKENS,
    Z_SELF_COND_DIM,
)
from e6.dinov3_data import DINOv3SelfPairFeatureBank, verify_dinov3_cache_compatibility
from e6.stage1_dinov3_model import (
    DEFAULT_CLIP_STAGE1_CKPT,
    DEFAULT_DEMO_TOKENS_PER_STEP,
    E6Stage1DINOv3Policy,
    SpatialBehaviorReferenceReader,
    load_frozen_stage1_dinov3_teacher,
)
from e6.train_stage1_dinov3 import build_clip_vs_dinov3_comparison, parse_args, run_training


class TestE6Stage1DINOv3(unittest.TestCase):
    def test_no_query_residual_bypass_strict_bottleneck_spatial(self) -> None:
        """Prove mathematically that Query spatial features cannot bypass Demo Memory:
        when spatiotemporal Demo Memory is constant across (P, K_m), `z_self` is strictly
        invariant to `query_feats` and `d(z_self)/d(query_feats) == 0`."""
        torch.manual_seed(0)
        policy = E6Stage1DINOv3Policy(
            feature_mode="spatial",
            demo_tokens_per_step=4,
            use_attention_sink=False,
        )
        policy.eval()
        reader = policy.reference_reader
        self.assertIsNone(reader.attention_sink_tokens)

        B, P, K_m = 2, 8, 4
        with torch.no_grad():
            reader.prompt_pos_emb.zero_()
            reader.memory_spatial_pos_emb.zero_()

        const_tok = torch.randn(B, 1, 1, EMBED_DIM)
        M_d_const = const_tok.expand(B, P, K_m, EMBED_DIM).clone()

        q1 = torch.randn(B, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM, requires_grad=True)
        q2 = torch.randn(B, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM)

        h_q1 = policy.encode_query_tokens(q1)
        h_q2 = policy.encode_query_tokens(q2)
        z1, _ = reader(M_d_const, h_q1)
        z2, _ = reader(M_d_const, h_q2)
        torch.testing.assert_close(z1, z2, atol=1e-5, rtol=1e-5)

        z1.sum().backward()
        self.assertIsNotNone(q1.grad)
        self.assertLess(q1.grad.abs().max().item(), 1e-5)

        # With non-constant Demo Memory, different queries must yield distinct z_self & non-zero grad
        M_d_var = torch.randn(B, P, K_m, EMBED_DIM)
        q3 = torch.randn(B, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM, requires_grad=True)
        q4 = torch.randn(B, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM)
        z3, attn3 = reader(M_d_var, policy.encode_query_tokens(q3), need_weights=True)
        z4, _ = reader(M_d_var, policy.encode_query_tokens(q4))
        self.assertEqual(tuple(z3.shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(attn3.shape), (6, B, 8, NUM_QUERY_TOKENS, P * K_m))
        self.assertGreater((z3 - z4).abs().max().item(), 1e-2)
        z3.sum().backward()
        self.assertGreater(q3.grad.abs().max().item(), 1e-4)

    def test_temporal_attention_aggregation_and_padding_mask(self) -> None:
        torch.manual_seed(2)
        policy = E6Stage1DINOv3Policy(feature_mode="spatial", demo_tokens_per_step=4)
        policy.eval()

        B, P = 3, 10
        demo = torch.randn(B, P, 2, NUM_SPATIAL_TOKENS, EMBED_DIM)
        pad = torch.zeros(B, P, dtype=torch.bool)
        pad[0, 6:] = True
        pad[1, 8:] = True
        query = torch.randn(B, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM)

        with torch.inference_mode():
            z_self, attn_temporal = policy.encode_z_self(
                demo,
                pad,
                query,
                need_weights=True,
                average_attn_weights=True,
                aggregate_attn_temporal=True,
            )
        self.assertEqual(tuple(z_self.shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(attn_temporal.shape), (6, B, NUM_QUERY_TOKENS, P))
        # Aggregated temporal attention sums to 1.0 over P and is 0 on padded timesteps
        torch.testing.assert_close(
            attn_temporal.sum(dim=-1),
            torch.ones(6, B, NUM_QUERY_TOKENS),
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertEqual(attn_temporal[:, 0, :, 6:].abs().max().item(), 0.0)
        self.assertEqual(attn_temporal[:, 1, :, 8:].abs().max().item(), 0.0)

    def test_policy_unet_cond_grad_flow_and_frozen_teacher(self) -> None:
        torch.manual_seed(3)
        policy = E6Stage1DINOv3Policy(feature_mode="spatial", demo_tokens_per_step=4)
        self.assertEqual(policy.global_cond_dim, Z_SELF_COND_DIM)
        self.assertEqual(policy.global_cond_dim, 3072)
        self.assertFalse(hasattr(policy, "residual_transformer"))
        self.assertFalse(hasattr(policy, "latent_action_head"))

        # UNet cond_encoder takes 128 (diffusion step) + 3072 (z_self only) = 3200
        w0 = policy.action_decoder.down_modules[0][0].cond_encoder[1].weight
        self.assertEqual(tuple(w0.shape), (512, 128 + 3072))

        B, P = 3, 7
        demo = torch.randn(B, P, 2, NUM_SPATIAL_TOKENS, EMBED_DIM)
        pad = torch.zeros(B, P, dtype=torch.bool)
        pad[0, 5:] = True
        query = torch.randn(B, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        policy.train()
        out = policy.compute_loss(demo, pad, query, actions)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

        for name, mod in (
            ("prompt_pooler", policy.prompt_pooler),
            ("query_pooler", policy.query_pooler),
            ("reference_reader", policy.reference_reader),
            ("action_decoder", policy.action_decoder),
        ):
            grads = [p.grad for p in mod.parameters() if p.requires_grad]
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in grads),
                f"Missing or non-finite grad in {name}",
            )
            self.assertGreater(sum(g.float().norm().item() for g in grads), 0.0, name)

        policy.eval()
        pred = policy.predict_action(
            demo, pad, query, generator=torch.Generator().manual_seed(7)
        )
        self.assertEqual(tuple(pred["action_10d"].shape), (B, ACTION_HORIZON, ACTION_DIM_10D))
        self.assertEqual(tuple(pred["action_7d"].shape), (B, ACTION_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["exec_action_7d"].shape), (B, EXEC_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["z_self"].shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))

        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "stage1_dinov3_test.pt"
            torch.save(
                {
                    "stage": "e6_stage1_self_latent_dinov3",
                    "model": policy.state_dict(),
                    "config": {
                        "feature_mode": "spatial",
                        "num_spatial_tokens": NUM_SPATIAL_TOKENS,
                        "demo_tokens_per_step": 4,
                        "use_attention_sink": False,
                    },
                },
                ckpt,
            )
            teacher = load_frozen_stage1_dinov3_teacher(ckpt, device="cpu")
            self.assertEqual(
                sum(p.numel() for p in teacher.parameters() if p.requires_grad), 0
            )
            with torch.inference_mode():
                z_teacher, _ = teacher.encode_z_self(demo, pad, query)
            torch.testing.assert_close(z_teacher, pred["z_self"], atol=1e-5, rtol=1e-5)

    def test_clip_stage1_weight_compatibility_modes(self) -> None:
        if not DEFAULT_CLIP_STAGE1_CKPT.is_file():
            self.skipTest(f"CLIP Stage 1 ckpt not found: {DEFAULT_CLIP_STAGE1_CKPT}")

        # 1. Default action_decoder_only: 100% of action_decoder (148 tensors) loaded; visual modules skipped cleanly
        p1 = E6Stage1DINOv3Policy(feature_mode="spatial", demo_tokens_per_step=4)
        rep1 = p1.load_compatible_from_checkpoint(
            DEFAULT_CLIP_STAGE1_CKPT, init_mode="action_decoder_only"
        )
        self.assertEqual(rep1["per_module"]["action_decoder"]["loaded_tensors"], 148)
        self.assertEqual(rep1["per_module"]["action_decoder"]["reinit_tensors"], 0)
        self.assertEqual(rep1["per_module"]["prompt_pooler"]["loaded_tensors"], 0)
        self.assertEqual(rep1["per_module"]["query_pooler"]["loaded_tensors"], 0)
        self.assertEqual(rep1["per_module"]["reference_reader"]["loaded_tensors"], 0)
        self.assertEqual(rep1["shape_incompatible_tensors"], [])

        # 2. all_compatible: explicitly catches prompt_pooler.prompt_attention_pool.latent shape mismatch (1,4,768 vs 1,1,768)
        p2 = E6Stage1DINOv3Policy(feature_mode="spatial", demo_tokens_per_step=4)
        rep2 = p2.load_compatible_from_checkpoint(
            DEFAULT_CLIP_STAGE1_CKPT, init_mode="all_compatible"
        )
        self.assertEqual(rep2["per_module"]["action_decoder"]["loaded_tensors"], 148)
        self.assertEqual(rep2["per_module"]["reference_reader"]["loaded_tensors"], 110)
        incompat_keys = [x["key"] for x in rep2["shape_incompatible_tensors"]]
        self.assertIn("prompt_pooler.prompt_attention_pool.latent", incompat_keys)
        self.assertIn("reference_reader.memory_spatial_pos_emb", rep2["missing_in_source"])

    def test_real_dinov3_cache_compatibility_and_reader(self) -> None:
        if not (DINOV3_CACHE_DIR / "cache_meta.json").is_file():
            self.skipTest(f"DINOv3 cache not found: {DINOV3_CACHE_DIR}")

        audit = verify_dinov3_cache_compatibility(
            dinov3_cache_dir=DINOV3_CACHE_DIR,
            clip_cache_dir=DEFAULT_CLIP_CACHE_DIR,
            feature_mode="spatial",
            sample_rows=256,
        )
        self.assertEqual(audit["demo_shape"], [188840, 2, 16, 768])
        self.assertEqual(audit["query_shape"], [1825431, 2, 16, 768])
        self.assertTrue(audit["index_parity"]["episodes_identical"])
        self.assertTrue(audit["index_parity"]["query_frame_nums_identical"])
        self.assertTrue(audit["full_completeness_audit_passed"])

        # Lightweight mmap bank check (no 9.3 GB RAM copy in unit test)
        bank = DINOv3SelfPairFeatureBank(
            cache_dir=DINOV3_CACHE_DIR,
            feature_mode="spatial",
            device=torch.device("cpu"),
            demo_in_ram=False,
            preload_val_query=False,
            superchunk_episodes=4,
            seed=42,
        )
        self.assertEqual(bank.num_train_samples(), 1_373_035)
        self.assertEqual(bank.num_val_samples(), 6_248)
        self.assertEqual(len(bank.val_timestep_pairs), 2_557)

        # Pull 1 train batch via super-chunk streaming
        d_f, pad, q_f, act = next(
            bank.iter_train_batches(1, batch_size=4, max_samples=16)
        )
        self.assertEqual(d_f.shape[0], 4)
        self.assertEqual(d_f.shape[2:], (2, NUM_SPATIAL_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(q_f.shape), (4, 2, 2, NUM_SPATIAL_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(act.shape), (4, ACTION_HORIZON, ACTION_DIM_10D))
        self.assertTrue(torch.isfinite(d_f).all())
        self.assertTrue(torch.isfinite(q_f).all())
        self.assertTrue(torch.isfinite(act).all())

    def test_end_to_end_mini_cpu_smoke_training(self) -> None:
        """End-to-end CPU smoke test of train_stage1_dinov3 on real DINOv3 cache (tiny batch)."""
        if not (DINOV3_CACHE_DIR / "cache_meta.json").is_file():
            self.skipTest(f"DINOv3 cache not found: {DINOV3_CACHE_DIR}")

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "stage1"
            args = parse_args([])
            args.cache_dir = DINOV3_CACHE_DIR
            args.output_dir = out_dir
            args.run_name = "cpu_smoke_test"
            args.feature_mode = "spatial"
            args.demo_tokens_per_step = DEFAULT_DEMO_TOKENS_PER_STEP
            args.init_mode = "action_decoder_only"
            args.epochs = 1
            args.global_batch = 2
            args.val_batch_size = 2
            args.superchunk_episodes = 2
            args.demo_mmap = True
            args.no_preload_val_query = True
            args.max_train_samples = 4
            args.max_val_samples = 4
            args.sample_count = 2
            args.timestep_pairs = 2
            args.final_ddim_samples = 2
            args.final_timestep_pairs = 2
            args.sample_every = 1
            args.skip_init_val = True
            args.cpu = True

            run_training(args)

            run_dir = out_dir / "cpu_smoke_test"
            for fname in (
                "config.json",
                "metrics.json",
                "best.pt",
                "last.pt",
                "final_eval.json",
                "clip_vs_dinov3_comparison.json",
                "clip_vs_dinov3_comparison.md",
            ):
                self.assertTrue((run_dir / fname).is_file(), f"Missing output: {fname}")

            final_rep = json.loads((run_dir / "final_eval.json").read_text())
            self.assertEqual(final_rep["frozen_teacher_trainable_params"], 0)
            self.assertIn("ddim", final_rep["final_eval"])
            self.assertIn("timestep_distinctiveness", final_rep["final_eval"])
            self.assertIn(
                "layer0_attn_peak_shift_steps",
                final_rep["final_eval"]["timestep_distinctiveness"],
            )


if __name__ == "__main__":
    unittest.main()
