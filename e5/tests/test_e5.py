"""Unit tests for E5 architecture, gradients, cache consistency, diffusion, and 10D<->7D actions."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from e5 import ACTION_DIM_10D, ACTION_DIM_7D, ACTION_HORIZON, DEMO_STRIDE, EMBED_DIM, EXEC_HORIZON
from e5.actions import (
    ActionNormalizer,
    action_10d_to_7d,
    action_7d_to_10d,
    extract_exec_actions,
)
from e5.bpp_compat import load_bpp_pretrained_weights
from e5.cache import DEFAULT_CACHE_DIR, build_feature_cache, verify_cache_consistency
from e5.data import E5FeatureBank
from e5.diffusion import DDIMScheduler, EMAModel, get_cosine_schedule_with_warmup
from e5.model import E5Policy, NUM_QUERY_TOKENS
from e5.vision import BPPVisualEncoder


class TestE5Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bpp_pretrained_weights()

    def test_action_10d_7d_roundtrip_and_e4_equivalence(self):
        from e4_chain.live_model import action10, environment_actions

        rng = np.random.default_rng(7)
        raw7 = rng.uniform(-0.9, 0.9, size=(128, ACTION_DIM_7D)).astype(np.float32)
        raw7[:, 6] = rng.choice([-1.0, 1.0], size=128).astype(np.float32)

        a10_np = action_7d_to_10d(raw7)
        a10_e4 = action10(raw7)
        self.assertEqual(a10_np.shape, (128, ACTION_DIM_10D))
        np.testing.assert_allclose(a10_np, a10_e4, atol=1e-6)

        back7_np = action_10d_to_7d(a10_np)
        back7_e4 = environment_actions(a10_np)
        np.testing.assert_allclose(back7_np, raw7, atol=1e-5)
        np.testing.assert_allclose(back7_np, back7_e4, atol=1e-5)

        # Torch path + 16-step -> 12-step exec extraction
        raw7_t = torch.from_numpy(raw7.reshape(8, ACTION_HORIZON, ACTION_DIM_7D))
        a10_t = action_7d_to_10d(raw7_t)
        exec7_t = extract_exec_actions(a10_t, exec_horizon=EXEC_HORIZON)
        self.assertEqual(tuple(exec7_t.shape), (8, EXEC_HORIZON, ACTION_DIM_7D))
        np.testing.assert_allclose(
            exec7_t.numpy(), raw7_t[:, :EXEC_HORIZON].numpy(), atol=1e-5
        )

        # ActionNormalizer round-trip & BPP behavior (pos scaled [-1,1], rot6d+grip identity)
        norm = ActionNormalizer.fit_from_train_actions_10d(a10_np)
        normed = norm.normalize(a10_t)
        unnormed = norm.unnormalize(normed)
        torch.testing.assert_close(unnormed, a10_t, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(norm.scale[3:], torch.ones(7))
        torch.testing.assert_close(norm.offset[3:], torch.zeros(7))

    def test_ddim_and_ema_behavior(self):
        sched = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            prediction_type="epsilon",
        )
        sched.set_timesteps(16)
        self.assertEqual(len(sched.timesteps), 16)
        self.assertEqual(int(sched.timesteps[-1].item()), 0)

        x0 = torch.randn(4, ACTION_HORIZON, ACTION_DIM_10D)
        eps = torch.randn_like(x0)
        t = torch.tensor([0, 10, 25, 49], dtype=torch.long)
        xt = sched.add_noise(x0, eps, t)
        self.assertEqual(xt.shape, x0.shape)
        self.assertTrue(torch.isfinite(xt).all())

        # Single denoising step returns valid prev_sample
        out = sched.step(eps, int(sched.timesteps[0].item()), xt)
        self.assertEqual(out.prev_sample.shape, x0.shape)
        self.assertTrue(torch.isfinite(out.prev_sample).all())

        # EMA store/copy/restore round-trip
        layer = torch.nn.Linear(4, 4)
        ema = EMAModel(layer.parameters(), decay=0.99, foreach=True)
        w_init = layer.weight.detach().clone()
        with torch.no_grad():
            layer.weight.add_(1.0)
        ema.step(layer.parameters())
        ema.step(layer.parameters())
        w_live = layer.weight.detach().clone()
        ema.store(layer.parameters())
        ema.copy_to(layer.parameters())
        self.assertFalse(torch.allclose(layer.weight, w_init))
        ema.restore(layer.parameters())
        torch.testing.assert_close(layer.weight, w_live)

        # Cosine LR schedule with warmup
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        lrs = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
        vals = []
        for _ in range(100):
            vals.append(lrs.get_last_lr()[0])
            opt.step()
            lrs.step()
        self.assertAlmostEqual(vals[0], 0.0, places=7)
        self.assertAlmostEqual(vals[10], 1e-3, places=6)
        self.assertLess(vals[-1], 1e-5)

    def test_frozen_clip_encoder_strict_load_and_freeze(self):
        enc = BPPVisualEncoder(pretrained_bundle=self.bundle, load_weights=True)
        total, trainable = enc.parameter_count()
        self.assertEqual(total, 171_598_848)
        self.assertEqual(trainable, 0)
        # Calling .train(True) must keep frozen CLIP in eval mode
        enc.train(True)
        self.assertFalse(enc.training)
        self.assertFalse(enc.key_model_map["agentview_rgb"].training)

        av = np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8)
        eh = np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8)
        feats = enc.encode_dual_camera_uint8(av, eh, convention="opengl")
        self.assertEqual(tuple(feats.shape), (2, 2, EMBED_DIM))
        self.assertTrue(torch.isfinite(feats).all())

    def test_e5_shapes_param_counts_zero_init_and_gradients(self):
        policy = E5Policy(
            use_residual=True,
            pretrained_bundle=self.bundle,
            load_pretrained_reader=True,
        )
        summary = policy.parameter_summary()
        mods = summary["modules"]
        # Check architectural parameter targets against prompt specs:
        # Prompt Pooling ~7.1M, Reference Reader ~57M, Residual Transformer ~28-29M, Action Decoder ~151M
        self.assertEqual(mods["prompt_pooler"]["trainable"], 7_085_568)
        self.assertEqual(mods["reference_reader"]["trainable"], 56_755_200)
        self.assertGreater(mods["residual_transformer"]["trainable"], 28_000_000)
        self.assertLess(mods["residual_transformer"]["trainable"], 29_500_000)
        self.assertEqual(mods["action_decoder"]["trainable"], 151_075_722)
        self.assertTrue(policy.weight_compat_report["reference_reader"]["decoder_loaded"])
        self.assertTrue(policy.weight_compat_report["reference_reader"]["prompt_pos_emb_loaded"])
        self.assertTrue(policy.weight_compat_report["reference_reader"]["attention_sink_loaded"])
        self.assertTrue(
            policy.weight_compat_report["reference_reader"]["current_obs_pos_emb_loaded"]
        )

        # Matched baseline BPP RGB-only has exact same pooling, reader, decoder and 0 residual params
        baseline = E5Policy(
            use_residual=False,
            pretrained_bundle=self.bundle,
            load_pretrained_reader=True,
        )
        b_mods = baseline.parameter_summary()["modules"]
        self.assertEqual(b_mods["residual_transformer"]["total"], 0)
        self.assertEqual(b_mods["prompt_pooler"], mods["prompt_pooler"])
        self.assertEqual(b_mods["reference_reader"], mods["reference_reader"])
        self.assertEqual(b_mods["action_decoder"], mods["action_decoder"])

        # Forward shapes & exact zero-init identity: delta_z == 0, z_q == z_ref
        B, P = 4, 12
        demo = torch.randn(B, P, 2, EMBED_DIM)
        pad = torch.zeros(B, P, dtype=torch.bool)
        pad[0, 8:] = True
        pad[1, 5:] = True
        query = torch.randn(B, 2, 2, EMBED_DIM)
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
        is_self = torch.tensor([True, True, False, False], dtype=torch.bool)

        policy.eval()
        hq, z_ref, delta_z, z_q, attn = policy.encode_conditioning(
            demo, pad, query, need_weights=True
        )
        self.assertEqual(tuple(hq.shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(z_ref.shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(delta_z.shape), (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(tuple(attn.shape), (6, B, 8, NUM_QUERY_TOKENS, P))
        torch.testing.assert_close(delta_z, torch.zeros_like(delta_z), atol=0.0, rtol=0.0)
        torch.testing.assert_close(z_q, z_ref, atol=0.0, rtol=0.0)

        # Check Self-pair isolation: on self-only batch with zero residual penalty,
        # diffusion loss MUST NOT send gradient to residual_transformer!
        policy.train()
        out_self_only = policy.compute_loss(
            demo,
            pad,
            query,
            actions,
            torch.ones(B, dtype=torch.bool),
            self_res_weight=0.0,
            cross_res_weight=0.0,
        )
        out_self_only["loss"].backward()
        res_grad_norm = sum(
            p.grad.abs().sum().item()
            for p in policy.residual_transformer.parameters()
            if p.grad is not None
        )
        self.assertEqual(res_grad_norm, 0.0)

        # Step 1 & Step 2 on mixed self/cross batch: after step 1 updates zero-init out_proj,
        # step 2 must produce finite non-zero gradients in ALL trainable modules!
        opt = torch.optim.AdamW(policy.get_optim_groups(lr=1e-3, weight_decay=1e-6))
        for step in (1, 2):
            opt.zero_grad(set_to_none=True)
            out = policy.compute_loss(demo, pad, query, actions, is_self)
            out["loss"].backward()
            if step == 2:
                for mod_name, mod in (
                    ("prompt_pooler", policy.prompt_pooler),
                    ("reference_reader", policy.reference_reader),
                    ("residual_transformer", policy.residual_transformer),
                    ("action_decoder", policy.action_decoder),
                ):
                    grads = [p.grad for p in mod.parameters() if p.requires_grad]
                    self.assertTrue(
                        all(g is not None and torch.isfinite(g).all() for g in grads),
                        f"Missing or non-finite grad in {mod_name}",
                    )
                    gn = sum(g.float().norm().item() for g in grads)
                    self.assertGreater(gn, 0.0, f"Zero grad norm in {mod_name}")
            opt.step()

        # Predict action interface returns 16x10D, 16x7D, and 12x7D finite controls
        policy.eval()
        pred = policy.predict_action(demo, pad, query)
        self.assertEqual(tuple(pred["action_10d"].shape), (B, ACTION_HORIZON, ACTION_DIM_10D))
        self.assertEqual(tuple(pred["action_7d"].shape), (B, ACTION_HORIZON, ACTION_DIM_7D))
        self.assertEqual(tuple(pred["exec_action_7d"].shape), (B, EXEC_HORIZON, ACTION_DIM_7D))
        self.assertTrue(torch.isfinite(pred["exec_action_7d"]).all())

    def test_smoke_cache_end_to_end_and_online_offline_consistency(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "smoke_cache"
            build_feature_cache(
                cache_dir=cache_dir,
                max_tasks=2,
                batch_size=256,
                precision="fp16",
                num_gpus=1,
            )
            audit = verify_cache_consistency(cache_dir=cache_dir, num_episodes=4)
            self.assertGreater(audit["demo_min_cosine_sim"], 0.99999)
            self.assertGreater(audit["query_min_cosine_sim"], 0.99999)
            self.assertLess(audit["demo_max_rel_l2"], 2e-3)
            self.assertLess(audit["query_max_rel_l2"], 2e-3)
            self.assertLess(audit["action_10d_max_abs_diff"], 1e-6)

            # Verify E5FeatureBank 50/50 self/cross schedule and zero-I/O batch gathering
            bank = E5FeatureBank(
                cache_dir=cache_dir,
                device=torch.device("cpu"),
                resident_on_device=False,
                self_ratio=0.5,
                seed=42,
            )
            demos, queries, frames, q_rows, is_self = bank.build_train_epoch_schedule(1)
            self.assertGreater(len(demos), 0)
            self_frac = float(is_self.mean())
            self.assertAlmostEqual(self_frac, 0.5, delta=0.05)
            # Check demo==query iff is_self
            self.assertTrue(np.array_equal(demos == queries, is_self))
            # Gather a batch and verify exact shapes
            b = min(8, len(demos))
            d_rgb, pad, q_rgb, act, s_t = bank.gather_batch(
                demos[:b], queries[:b], frames[:b], q_rows[:b], is_self[:b]
            )
            self.assertEqual(d_rgb.shape[0], b)
            self.assertEqual(d_rgb.shape[2:], (2, EMBED_DIM))
            self.assertEqual(tuple(q_rgb.shape), (b, 2, 2, EMBED_DIM))
            self.assertEqual(tuple(act.shape), (b, ACTION_HORIZON, ACTION_DIM_10D))
            self.assertEqual(tuple(s_t.shape), (b,))

            # Check temporal indices preserve 1Hz stride=20
            meta = json.loads((cache_dir / "cache_meta.json").read_text())
            self.assertEqual(meta["demo_stride"], DEMO_STRIDE)


if __name__ == "__main__":
    unittest.main()
