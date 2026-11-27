"""Unit tests for E8: Source-conditioned Latent Behavior Transfer."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from e6.stage1_model import E6Stage1Policy
from e8 import ACTION_DIM_10D, ACTION_HORIZON, EMBED_DIM, FIXED_LATENT_SCALE_S_Z, NUM_QUERY_TOKENS
from e8.model import (
    E8Policy,
    SourceConditionedBehaviorReader,
    load_e8_policy,
)
from e8.train import _build_cross_episode_derangement, _shift_weights_within_demo


class TestE8(unittest.TestCase):
    def _build_dummy_e8(
        self,
        *,
        direct_latent_bias: torch.Tensor | None = None,
        direct_out_proj_init_std: float = 0.002,
    ) -> E8Policy:
        torch.manual_seed(0)
        teacher = E6Stage1Policy(use_attention_sink=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return E8Policy(
            teacher,
            latent_scale_s_z=FIXED_LATENT_SCALE_S_Z,
            direct_latent_bias=direct_latent_bias,
            direct_out_proj_init_std=direct_out_proj_init_std,
        )

    def test_shared_weights_behavior_and_source_state_readout(self):
        torch.manual_seed(1)
        reader = SourceConditionedBehaviorReader()

        # Verify ONLY Q/K parameters exist (no V proj, no out proj, no skip)
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

        h_mem_corrupt = h_mem.clone()
        b_mem_corrupt = b_mem.clone()
        h_mem_corrupt[pad_mask] = 1e4
        b_mem_corrupt[pad_mask] = -1e4

        z_c1, h_ref1, w1 = reader(h_q, h_mem, b_mem, mem_pad_mask=pad_mask)
        z_c2, h_ref2, w2 = reader(h_q, h_mem_corrupt, b_mem_corrupt, mem_pad_mask=pad_mask)

        self.assertEqual(w1.shape, (B, M))
        self.assertEqual(z_c1.shape, (B, NUM_QUERY_TOKENS, EMBED_DIM))
        self.assertEqual(h_ref1.shape, (B, NUM_QUERY_TOKENS, EMBED_DIM))
        torch.testing.assert_close(w1, w2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(z_c1, z_c2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(h_ref1, h_ref2, atol=1e-6, rtol=1e-6)

        # Padded weights strictly 0, valid sum to 1
        self.assertEqual(float(w1[pad_mask].abs().max().item()), 0.0)
        torch.testing.assert_close(w1.sum(dim=-1), torch.ones(B), atol=1e-6, rtol=1e-6)

        # BOTH z_c and h_ref use the EXACT SAME weights w1
        manual_zc = sum(w1[:, i, None, None] * b_mem[:, i] for i in range(M))
        manual_href = sum(w1[:, i, None, None] * h_mem[:, i] for i in range(M))
        torch.testing.assert_close(z_c1, manual_zc, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(h_ref1, manual_href, atol=1e-6, rtol=1e-6)

    def test_independent_type_embeddings_and_optim_groups(self):
        policy = self._build_dummy_e8()
        xf = policy.transfer_transformer

        # Three distinct type embeddings with different initial values
        self.assertEqual(xf.behavior_type_emb.shape, (1, 1, EMBED_DIM))
        self.assertEqual(xf.source_type_emb.shape, (1, 1, EMBED_DIM))
        self.assertEqual(xf.target_type_emb.shape, (1, 1, EMBED_DIM))
        self.assertFalse(torch.equal(xf.behavior_type_emb, xf.source_type_emb))
        self.assertFalse(torch.equal(xf.source_type_emb, xf.target_type_emb))
        self.assertFalse(torch.equal(xf.behavior_type_emb, xf.target_type_emb))

        groups = policy.get_optim_groups(reader_lr=1e-4, transformer_lr=1e-4, weight_decay=0.01)
        by_name = {g["name"]: g for g in groups}
        no_decay_ids = {id(p) for p in by_name["transfer_transformer_no_decay"]["params"]}
        for p in (
            xf.behavior_type_emb,
            xf.source_type_emb,
            xf.target_type_emb,
            xf.ref_pos_emb,
            xf.state_pos_emb,
        ):
            self.assertIn(id(p), no_decay_ids)

    def test_direct_no_skip_frozen_teacher_and_three_input_grad_flow(self):
        torch.manual_seed(7)
        mean_bias = torch.randn(1, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        policy = self._build_dummy_e8(
            direct_latent_bias=mean_bias, direct_out_proj_init_std=0.002
        )
        self.assertFalse(policy.direct_latent_bias.requires_grad)

        B, M = 3, 5
        h_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        mem_mask = torch.zeros(B, M, dtype=torch.bool)
        mem_mask[:, -1] = True
        query_rgb = torch.randn(B, 2, 2, EMBED_DIM)
        z_self = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        # Verify NO residual skip connection: when out_proj == 0, z_q == direct_latent_bias
        zero_policy = self._build_dummy_e8(
            direct_latent_bias=mean_bias, direct_out_proj_init_std=0.0
        )
        zero_policy.eval()
        z_c0, h_ref0, h_q0, _ = zero_policy.read_cross_latent_and_source(
            h_mem, b_mem, mem_mask, query_rgb
        )
        z_q0, _ = zero_policy.transfer_latent(z_c0, h_ref0, h_q0)
        torch.testing.assert_close(z_q0, mean_bias.expand(B, -1, -1), atol=1e-6, rtol=1e-6)
        self.assertGreater((z_q0 - z_c0).abs().mean().item(), 1.0)

        # Verify TransferTransformer actually depends on all 3 inputs (z_c, h_ref, h_q)
        policy.eval()
        zc_in = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM, requires_grad=True)
        href_in = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM, requires_grad=True)
        hq_in = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM, requires_grad=True)
        zq_out, _ = policy.transfer_latent(zc_in, href_in, hq_in)
        zq_out.square().mean().backward()
        self.assertGreater(zc_in.grad.norm().item(), 0.0)
        self.assertGreater(href_in.grad.norm().item(), 0.0)
        self.assertGreater(hq_in.grad.norm().item(), 0.0)

        # Verify changing h_ref while fixing (z_c, h_q) alters z_q
        with torch.no_grad():
            zq_a, _ = policy.transfer_latent(zc_in.detach(), href_in.detach(), hq_in.detach())
            zq_b, _ = policy.transfer_latent(
                zc_in.detach(), torch.randn_like(href_in), hq_in.detach()
            )
            self.assertGreater((zq_a - zq_b).abs().mean().item(), 0.0)

        # Pure L_diff flows through frozen UNet into Reader Q/K + TransferTransformer + 3 type embs
        teacher_snap = {k: v.detach().clone() for k, v in policy.teacher.state_dict().items()}
        policy.train(True)
        self.assertFalse(policy.teacher.training)
        opt = torch.optim.AdamW(
            policy.get_optim_groups(reader_lr=1e-4, transformer_lr=1e-4, weight_decay=0.01),
            betas=(0.9, 0.95),
        )
        opt.zero_grad(set_to_none=True)
        out1 = policy(
            h_mem, b_mem, mem_mask, query_rgb, z_self, actions, lambda_a=1.0, lambda_z=0.0
        )
        out1["loss"].backward()
        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertGreater(policy.reader.q_proj.weight.grad.float().norm().item(), 0.0)
        self.assertGreater(policy.reader.k_proj.weight.grad.float().norm().item(), 0.0)
        self.assertGreater(
            policy.transfer_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )
        self.assertGreater(
            policy.transfer_transformer.source_in_norm.weight.grad.float().norm().item(), 0.0
        )
        self.assertGreater(
            policy.transfer_transformer.behavior_type_emb.grad.float().norm().item(), 0.0
        )
        self.assertGreater(
            policy.transfer_transformer.source_type_emb.grad.float().norm().item(), 0.0
        )
        self.assertGreater(
            policy.transfer_transformer.target_type_emb.grad.float().norm().item(), 0.0
        )
        opt.step()

        # Full loss L = L_diff + 0.5 * L_z
        opt.zero_grad(set_to_none=True)
        out2 = policy(
            h_mem, b_mem, mem_mask, query_rgb, z_self, actions, lambda_a=1.0, lambda_z=0.5
        )
        expected = out2["diff_loss"] + 0.5 * out2["latent_loss_norm"]
        torch.testing.assert_close(out2["loss"].detach(), expected, atol=1e-6, rtol=1e-6)
        out2["loss"].backward()
        opt.step()

        # Frozen Stage 1 Teacher & direct_latent_bias remain bit-exact
        for k, v in policy.teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Teacher param changed: {k}")
        torch.testing.assert_close(policy.direct_latent_bias, mean_bias, atol=0.0, rtol=0.0)

    def test_video_only_extraction_shuffle_helpers_and_ckpt_roundtrip(self):
        mean_bias = torch.randn(1, NUM_QUERY_TOKENS, EMBED_DIM) * 7.5
        policy = self._build_dummy_e8(direct_latent_bias=mean_bias)
        policy.eval()

        P, M = 10, 12
        demo_1hz = torch.randn(P, 2, EMBED_DIM)
        anchor_q = torch.randn(M, 2, 2, EMBED_DIM)
        h_mem, b_mem = policy.extract_behavior_memory_from_single_demo(demo_1hz, anchor_q)
        z_ref, _ = policy.teacher.encode_z_self(
            demo_1hz.unsqueeze(0).expand(M, P, 2, EMBED_DIM), None, anchor_q
        )
        torch.testing.assert_close(b_mem, z_ref, atol=1e-5, rtol=1e-5)

        # Derangement & within-demo weight shift helpers
        q_ep = np.repeat(np.arange(10), 5)
        d_ep = np.repeat(np.arange(10) + 100, 5)
        perm = _build_cross_episode_derangement(q_ep, d_ep, seed=123)
        self.assertEqual(sorted(perm.tolist()), list(range(len(q_ep))))
        self.assertTrue(np.all(q_ep[perm] != q_ep))
        self.assertTrue(np.all(d_ep[perm] != d_ep))

        w = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0, 0.0]])
        pm = torch.tensor(
            [[False, False, False, False, True], [False, False, False, True, True]]
        )
        ws = _shift_weights_within_demo(w, pm)
        torch.testing.assert_close(ws.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
        self.assertEqual(float(ws[pm].abs().max().item()), 0.0)
        self.assertAlmostEqual(float(ws[0, 2].item()), 0.9, places=6)

        # Checkpoint roundtrip
        tmp_root = Path(__file__).resolve().parents[1] / "outputs" / "source_conditioned"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=tmp_root) as tmp:
            s1_path = Path(tmp) / "s1.pt"
            e8_path = Path(tmp) / "e8.pt"
            torch.save({"model": policy.teacher.state_dict(), "config": {}}, s1_path)
            torch.save(
                {
                    "stage": "e8_source_conditioned_transfer",
                    "prediction_mode": "direct",
                    "source_conditioned": True,
                    "direct_latent_bias": policy.direct_latent_bias.detach().cpu(),
                    "epoch": 1,
                    "latent_scale_s_z": FIXED_LATENT_SCALE_S_Z,
                    "reader": policy.reader.state_dict(),
                    "transfer_transformer": policy.transfer_transformer.state_dict(),
                    "config": {
                        "stage1_ckpt": str(s1_path),
                        "direct_out_proj_init_std": 0.002,
                    },
                },
                e8_path,
            )
            reloaded, _ = load_e8_policy(e8_path, device="cpu")
            torch.testing.assert_close(
                reloaded.direct_latent_bias, policy.direct_latent_bias, atol=0.0, rtol=0.0
            )
            q = torch.randn(2, 2, 2, EMBED_DIM)
            z0 = torch.randn(2, ACTION_HORIZON, ACTION_DIM_10D)
            hm_b = h_mem.unsqueeze(0).expand(2, M, NUM_QUERY_TOKENS, EMBED_DIM)
            bm_b = b_mem.unsqueeze(0).expand(2, M, NUM_QUERY_TOKENS, EMBED_DIM)
            pm_b = torch.zeros(2, M, dtype=torch.bool)
            o1 = policy.predict_action_from_behavior_memory(hm_b, bm_b, pm_b, q, init_noise=z0)
            o2 = reloaded.predict_action_from_behavior_memory(hm_b, bm_b, pm_b, q, init_noise=z0)
            torch.testing.assert_close(o1["action_10d"], o2["action_10d"], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(o1["z_adapted"], o2["z_adapted"], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(o1["h_ref"], o2["h_ref"], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
