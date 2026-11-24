"""Unit tests for E7.1: Adjacent-Pair Behavior Retrieval + Latent Adaptation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from e6.stage1_model import E6Stage1Policy
from e7 import ACTION_DIM_10D, ACTION_HORIZON, EMBED_DIM, FIXED_LATENT_SCALE_S_Z, NUM_QUERY_TOKENS
from e7.adjacent_model import (
    AdjacentPairBehaviorReader,
    E7AdjacentPolicy,
    load_e7_adjacent_policy,
)


class TestE7AdjacentPair(unittest.TestCase):
    def _build_dummy_e7_1(self) -> E7AdjacentPolicy:
        torch.manual_seed(0)
        teacher = E6Stage1Policy(use_attention_sink=False)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return E7AdjacentPolicy(teacher, latent_scale_s_z=FIXED_LATENT_SCALE_S_Z)

    def test_strict_adjacent_pair_only_and_train_eval_equivalence(self):
        torch.manual_seed(7)
        reader = AdjacentPairBehaviorReader(temperature=1.0)

        B, M = 4, 9
        h_q = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        h_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        pad_mask = torch.zeros(B, M, dtype=torch.bool)
        pad_mask[0, 6:] = True
        pad_mask[1, 4:] = True
        pad_mask[2, 2:] = True  # only 1 valid pair (0, 1)

        # 1. Training hard forward vs Eval deterministic forward MUST match identically
        reader.train(True)
        z_train, info_tr = reader(h_q, h_mem, b_mem, mem_pad_mask=pad_mask)
        reader.eval()
        z_eval, info_ev = reader(h_q, h_mem, b_mem, mem_pad_mask=pad_mask)

        self.assertTrue(torch.equal(info_tr["pair_idx"], info_ev["pair_idx"]))
        torch.testing.assert_close(z_train, z_eval, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(info_tr["beta_selected"], info_ev["beta_selected"], atol=1e-6, rtol=1e-6)

        # 2. Verify selected pair is strictly valid (never touches padded entry)
        valid_counts = (~pad_mask).sum(dim=-1)
        for b in range(B):
            i_star = int(info_tr["pair_idx"][b].item())
            self.assertGreaterEqual(i_star, 0)
            self.assertLess(i_star + 1, int(valid_counts[b].item()))

        # 3. Corrupt ALL entries in b_mem EXCEPT (i*, i*+1) with 1e6 -> train forward MUST NOT change!
        b_corrupt = b_mem.clone()
        for b in range(B):
            i_star = int(info_tr["pair_idx"][b].item())
            for j in range(M):
                if j not in (i_star, i_star + 1):
                    b_corrupt[b, j] = 1e6
        reader.train(True)
        z_corrupt, _ = reader(h_q, h_mem, b_corrupt, mem_pad_mask=pad_mask)
        torch.testing.assert_close(z_train, z_corrupt, atol=0.0, rtol=0.0)

        # 4. Exact formula z_c = (1 - beta_{i*}) * b_{i*} + beta_{i*} * b_{i*+1}
        for b in range(B):
            i_star = int(info_tr["pair_idx"][b].item())
            beta = info_tr["beta_selected"][b]
            expected = (1.0 - beta) * b_mem[b, i_star] + beta * b_mem[b, i_star + 1]
            torch.testing.assert_close(z_train[b], expected, atol=1e-6, rtol=1e-6)

        # 5. Effective entry weights have strictly at most 2 adjacent non-zeros summing to 1
        eff_w = info_tr["eff_entry_weights"]
        torch.testing.assert_close(eff_w.sum(dim=-1), torch.ones(B), atol=1e-6, rtol=1e-6)
        for b in range(B):
            nz = torch.nonzero(eff_w[b] > 0, as_tuple=False).flatten().tolist()
            i_star = int(info_tr["pair_idx"][b].item())
            self.assertEqual(nz, [i_star, i_star + 1])

    def test_single_valid_entry_edge_cases(self):
        torch.manual_seed(3)
        reader = AdjacentPairBehaviorReader()

        # Case A: M == 1
        h_q = torch.randn(3, NUM_QUERY_TOKENS, EMBED_DIM)
        h_mem1 = torch.randn(3, 1, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem1 = torch.randn(3, 1, NUM_QUERY_TOKENS, EMBED_DIM)
        for mode in (True, False):
            reader.train(mode)
            z1, inf1 = reader(h_q, h_mem1, b_mem1)
            torch.testing.assert_close(z1, b_mem1[:, 0], atol=0.0, rtol=0.0)
            self.assertFalse(inf1["has_pair"].any().item())

        # Case B: Mixed batch (row 0 has 1 valid entry, row 1 has 5 valid entries)
        h_mem = torch.randn(2, 6, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(2, 6, NUM_QUERY_TOKENS, EMBED_DIM)
        pm = torch.zeros(2, 6, dtype=torch.bool)
        pm[0, 1:] = True  # only entry 0 valid
        pm[1, 5:] = True  # 5 valid entries -> 4 valid pairs
        for mode in (True, False):
            reader.train(mode)
            zm, infm = reader(h_q[:2], h_mem, b_mem, mem_pad_mask=pm)
            self.assertTrue(torch.isfinite(zm).all().item())
            torch.testing.assert_close(zm[0], b_mem[0, 0], atol=1e-6, rtol=1e-6)
            self.assertFalse(bool(infm["has_pair"][0].item()))
            self.assertTrue(bool(infm["has_pair"][1].item()))

    def test_ste_gradients_through_frozen_unet_and_checkpoint_roundtrip(self):
        policy = self._build_dummy_e7_1()
        teacher_snap = {k: v.detach().clone() for k, v in policy.teacher.state_dict().items()}

        B, M = 4, 7
        h_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        b_mem = torch.randn(B, M, NUM_QUERY_TOKENS, EMBED_DIM)
        pm = torch.zeros(B, M, dtype=torch.bool)
        pm[0, 5:] = True
        q_rgb = torch.randn(B, 2, 2, EMBED_DIM)
        z_self = torch.randn(B, NUM_QUERY_TOKENS, EMBED_DIM)
        act = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)

        policy.train(True)
        opt = torch.optim.AdamW(
            policy.get_optim_groups(reader_lr=1e-4, residual_lr=1e-4, weight_decay=0.01),
            betas=(0.9, 0.95),
        )
        opt.zero_grad(set_to_none=True)
        out = policy(h_mem, b_mem, pm, q_rgb, z_self, act, lambda_a=1.0, lambda_z=0.0)
        out["loss"].backward()

        self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
        self.assertGreater(
            policy.residual_transformer.out_proj.weight.grad.float().norm().item(), 0.0
        )
        for name, p in policy.reader.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing grad for reader.{name}")
            gn = float(p.grad.float().norm().item())
            self.assertGreater(gn, 0.0, f"Zero grad for reader.{name}")
            self.assertTrue(torch.isfinite(p.grad).all().item(), f"Non-finite grad for reader.{name}")
        opt.step()

        for k, v in policy.teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Teacher param changed: {k}")

        tmp_root = Path(__file__).resolve().parents[1] / "outputs"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=tmp_root) as tmp:
            s1_path = Path(tmp) / "s1.pt"
            e71_path = Path(tmp) / "e71.pt"
            torch.save({"model": policy.teacher.state_dict(), "config": {}}, s1_path)
            torch.save(
                {
                    "stage": "e7_1_adjacent_pair_retrieval",
                    "epoch": 1,
                    "latent_scale_s_z": FIXED_LATENT_SCALE_S_Z,
                    "reader": policy.reader.state_dict(),
                    "residual_transformer": policy.residual_transformer.state_dict(),
                    "config": {"stage1_ckpt": str(s1_path), "reader_temperature": 1.0},
                },
                e71_path,
            )
            reloaded, _ = load_e7_adjacent_policy(e71_path, device="cpu")
            policy.eval()
            z0 = torch.randn(B, ACTION_HORIZON, ACTION_DIM_10D)
            o1 = policy.predict_action_from_behavior_memory(h_mem, b_mem, pm, q_rgb, init_noise=z0)
            o2 = reloaded.predict_action_from_behavior_memory(h_mem, b_mem, pm, q_rgb, init_noise=z0)
            torch.testing.assert_close(o1["action_10d"], o2["action_10d"], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
