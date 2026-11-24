"""Stage 2 Cross/Self feature & cached `z_self` bank for E6 Latent Behavior Transfer.

Features:
- Reuses `e5/outputs/cache/` CLIP features & 10D actions + `e6/outputs/stage2/z_self_cache/`
  (`demo_memories.npy`, `z_self_train.npy`, `z_self_val.npy`).
- Strict train/val fold isolation (zero trajectory leakage).
- Random Cross Demo resampling every epoch:
  - 97.5% Cross-pair (`demo != query`, strictly preferring different layout when available)
  - 2.5% Self-pair (`demo == query`, identity anchor)
- Deterministic validation modes:
  - `cross_a` & `cross_b`: two distinct valid val Cross Demos for every val Query (for
    evaluating Cross No-Delta, Cross + Delta, AND multi-demo consistency!)
  - `self`: Self-pair (`demo == query`)
  - `wrong_demo`: different-task val Demo (checks Residual does not bypass Demo!)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import ActionNormalizer

from . import ACTION_HORIZON, E5_CACHE_DIR, OUTPUT_ROOT

DEFAULT_STAGE2_CACHE_DIR = OUTPUT_ROOT / "stage2/z_self_cache"


class Stage2TransferBank:
    """Zero-I/O GPU-resident (or CPU-pinned) bank combining cached `demo_memories`,
    cached `z_self`, deduplicated `query_features`, and `actions_10d`."""

    def __init__(
        self,
        *,
        clip_cache_dir: Path = E5_CACHE_DIR,
        z_self_cache_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
        device: torch.device = torch.device("cpu"),
        resident_on_device: bool = True,
        z_self_on_device: bool = True,
        self_ratio: float = 0.025,
        seed: int = 42,
    ):
        self.clip_cache_dir = Path(clip_cache_dir)
        self.z_self_cache_dir = Path(z_self_cache_dir)
        self.device = device
        self.storage_device = device if resident_on_device else torch.device("cpu")
        self.z_self_device = (
            device if (resident_on_device and z_self_on_device) else torch.device("cpu")
        )
        self.self_ratio = float(self_ratio)
        self.seed = int(seed)

        self.clip_meta = json.loads((self.clip_cache_dir / "cache_meta.json").read_text())
        self.z_self_meta = json.loads((self.z_self_cache_dir / "z_self_meta.json").read_text())
        self.episodes = json.loads((self.clip_cache_dir / "episodes.json").read_text())
        self.num_episodes = len(self.episodes)
        self.latent_scale_s_z = float(self.z_self_meta["train_z_self_rms"])

        # Load frozen Demo memories (M_D + prompt_pos_emb) and offsets/lengths
        demo_mem_np = np.load(self.z_self_cache_dir / "demo_memories.npy", allow_pickle=False)
        demo_offsets_np = np.load(self.clip_cache_dir / "demo_offsets.npy", allow_pickle=False)
        demo_lengths_np = np.load(self.clip_cache_dir / "demo_lengths.npy", allow_pickle=False)

        # Load query CLIP features & 10D actions from E5 cache
        query_feats_np = np.load(self.clip_cache_dir / "query_features.npy", allow_pickle=False)
        query_ep_offsets_np = np.load(
            self.clip_cache_dir / "query_ep_offsets.npy", allow_pickle=False
        )
        query_frame_nums_np = np.load(
            self.clip_cache_dir / "query_frame_nums.npy", allow_pickle=False
        )
        actions_10d_np = np.load(self.clip_cache_dir / "actions_10d.npy", allow_pickle=False)
        action_offsets_np = np.load(
            self.clip_cache_dir / "action_offsets.npy", allow_pickle=False
        )

        # Load pre-cached z_self arrays (aligned 1-to-1 with valid starts)
        z_self_train_np = np.load(
            self.z_self_cache_dir / "z_self_train.npy", allow_pickle=False
        )
        z_self_val_np = np.load(self.z_self_cache_dir / "z_self_val.npy", allow_pickle=False)

        norm_state = torch.load(
            self.clip_cache_dir / "action_normalizer.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.action_normalizer = ActionNormalizer(
            scale=norm_state["scale"], offset=norm_state["offset"]
        )

        self.demo_offsets_np = demo_offsets_np
        self.demo_lengths_np = demo_lengths_np
        self.query_ep_offsets_np = query_ep_offsets_np
        self.query_frame_nums_np = query_frame_nums_np
        self.action_offsets_np = action_offsets_np
        self.ep_lengths_np = np.array([e["frames"] for e in self.episodes], dtype=np.int32)

        # Collect identical valid starts ordering as Stage 1 & z_self_cache
        (
            self.train_episodes,
            self.train_frames,
            self.train_q_rows,
        ) = self._collect_valid_starts("train")
        (
            self.val_episodes,
            self.val_frames,
            self.val_q_rows,
        ) = self._collect_valid_starts("val")

        if len(z_self_train_np) != len(self.train_episodes):
            raise ValueError(
                f"z_self_train count {len(z_self_train_np)} != train starts {len(self.train_episodes)}"
            )
        if len(z_self_val_np) != len(self.val_episodes):
            raise ValueError(
                f"z_self_val count {len(z_self_val_np)} != val starts {len(self.val_episodes)}"
            )

        # Move bulk tensors to storage_device / z_self_device
        self.demo_memories = torch.from_numpy(demo_mem_np).to(self.storage_device)
        self.demo_offsets = torch.from_numpy(demo_offsets_np).to(self.storage_device)
        self.demo_lengths = torch.from_numpy(demo_lengths_np.astype(np.int64)).to(
            self.storage_device
        )
        self.query_features = torch.from_numpy(query_feats_np).to(self.storage_device)
        self.actions_10d = torch.from_numpy(actions_10d_np).to(self.storage_device)
        self.action_offsets = torch.from_numpy(action_offsets_np).to(self.storage_device)
        self.ep_lengths = torch.from_numpy(self.ep_lengths_np.astype(np.int64)).to(
            self.storage_device
        )
        self.z_self_train = torch.from_numpy(z_self_train_np).to(self.z_self_device)
        self.z_self_val = torch.from_numpy(z_self_val_np).to(self.z_self_device)

        # Build strict same-fold, same-(task, procedure) cross donors preferring different layouts
        self.train_cross_donors = self._build_cross_donors("train")
        self.val_cross_donors = self._build_cross_donors("val")

        # Pre-group train sample indices by query episode for fast vectorized epoch resampling
        self._train_indices_by_ep = self._group_sample_indices(self.train_episodes)

        # Deterministic validation donor assignments:
        # - val_cross_a: primary cross-layout val Demo (only for val samples with >=1 cross donor)
        # - val_cross_b: second distinct cross-layout val Demo (for multi-demo consistency check)
        # - val_wrong_demos: different-task val Demo
        self.val_valid_cross_mask = np.array(
            [int(ep) in self.val_cross_donors for ep in self.val_episodes], dtype=bool
        )
        self.val_cross_a, self.val_cross_b, self.val_has_two_demos = (
            self._build_val_cross_assignments()
        )
        self.val_wrong_demos = self._build_wrong_demos(self.val_episodes)

    def _collect_valid_starts(
        self, fold: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ep_off = self.query_ep_offsets_np
        f_nums = self.query_frame_nums_np
        ep_list: List[np.ndarray] = []
        frame_list: List[np.ndarray] = []
        r0_list: List[np.ndarray] = []
        r1_list: List[np.ndarray] = []

        for ep_idx, row in enumerate(self.episodes):
            if row["fold"] != fold:
                continue
            lo, hi = int(ep_off[ep_idx]), int(ep_off[ep_idx + 1])
            ep_frames = f_nums[lo:hi]
            if len(ep_frames) == 0:
                continue
            prev_f = np.maximum(ep_frames - 1, 0)
            p0 = np.searchsorted(ep_frames, prev_f)
            valid = (p0 < len(ep_frames)) & (
                ep_frames[np.minimum(p0, len(ep_frames) - 1)] == prev_f
            )
            if not np.any(valid):
                continue
            v_idx = np.flatnonzero(valid)
            cnt = len(v_idx)
            ep_list.append(np.full(cnt, ep_idx, dtype=np.int64))
            frame_list.append(ep_frames[v_idx].astype(np.int64))
            r0_list.append((lo + p0[v_idx]).astype(np.int64))
            r1_list.append((lo + v_idx).astype(np.int64))

        episodes = np.concatenate(ep_list, axis=0)
        frames = np.concatenate(frame_list, axis=0)
        q_rows = np.stack(
            [np.concatenate(r0_list, axis=0), np.concatenate(r1_list, axis=0)], axis=1
        )
        return episodes, frames, q_rows

    def _build_cross_donors(self, fold: str) -> Dict[int, np.ndarray]:
        """For each episode `q` in `fold`, return same-(task, procedure, fold) donor episodes
        strictly excluding `q`, prioritizing episodes with a DIFFERENT `layout` whenever any
        different-layout donor exists."""
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i, e in enumerate(self.episodes):
            if e["fold"] == fold:
                groups[(e["task"], e["procedure"])].append(i)

        donors: Dict[int, np.ndarray] = {}
        for ids in groups.values():
            if len(ids) < 2:
                continue
            for q in ids:
                q_layout = self.episodes[q].get("layout")
                diff_layout = [
                    d for d in ids if d != q and self.episodes[d].get("layout") != q_layout
                ]
                same_group = diff_layout if diff_layout else [d for d in ids if d != q]
                donors[q] = np.asarray(same_group, dtype=np.int64)
        return donors

    @staticmethod
    def _group_sample_indices(episodes_arr: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        order = np.argsort(episodes_arr, kind="mergesort")
        sorted_eps = episodes_arr[order]
        uniq, starts = np.unique(sorted_eps, return_index=True)
        ends = np.append(starts[1:], len(sorted_eps))
        return [
            (int(ep), order[int(s) : int(e)]) for ep, s, e in zip(uniq, starts, ends)
        ]

    def _build_val_cross_assignments(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 77777)
        n = len(self.val_episodes)
        cross_a = np.empty(n, dtype=np.int64)
        cross_b = np.empty(n, dtype=np.int64)
        has_two = np.zeros(n, dtype=bool)
        for i, q in enumerate(self.val_episodes):
            q_int = int(q)
            cands = self.val_cross_donors.get(q_int)
            if cands is None or len(cands) == 0:
                cross_a[i] = q_int
                cross_b[i] = q_int
            elif len(cands) == 1:
                cross_a[i] = int(cands[0])
                cross_b[i] = int(cands[0])
            else:
                pick = rng.choice(len(cands), size=2, replace=False)
                cross_a[i] = int(cands[pick[0]])
                cross_b[i] = int(cands[pick[1]])
                has_two[i] = True
        return cross_a, cross_b, has_two

    def _build_wrong_demos(self, val_episodes: np.ndarray) -> np.ndarray:
        val_eps_by_task: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.episodes):
            if e["fold"] == "val":
                val_eps_by_task[e["task"]].append(i)
        tasks = sorted(val_eps_by_task.keys())
        task_to_idx = {t: i for i, t in enumerate(tasks)}
        rng = np.random.default_rng(self.seed + 99991)
        wrong = np.empty(len(val_episodes), dtype=np.int64)
        for idx, q in enumerate(val_episodes):
            q_int = int(q)
            q_task = self.episodes[q_int]["task"]
            t_i = task_to_idx[q_task]
            other_t = tasks[(t_i + 1 + int(rng.integers(len(tasks) - 1))) % len(tasks)]
            cands = val_eps_by_task[other_t]
            wrong[idx] = cands[int(rng.integers(len(cands)))]
        return wrong

    def num_train_samples(self) -> int:
        return len(self.train_episodes)

    def num_val_samples(self) -> int:
        return len(self.val_episodes)

    def build_train_epoch_schedule(
        self, epoch: int, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build deterministic epoch schedule with random Cross Demo resampling:
        - `(1 - self_ratio)` (97.5%) Cross-pair (`demo != query`, different layout preferred)
        - `self_ratio` (2.5%) Self-pair (`demo == query`, identity anchor)
        Returns `(sample_indices, demo_episodes, is_self)` shuffled for `epoch`.
        """
        rng = np.random.default_rng([self.seed, int(epoch), 6262])
        n = len(self.train_episodes)
        demos = self.train_episodes.copy()
        has_cross = np.zeros(n, dtype=bool)

        # Resample a fresh random Cross Demo for every sample whose episode has cross donors
        for ep, idxs in self._train_indices_by_ep:
            cands = self.train_cross_donors.get(ep)
            if cands is not None and len(cands) > 0:
                demos[idxs] = cands[rng.integers(len(cands), size=len(idxs))]
                has_cross[idxs] = True

        # Assign exact self_ratio (2.5%) of samples to Self-pair (identity anchor)
        n_self_target = int(round(n * self.self_ratio))
        is_self = ~has_cross
        already_self = int(is_self.sum())
        need_more_self = max(0, n_self_target - already_self)
        if need_more_self > 0:
            cross_pool = np.flatnonzero(has_cross)
            chosen_self = rng.choice(
                cross_pool, size=min(need_more_self, len(cross_pool)), replace=False
            )
            is_self[chosen_self] = True
            demos[chosen_self] = self.train_episodes[chosen_self]

        order = rng.permutation(n)
        if max_samples is not None:
            order = order[: min(int(max_samples), n)]
        return order, demos[order], is_self[order]

    def gather_batch(
        self,
        *,
        split: str,
        sample_indices: np.ndarray | torch.Tensor,
        demo_episodes: np.ndarray | torch.Tensor,
        is_self: np.ndarray | torch.Tensor,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather `(demo_memory, pad_mask, query_rgb, z_self, actions_10d, is_self)` with zero I/O."""
        dev = self.storage_device
        idx_np = np.asarray(sample_indices, dtype=np.int64)
        if split == "train":
            queries_np = self.train_episodes[idx_np]
            frames_np = self.train_frames[idx_np]
            q_rows_np = self.train_q_rows[idx_np]
            z_self_bank = self.z_self_train
        elif split == "val":
            queries_np = self.val_episodes[idx_np]
            frames_np = self.val_frames[idx_np]
            q_rows_np = self.val_q_rows[idx_np]
            z_self_bank = self.z_self_val
        else:
            raise ValueError(f"Unknown split: {split}")

        demos_t = torch.as_tensor(demo_episodes, dtype=torch.long, device=dev)
        queries_t = torch.as_tensor(queries_np, dtype=torch.long, device=dev)
        frames_t = torch.as_tensor(frames_np, dtype=torch.long, device=dev)
        q_rows_t = torch.as_tensor(q_rows_np, dtype=torch.long, device=dev)
        is_self_t = torch.as_tensor(is_self, dtype=torch.bool, device=dev)

        # 1. Gather ragged frozen Demo memory tokens -> (B, max_P, 768) + pad_mask (B, max_P)
        lengths = self.demo_lengths[demos_t]
        max_p = int(lengths.max().item())
        starts = self.demo_offsets[demos_t]
        pos = torch.arange(max_p, device=dev, dtype=torch.long)[None, :]
        pad_mask = pos >= lengths[:, None]
        valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
        flat_idx = starts[:, None] + valid_pos
        demo_mem = self.demo_memories[flat_idx].to(dtype=out_dtype)
        demo_mem = demo_mem.masked_fill(pad_mask[:, :, None], 0.0)

        # 2. Gather deduplicated Query CLIP features -> (B, 2, 2, 768)
        query_rgb = self.query_features[q_rows_t].to(dtype=out_dtype)

        # 3. Gather pre-cached `z_self` -> (B, 4, 768)
        idx_z = torch.as_tensor(idx_np, dtype=torch.long, device=self.z_self_device)
        z_self = z_self_bank[idx_z].to(device=self.device, dtype=out_dtype, non_blocking=True)

        # 4. Gather 16-step 10D GT action chunk -> (B, 16, 10)
        ep_len = self.ep_lengths[queries_t]
        act_off = self.action_offsets[queries_t]
        step_offsets = torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :]
        rel_steps = torch.minimum(frames_t[:, None] + step_offsets, (ep_len - 1)[:, None])
        actions = self.actions_10d[act_off[:, None] + rel_steps]

        if dev != self.device:
            demo_mem = demo_mem.to(self.device, non_blocking=True)
            pad_mask = pad_mask.to(self.device, non_blocking=True)
            query_rgb = query_rgb.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            is_self_t = is_self_t.to(self.device, non_blocking=True)

        return demo_mem, pad_mask, query_rgb, z_self, actions, is_self_t

    def iter_train_batches(
        self,
        epoch: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        sample_idx, demos, is_self = self.build_train_epoch_schedule(
            epoch, max_samples=max_samples
        )
        sample_idx = sample_idx[rank::world_size]
        demos = demos[rank::world_size]
        is_self = is_self[rank::world_size]
        for start in range(0, len(sample_idx), batch_size):
            end = min(start + batch_size, len(sample_idx))
            yield self.gather_batch(
                split="train",
                sample_indices=sample_idx[start:end],
                demo_episodes=demos[start:end],
                is_self=is_self[start:end],
            )

    def iter_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        require_cross: bool = True,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Iterate validation split under modes:
        - 'cross_a': primary cross-layout val Demo
        - 'cross_b': second distinct cross-layout val Demo (for multi-demo consistency)
        - 'self': Self-pair (`demo == query`)
        - 'wrong_demo': different-task val Demo
        """
        if mode == "cross_b":
            pool = np.flatnonzero(self.val_has_two_demos)
        elif require_cross:
            pool = np.flatnonzero(self.val_valid_cross_mask)
        else:
            pool = np.arange(len(self.val_episodes), dtype=np.int64)

        n = len(pool) if max_samples is None else min(int(max_samples), len(pool))
        sel = pool[:n][rank::world_size]

        if mode == "cross_a":
            demos = self.val_cross_a[sel]
            is_self = np.zeros(len(sel), dtype=bool)
        elif mode == "cross_b":
            demos = self.val_cross_b[sel]
            is_self = np.zeros(len(sel), dtype=bool)
        elif mode == "self":
            demos = self.val_episodes[sel]
            is_self = np.ones(len(sel), dtype=bool)
        elif mode == "wrong_demo":
            demos = self.val_wrong_demos[sel]
            is_self = np.zeros(len(sel), dtype=bool)
        else:
            raise ValueError(f"Unknown val mode: {mode}")

        for start in range(0, len(sel), batch_size):
            end = min(start + batch_size, len(sel))
            yield self.gather_batch(
                split="val",
                sample_indices=sel[start:end],
                demo_episodes=demos[start:end],
                is_self=is_self[start:end],
            )
