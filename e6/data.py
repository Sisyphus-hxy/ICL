"""Cache-only GPU-resident 100% Self-pair data loader for E6 Stage 1.

Directly reuses the offline frozen CLIP feature cache produced by E5 (`e5/outputs/cache/`)
without requiring external manifest files or re-extracting visual features.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import ActionNormalizer

from . import ACTION_HORIZON, E5_CACHE_DIR


class SelfPairFeatureBank:
    """Holds cached Demo 1Hz CLIP features, deduplicated Query CLIP features, and 10D
    actions in memory (or GPU VRAM) for zero-I/O 100% Self-pair training & validation."""

    def __init__(
        self,
        cache_dir: Path = E5_CACHE_DIR,
        *,
        device: torch.device = torch.device("cpu"),
        resident_on_device: bool = True,
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.storage_device = device if resident_on_device else torch.device("cpu")
        self.seed = int(seed)

        self.meta = json.loads((self.cache_dir / "cache_meta.json").read_text())
        self.episodes = json.loads((self.cache_dir / "episodes.json").read_text())
        self.num_episodes = len(self.episodes)

        demo_feats_np = np.load(self.cache_dir / "demo_features.npy", allow_pickle=False)
        demo_offsets_np = np.load(self.cache_dir / "demo_offsets.npy", allow_pickle=False)
        demo_lengths_np = np.load(self.cache_dir / "demo_lengths.npy", allow_pickle=False)
        demo_frame_idx_np = np.load(self.cache_dir / "demo_frame_indices.npy", allow_pickle=False)

        query_feats_np = np.load(self.cache_dir / "query_features.npy", allow_pickle=False)
        query_ep_offsets_np = np.load(self.cache_dir / "query_ep_offsets.npy", allow_pickle=False)
        query_frame_nums_np = np.load(self.cache_dir / "query_frame_nums.npy", allow_pickle=False)

        actions_10d_np = np.load(self.cache_dir / "actions_10d.npy", allow_pickle=False)
        action_offsets_np = np.load(self.cache_dir / "action_offsets.npy", allow_pickle=False)

        norm_state = torch.load(
            self.cache_dir / "action_normalizer.pt", map_location="cpu", weights_only=True
        )
        self.action_normalizer = ActionNormalizer(
            scale=norm_state["scale"], offset=norm_state["offset"]
        )

        # Keep index arrays on CPU numpy for fast deterministic scheduling.
        self.demo_offsets_np = demo_offsets_np
        self.demo_lengths_np = demo_lengths_np
        self.demo_frame_indices_np = demo_frame_idx_np
        self.query_ep_offsets_np = query_ep_offsets_np
        self.query_frame_nums_np = query_frame_nums_np
        self.action_offsets_np = action_offsets_np
        self.ep_lengths_np = np.array([e["frames"] for e in self.episodes], dtype=np.int32)
        self.max_demo_len = int(demo_lengths_np.max())

        # Move bulk tensors onto storage_device (GPU VRAM when resident_on_device=True).
        self.demo_features = torch.from_numpy(demo_feats_np).to(self.storage_device)
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

        # Collect all valid cached (ep_idx, t) starts where both max(0, t-1) and t exist.
        (
            self.train_episodes,
            self.train_frames,
            self.train_q_rows,
        ) = self._collect_valid_starts(fold="train")
        (
            self.val_episodes,
            self.val_frames,
            self.val_q_rows,
        ) = self._collect_valid_starts(fold="val")

        # Deterministic wrong-task val Demo for read-only bottleneck sensitivity diagnostics.
        self.val_wrong_demos = self._build_wrong_demos(self.val_episodes)

        # Deterministic same-episode distinct-timestep pairs on val for temporal distinctiveness check.
        self.val_timestep_pairs = self._build_same_episode_timestep_pairs(
            self.val_episodes, self.val_frames, min_frame_gap=20
        )

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

        if not ep_list:
            empty1 = np.empty(0, dtype=np.int64)
            return empty1, empty1, np.empty((0, 2), dtype=np.int64)

        episodes = np.concatenate(ep_list, axis=0)
        frames = np.concatenate(frame_list, axis=0)
        q_rows = np.stack(
            [np.concatenate(r0_list, axis=0), np.concatenate(r1_list, axis=0)], axis=1
        )
        return episodes, frames, q_rows

    def _build_wrong_demos(self, val_episodes: np.ndarray) -> np.ndarray:
        """Assign each val sample a deterministic val Demo from a different task."""
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
            if len(tasks) > 1:
                t_i = task_to_idx[q_task]
                other_t = tasks[(t_i + 1 + int(rng.integers(len(tasks) - 1))) % len(tasks)]
                cands = val_eps_by_task[other_t]
                wrong[idx] = cands[int(rng.integers(len(cands)))]
            else:
                cands = [i for i in val_eps_by_task[q_task] if i != q_int]
                wrong[idx] = cands[0] if cands else q_int
        return wrong

    @staticmethod
    def _build_same_episode_timestep_pairs(
        val_episodes: np.ndarray,
        val_frames: np.ndarray,
        min_frame_gap: int = 20,
    ) -> np.ndarray:
        """Pair distinct cached query timesteps (idx_a, idx_b) belonging to the SAME val
        episode with |t_a - t_b| >= min_frame_gap (falls back to max-gap pair per episode
        if gap < min_frame_gap). Used to test whether different query timesteps on the same
        Demo produce distinguishable latents and action predictions."""
        by_ep: Dict[int, List[int]] = defaultdict(list)
        for idx, ep in enumerate(val_episodes):
            by_ep[int(ep)].append(idx)

        pairs: List[Tuple[int, int]] = []
        for ep, indices in sorted(by_ep.items()):
            if len(indices) < 2:
                continue
            # Sort indices by frame number and take well-separated pairs.
            indices_sorted = sorted(indices, key=lambda i: int(val_frames[i]))
            f_first = int(val_frames[indices_sorted[0]])
            f_last = int(val_frames[indices_sorted[-1]])
            if f_last - f_first >= min_frame_gap:
                # Pair earliest with latest, and if >=4 frames also pair inner quartiles.
                pairs.append((indices_sorted[0], indices_sorted[-1]))
                if len(indices_sorted) >= 4:
                    ia = indices_sorted[1]
                    ib = indices_sorted[-2]
                    if int(val_frames[ib]) - int(val_frames[ia]) >= min_frame_gap:
                        pairs.append((ia, ib))
            elif f_last > f_first:
                pairs.append((indices_sorted[0], indices_sorted[-1]))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def num_train_samples(self) -> int:
        return len(self.train_episodes)

    def num_val_samples(self) -> int:
        return len(self.val_episodes)

    def build_train_epoch_schedule(
        self, epoch: int, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deterministic 100% Self-pair shuffled schedule for `epoch` (1-indexed).
        Returns (demos, queries, frames, q_rows) with `demos == queries` identically."""
        rng = np.random.default_rng([self.seed, int(epoch), 6060])
        n = len(self.train_episodes)
        order = rng.permutation(n)
        if max_samples is not None:
            order = order[: min(int(max_samples), n)]
        eps = self.train_episodes[order]
        frames = self.train_frames[order]
        q_rows = self.train_q_rows[order]
        # 100% Self-pair: demo_idx == query_idx
        return eps, eps, frames, q_rows

    def gather_batch(
        self,
        demos: np.ndarray | torch.Tensor,
        queries: np.ndarray | torch.Tensor,
        frames: np.ndarray | torch.Tensor,
        q_rows: np.ndarray | torch.Tensor,
        *,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather a padded batch of Demo 1Hz RGB features, Query RGB features, and
        16-step 10D GT action chunks with zero disk I/O."""
        dev = self.storage_device
        demos_t = torch.as_tensor(demos, dtype=torch.long, device=dev)
        queries_t = torch.as_tensor(queries, dtype=torch.long, device=dev)
        frames_t = torch.as_tensor(frames, dtype=torch.long, device=dev)
        q_rows_t = torch.as_tensor(q_rows, dtype=torch.long, device=dev)

        # 1. Ragged Demo 1Hz features -> padded (B, max_P, 2, 768) + bool pad mask (B, max_P)
        lengths = self.demo_lengths[demos_t]
        max_p = int(lengths.max().item())
        starts = self.demo_offsets[demos_t]
        pos = torch.arange(max_p, device=dev, dtype=torch.long)[None, :]
        pad_mask = pos >= lengths[:, None]
        valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
        flat_idx = starts[:, None] + valid_pos
        demo_rgb = self.demo_features[flat_idx].to(dtype=out_dtype)
        demo_rgb = demo_rgb.masked_fill(pad_mask[:, :, None, None], 0.0)

        # 2. Deduplicated Query history features [max(0, t-1), t] -> (B, 2, 2, 768)
        query_rgb = self.query_features[q_rows_t].to(dtype=out_dtype)

        # 3. 16-step 10D GT action chunk with BPP repeat_last padding -> (B, 16, 10)
        ep_len = self.ep_lengths[queries_t]
        act_off = self.action_offsets[queries_t]
        step_offsets = torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :]
        rel_steps = torch.minimum(frames_t[:, None] + step_offsets, (ep_len - 1)[:, None])
        actions = self.actions_10d[act_off[:, None] + rel_steps]

        if dev != self.device:
            demo_rgb = demo_rgb.to(self.device, non_blocking=True)
            pad_mask = pad_mask.to(self.device, non_blocking=True)
            query_rgb = query_rgb.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)

        return demo_rgb, pad_mask, query_rgb, actions

    def iter_train_batches(
        self,
        epoch: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        demos, queries, frames, q_rows = self.build_train_epoch_schedule(
            epoch, max_samples=max_samples
        )
        demos = demos[rank::world_size]
        queries = queries[rank::world_size]
        frames = frames[rank::world_size]
        q_rows = q_rows[rank::world_size]
        for start in range(0, len(demos), batch_size):
            end = min(start + batch_size, len(demos))
            yield self.gather_batch(
                demos[start:end],
                queries[start:end],
                frames[start:end],
                q_rows[start:end],
            )

    def iter_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate validation split:
        - 'self': 100% Self-pair (demo_idx == query_idx)
        - 'wrong_demo': read-only diagnostic (different-task val Demo for same Query)
        """
        n = (
            len(self.val_episodes)
            if max_samples is None
            else min(int(max_samples), len(self.val_episodes))
        )
        idx = np.arange(n)[rank::world_size]
        queries = self.val_episodes[idx]
        frames = self.val_frames[idx]
        q_rows = self.val_q_rows[idx]

        if mode == "self":
            demos = queries
        elif mode == "wrong_demo":
            demos = self.val_wrong_demos[idx]
        else:
            raise ValueError(f"Unknown val mode: {mode} (expected 'self' or 'wrong_demo')")

        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            yield self.gather_batch(
                demos[start:end],
                queries[start:end],
                frames[start:end],
                q_rows[start:end],
            )
