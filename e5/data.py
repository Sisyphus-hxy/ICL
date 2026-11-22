"""In-memory / GPU-resident dataset and deterministic 50/50 Self/Cross-pair scheduler for E5.

Zero per-sample disk I/O during training and validation: loads the compact (~6.4 GB)
cached Demo/Query CLIP feature arrays and 10D actions once into memory (or directly onto
each GPU rank) and indexes every epoch cleanly from the existing 80-epoch unique query
starts manifest.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from . import ACTION_HORIZON, E4_OUTPUTS, EMBED_DIM, MANIFEST_80_DIR
from .actions import ActionNormalizer
from .cache import DEFAULT_CACHE_DIR, load_episodes_and_manifest


class E5FeatureBank:
    """Holds all cached Demo 1Hz features, deduplicated Query features, and 10D actions
    in memory (or on GPU) for zero-I/O training and validation."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        device: torch.device = torch.device("cpu"),
        resident_on_device: bool = True,
        manifest_dir: Path = MANIFEST_80_DIR,
        pairs_root: Path = E4_OUTPUTS,
        self_ratio: float = 0.5,
        seed: int = 42,
        schedule_mode: str = "manifest_unique",
        num_epochs: int = 80,
    ):
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.storage_device = device if resident_on_device else torch.device("cpu")
        self.self_ratio = float(self_ratio)
        self.seed = int(seed)
        self.schedule_mode = schedule_mode
        self.num_epochs = int(num_epochs)

        self.meta = json.loads((self.cache_dir / "cache_meta.json").read_text())
        self.episodes = json.loads((self.cache_dir / "episodes.json").read_text())
        self.num_episodes = len(self.episodes)

        # Load cached numpy arrays into RAM then optionally pin/move to GPU VRAM (~6.4 GB).
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

        # Keep index helpers on CPU numpy for instant lookup resolution.
        self.demo_offsets_np = demo_offsets_np
        self.demo_lengths_np = demo_lengths_np
        self.demo_frame_indices_np = demo_frame_idx_np
        self.query_ep_offsets_np = query_ep_offsets_np
        self.query_frame_nums_np = query_frame_nums_np
        self.action_offsets_np = action_offsets_np
        self.ep_lengths_np = np.array([e["frames"] for e in self.episodes], dtype=np.int32)
        self.max_demo_len = int(demo_lengths_np.max())

        # Move bulk tensors to storage_device (GPU VRAM if resident_on_device else CPU RAM).
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

        # Load existing 80-epoch unique query starts manifest + val split.
        _, self.manifest, epoch_keys, val_keys = load_episodes_and_manifest(
            episodes_path=E4_OUTPUTS / "episodes.json",
            manifest_dir=manifest_dir,
            verify_manifest_sha=False,
        )
        if self.meta.get("max_tasks") is not None:
            # Smoke cache subset: remap keys to cached episode subset.
            full_eps = json.loads((E4_OUTPUTS / "episodes.json").read_text())
            id_to_new = {e["id"]: i for i, e in enumerate(self.episodes)}
            old_to_new = {
                old_i: id_to_new[e["id"]]
                for old_i, e in enumerate(full_eps)
                if e["id"] in id_to_new
            }

            def _filter(arr: np.ndarray) -> np.ndarray:
                rows = [
                    [old_to_new[int(d)], old_to_new[int(q)], int(f)]
                    for d, q, f in arr
                    if int(d) in old_to_new and int(q) in old_to_new
                ]
                return np.asarray(rows, dtype=np.int32).reshape(-1, 3)

            epoch_keys = [_filter(k) for k in epoch_keys]
            val_keys = _filter(val_keys)

        self.raw_epoch_keys = epoch_keys
        self.num_manifest_epochs = len(epoch_keys)
        self.val_keys = val_keys

        # Build cross-layout donor lookup per query episode (for replacing the ~2.38%
        # manifest self-pairs when a sample is assigned to the Cross-pair half).
        self.cross_donors_by_query = self._build_cross_donors(pairs_root)
        self.all_donors_by_query = {
            q: np.append(ds, np.int32(q)) for q, ds in self.cross_donors_by_query.items()
        }

        # Build deterministic wrong-task Demo index for each validation sample.
        self.val_wrong_demos = self._build_wrong_demos(val_keys)

        # Pre-resolve (query_idx, start_frame) -> (row_prev, row_curr) in query_features for val!
        self.val_q_rows = self._resolve_query_rows(val_keys[:, 1], val_keys[:, 2])

        # Optional full-pass pool of all valid cached train (query, frame) starts in VRAM.
        self.all_train_queries: Optional[np.ndarray] = None
        self.all_train_frames: Optional[np.ndarray] = None
        self.all_train_q_rows: Optional[np.ndarray] = None
        self.train_group_Rows: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.schedule_mode == "all_cached_starts":
            self._init_all_cached_train_starts()

    def _build_cross_donors(self, pairs_root: Path) -> Dict[int, np.ndarray]:
        cross_by_q: Dict[int, List[int]] = defaultdict(list)
        if self.meta.get("max_tasks") is None:
            tp = np.load(pairs_root / "train_pairs.npy", allow_pickle=False)
            cross = tp[tp[:, 0] != tp[:, 1]]
            for d, q in cross:
                cross_by_q[int(q)].append(int(d))
        else:
            # Group by (task, procedure, fold=='train') within smoke subset
            groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
            for i, e in enumerate(self.episodes):
                if e["fold"] == "train":
                    groups[(e["task"], e["procedure"])].append(i)
            for ids in groups.values():
                if len(ids) > 1:
                    for q in ids:
                        cross_by_q[q] = [d for d in ids if d != q]
        return {q: np.asarray(ds, dtype=np.int32) for q, ds in cross_by_q.items()}

    def _build_wrong_demos(self, val_keys: np.ndarray) -> np.ndarray:
        """Assign each validation sample a deterministic val Demo from a DIFFERENT task."""
        val_eps_by_task: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.episodes):
            if e["fold"] == "val":
                val_eps_by_task[e["task"]].append(i)
        tasks = sorted(val_eps_by_task.keys())
        task_to_idx = {t: i for i, t in enumerate(tasks)}
        rng = np.random.default_rng(self.seed + 99991)
        wrong = np.empty(len(val_keys), dtype=np.int32)
        for idx, (_, q, _) in enumerate(val_keys):
            q_task = self.episodes[int(q)]["task"]
            if len(tasks) > 1:
                t_i = task_to_idx[q_task]
                other_t = tasks[(t_i + 1 + int(rng.integers(len(tasks) - 1))) % len(tasks)]
                cands = val_eps_by_task[other_t]
                wrong[idx] = cands[int(rng.integers(len(cands)))]
            else:
                # Fallback if single-task smoke subset
                cands = [i for i in val_eps_by_task[q_task] if i != int(q)]
                wrong[idx] = cands[0] if cands else int(q)
        return wrong

    def _resolve_query_rows(self, queries: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """Map (query_idx, start_frame) arrays -> (N, 2) row indices in `query_features`
        for [max(0, frame - 1), frame]."""
        n = len(queries)
        rows = np.empty((n, 2), dtype=np.int64)
        ep_off = self.query_ep_offsets_np
        f_nums = self.query_frame_nums_np
        prev_f = np.maximum(frames - 1, 0)
        for i in range(n):
            q = int(queries[i])
            lo = int(ep_off[q])
            hi = int(ep_off[q + 1])
            ep_frames = f_nums[lo:hi]
            p0 = int(np.searchsorted(ep_frames, prev_f[i]))
            p1 = int(np.searchsorted(ep_frames, frames[i]))
            if (
                p0 >= len(ep_frames)
                or p1 >= len(ep_frames)
                or ep_frames[p0] != prev_f[i]
                or ep_frames[p1] != frames[i]
            ):
                raise KeyError(
                    f"Missing cached query frame for ep={q}, frames=({prev_f[i]}, {frames[i]})"
                )
            rows[i, 0] = lo + p0
            rows[i, 1] = lo + p1
        return rows

    def _init_all_cached_train_starts(self) -> None:
        """Collect all valid cached train (query_ep, start_frame) pairs that have both
        `max(0, start_frame - 1)` and `start_frame` in `query_features`, and group by
        (task, procedure) for O(1) vectorized uniform Demo pairing every pass."""
        ep_off = self.query_ep_offsets_np
        f_nums = self.query_frame_nums_np
        q_list: List[np.ndarray] = []
        f_list: List[np.ndarray] = []
        r0_list: List[np.ndarray] = []
        r1_list: List[np.ndarray] = []

        group_eps: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        ep_to_group: Dict[int, Tuple[str, str]] = {}
        for i, e in enumerate(self.episodes):
            if e["fold"] == "train":
                g = (e["task"], e["procedure"])
                group_eps[g].append(i)
                ep_to_group[i] = g

        group_row_chunks: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
        cur_offset = 0
        for q in sorted(ep_to_group.keys()):
            lo, hi = int(ep_off[q]), int(ep_off[q + 1])
            ep_frames = f_nums[lo:hi]
            if len(ep_frames) == 0:
                continue
            prev_f = np.maximum(ep_frames - 1, 0)
            p0 = np.searchsorted(ep_frames, prev_f)
            valid = (p0 < len(ep_frames)) & (ep_frames[np.minimum(p0, len(ep_frames) - 1)] == prev_f)
            if not np.any(valid):
                continue
            v_idx = np.flatnonzero(valid)
            cnt = len(v_idx)
            q_list.append(np.full(cnt, q, dtype=np.int64))
            f_list.append(ep_frames[v_idx].astype(np.int64))
            r0_list.append((lo + p0[v_idx]).astype(np.int64))
            r1_list.append((lo + v_idx).astype(np.int64))
            group_row_chunks[ep_to_group[q]].append((cur_offset, cur_offset + cnt))
            cur_offset += cnt

        self.all_train_queries = np.concatenate(q_list, axis=0)
        self.all_train_frames = np.concatenate(f_list, axis=0)
        self.all_train_q_rows = np.stack(
            [np.concatenate(r0_list, axis=0), np.concatenate(r1_list, axis=0)], axis=1
        )
        self.train_group_Rows = []
        for g, chunks in group_row_chunks.items():
            rows = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in chunks], axis=0)
            demos = np.asarray(group_eps[g], dtype=np.int64)
            self.train_group_Rows.append((rows, demos))

    def samples_per_epoch(self) -> int:
        if self.schedule_mode == "all_cached_starts":
            assert self.all_train_queries is not None
            return len(self.all_train_queries)
        return len(self.raw_epoch_keys[0])

    def build_train_epoch_schedule(
        self, epoch: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build deterministic training schedule for `epoch` (1-indexed).

        Modes:
        - `manifest_unique`: uses `self.raw_epoch_keys[(epoch - 1) % 80]`.
        - `all_cached_starts`: uses all 1,373,035 valid cached train (query, frame) starts
          in VRAM, re-sampling Demo pairings every pass.
        When `self.self_ratio < 0`: samples Demo uniformly from all same-task/procedure
        train demos (including `query` itself, so self-pairs occur naturally at ~2.22%).
        """
        rng = np.random.default_rng([self.seed, epoch, 5050])

        if self.schedule_mode == "all_cached_starts":
            assert (
                self.all_train_queries is not None
                and self.all_train_frames is not None
                and self.all_train_q_rows is not None
            )
            queries = self.all_train_queries.copy()
            frames = self.all_train_frames.copy()
            q_rows = self.all_train_q_rows.copy()
            n = len(queries)
            demos = np.empty(n, dtype=np.int64)
            for rows, cand_demos in self.train_group_Rows:
                demos[rows] = cand_demos[rng.integers(len(cand_demos), size=len(rows))]
            if self.self_ratio < 0:
                is_self = demos == queries
            else:
                n_self_target = int(round(n * self.self_ratio))
                perm = rng.permutation(n)
                is_self = np.zeros(n, dtype=bool)
                is_self[perm[:n_self_target]] = True
                demos[is_self] = queries[is_self]
                cross_conflicts = np.flatnonzero((~is_self) & (demos == queries))
                for idx in cross_conflicts:
                    q = int(queries[idx])
                    cands = self.cross_donors_by_query.get(q)
                    if cands is not None and len(cands) > 0:
                        demos[idx] = int(cands[int(rng.integers(len(cands)))])
                    else:
                        is_self[idx] = True
            order = rng.permutation(n)
            return demos[order], queries[order], frames[order], q_rows[order], is_self[order]

        base = self.raw_epoch_keys[(epoch - 1) % self.num_manifest_epochs]
        n = len(base)
        demos = base[:, 0].astype(np.int64, copy=True)
        queries = base[:, 1].astype(np.int64, copy=True)
        frames = base[:, 2].astype(np.int64, copy=True)

        if self.self_ratio < 0:
            # Natural uniform sampling across all same-task train demos (~2.22% self)
            for idx in range(n):
                q = int(queries[idx])
                cands = self.all_donors_by_query.get(q)
                if cands is not None and len(cands) > 0:
                    demos[idx] = int(cands[int(rng.integers(len(cands)))])
                else:
                    demos[idx] = q
            is_self = demos == queries
        else:
            # Assign exact self_ratio (50%) of rows to self-pair, remaining 50% to cross-pair.
            n_self_target = int(round(n * self.self_ratio))
            perm = rng.permutation(n)
            is_self = np.zeros(n, dtype=bool)
            is_self[perm[:n_self_target]] = True

            # Apply self-pairs (demo = query) and ensure cross-pairs have demo != query.
            demos[is_self] = queries[is_self]
            cross_self_conflicts = np.flatnonzero((~is_self) & (demos == queries))
            for idx in cross_self_conflicts:
                q = int(queries[idx])
                cands = self.cross_donors_by_query.get(q)
                if cands is not None and len(cands) > 0:
                    demos[idx] = int(cands[int(rng.integers(len(cands)))])
                else:
                    # 11 episodes with 0 cross partners stay self-pair honestly.
                    is_self[idx] = True

        # Shuffle in-memory order for this epoch.
        order = rng.permutation(n)
        demos = demos[order]
        queries = queries[order]
        frames = frames[order]
        is_self = is_self[order]
        q_rows = self._resolve_query_rows(queries, frames)
        return demos, queries, frames, q_rows, is_self

    def gather_batch(
        self,
        demos: np.ndarray | torch.Tensor,
        queries: np.ndarray | torch.Tensor,
        frames: np.ndarray | torch.Tensor,
        q_rows: np.ndarray | torch.Tensor,
        is_self: np.ndarray | torch.Tensor,
        *,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather a padded batch of Demo 1Hz features, Query features, and 16-step 10D actions
        with zero disk I/O."""
        dev = self.storage_device
        demos_t = torch.as_tensor(demos, dtype=torch.long, device=dev)
        queries_t = torch.as_tensor(queries, dtype=torch.long, device=dev)
        frames_t = torch.as_tensor(frames, dtype=torch.long, device=dev)
        q_rows_t = torch.as_tensor(q_rows, dtype=torch.long, device=dev)
        is_self_t = torch.as_tensor(is_self, dtype=torch.bool, device=dev)

        B = demos_t.shape[0]
        # 1. Gather ragged Demo 1Hz features -> padded (B, max_P, 2, 768) + bool pad mask (B, max_P)
        lengths = self.demo_lengths[demos_t]  # (B,)
        max_p = int(lengths.max().item())
        starts = self.demo_offsets[demos_t]  # (B,)
        pos = torch.arange(max_p, device=dev, dtype=torch.long)[None, :]  # (1, max_p)
        pad_mask = pos >= lengths[:, None]  # (B, max_p), True = pad
        valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
        flat_idx = starts[:, None] + valid_pos  # (B, max_p)
        demo_rgb = self.demo_features[flat_idx].to(dtype=out_dtype)  # (B, max_p, 2, 768)
        demo_rgb = demo_rgb.masked_fill(pad_mask[:, :, None, None], 0.0)

        # 2. Gather deduplicated Query history features -> (B, 2, 2, 768)
        query_rgb = self.query_features[q_rows_t].to(dtype=out_dtype)

        # 3. Gather 16-step 10D action chunk with BPP repeat_last padding -> (B, 16, 10)
        ep_len = self.ep_lengths[queries_t]  # (B,)
        act_off = self.action_offsets[queries_t]  # (B,)
        step_offsets = torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :]
        rel_steps = torch.minimum(frames_t[:, None] + step_offsets, (ep_len - 1)[:, None])
        actions = self.actions_10d[act_off[:, None] + rel_steps]  # (B, 16, 10) fp32

        if dev != self.device:
            demo_rgb = demo_rgb.to(self.device, non_blocking=True)
            pad_mask = pad_mask.to(self.device, non_blocking=True)
            query_rgb = query_rgb.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            is_self_t = is_self_t.to(self.device, non_blocking=True)

        return demo_rgb, pad_mask, query_rgb, actions, is_self_t

    def iter_train_batches(
        self,
        epoch: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        demos, queries, frames, q_rows, is_self = self.build_train_epoch_schedule(epoch)
        demos = demos[rank::world_size]
        queries = queries[rank::world_size]
        frames = frames[rank::world_size]
        q_rows = q_rows[rank::world_size]
        is_self = is_self[rank::world_size]
        for start in range(0, len(demos), batch_size):
            end = min(start + batch_size, len(demos))
            yield self.gather_batch(
                demos[start:end],
                queries[start:end],
                frames[start:end],
                q_rows[start:end],
                is_self[start:end],
            )

    def iter_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate validation split in one of three modes:
        - 'cross': official cross-layout val pairs (6,180)
        - 'self': same query windows with demo_idx = query_idx
        - 'wrong_demo': same query windows with a deterministic wrong-task val Demo
        """
        n = len(self.val_keys) if max_samples is None else min(max_samples, len(self.val_keys))
        idx = np.arange(n)[rank::world_size]
        queries = self.val_keys[idx, 1].astype(np.int64)
        frames = self.val_keys[idx, 2].astype(np.int64)
        q_rows = self.val_q_rows[idx]

        if mode == "cross":
            demos = self.val_keys[idx, 0].astype(np.int64)
            is_self = np.zeros(len(idx), dtype=bool)
        elif mode == "self":
            demos = queries.copy()
            is_self = np.ones(len(idx), dtype=bool)
        elif mode == "wrong_demo":
            demos = self.val_wrong_demos[idx].astype(np.int64)
            is_self = np.zeros(len(idx), dtype=bool)
        else:
            raise ValueError(f"Unknown val mode: {mode}")

        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            yield self.gather_batch(
                demos[start:end],
                queries[start:end],
                frames[start:end],
                q_rows[start:end],
                is_self[start:end],
            )
