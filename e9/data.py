"""GPU-resident local video clip data loader for E9 Stage 1 (CLIP).

Strictly reuses the exact E6 Stage 1 data split, cached train/val start indices
(1,373,035 train starts / 6,248 val starts), deterministic epoch shuffle schedule,
and 16-step 10D action windows with repeat-last padding, while gathering dense
16-frame dual-camera local clips `V_{t:t+16}` and exact tail-window padding masks
`(t + s >= T)` from `e9/outputs/dense_clip_cache/`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import ActionNormalizer
from e6.data import SelfPairFeatureBank

from . import (
    ACTION_HORIZON,
    DEFAULT_DENSE_CACHE_DIR,
    E5_CACHE_DIR,
    LOCAL_WINDOW_LEN,
)


class LocalClipFeatureBank:
    """Zero-I/O GPU-resident loader yielding `(clip_rgb, clip_pad_mask, actions_10d)`.

    - `clip_rgb`: `(B, 16, 2, 768)` dual-camera frozen CLIP features for `V_{t:t+16}`
      with padded out-of-bounds tail frames zeroed out.
    - `clip_pad_mask`: `(B, 16)` bool mask (`True` where `t + s >= T`, `False` where valid).
    - `actions_10d`: `(B, 16, 10)` GT action chunks with BPP repeat-last padding (exact E6 match).
    """

    def __init__(
        self,
        e5_cache_dir: Path = E5_CACHE_DIR,
        dense_cache_dir: Path = DEFAULT_DENSE_CACHE_DIR,
        *,
        device: torch.device = torch.device("cpu"),
        resident_on_device: bool = True,
        seed: int = 42,
    ):
        self.e5_cache_dir = Path(e5_cache_dir)
        self.dense_cache_dir = Path(dense_cache_dir)
        self.device = device
        self.storage_device = device if resident_on_device else torch.device("cpu")
        self.seed = int(seed)

        self.e5_meta = json.loads((self.e5_cache_dir / "cache_meta.json").read_text())
        self.dense_meta = json.loads(
            (self.dense_cache_dir / "dense_cache_meta.json").read_text()
        )
        self.episodes = json.loads((self.e5_cache_dir / "episodes.json").read_text())
        self.num_episodes = len(self.episodes)

        # Load dense 20Hz dual-camera CLIP features (N_total, 2, 768) float16
        frame_feats_np = np.load(
            self.dense_cache_dir / "frame_features.npy", allow_pickle=False
        )
        frame_offsets_np = np.load(
            self.dense_cache_dir / "frame_offsets.npy", allow_pickle=False
        )
        ep_lengths_np = np.load(
            self.dense_cache_dir / "ep_lengths.npy", allow_pickle=False
        )

        # Load exact E5/E6 10D actions & ActionNormalizer
        actions_10d_np = np.load(
            self.e5_cache_dir / "actions_10d.npy", allow_pickle=False
        )
        action_offsets_np = np.load(
            self.e5_cache_dir / "action_offsets.npy", allow_pickle=False
        )
        if not np.array_equal(frame_offsets_np, action_offsets_np):
            raise ValueError(
                "Dense cache frame_offsets do not match E5 action_offsets!"
            )

        norm_state = torch.load(
            self.e5_cache_dir / "action_normalizer.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.action_normalizer = ActionNormalizer(
            scale=norm_state["scale"], offset=norm_state["offset"]
        )

        # Reconstruct the EXACT E6 Stage 1 train/val start indices from E5 query_frame_nums
        self.query_ep_offsets_np = np.load(
            self.e5_cache_dir / "query_ep_offsets.npy", allow_pickle=False
        )
        self.query_frame_nums_np = np.load(
            self.e5_cache_dir / "query_frame_nums.npy", allow_pickle=False
        )
        self.frame_offsets_np = frame_offsets_np
        self.ep_lengths_np = ep_lengths_np.astype(np.int32)

        self.train_episodes, self.train_frames = self._collect_e6_valid_starts("train")
        self.val_episodes, self.val_frames = self._collect_e6_valid_starts("val")

        # Deterministic wrong-task val clips (for read-only sensitivity diagnostic)
        self.val_wrong_episodes, self.val_wrong_frames = self._build_wrong_clips(
            self.val_episodes, self.val_frames
        )

        # Exact same same-episode distinct-timestep val pairs as E6 Stage 1
        self.val_timestep_pairs = SelfPairFeatureBank._build_same_episode_timestep_pairs(
            self.val_episodes, self.val_frames, min_frame_gap=20
        )

        # Move bulk tensors onto storage_device (GPU VRAM when resident_on_device=True)
        self.frame_features = torch.from_numpy(frame_feats_np).to(self.storage_device)
        self.frame_offsets = torch.from_numpy(frame_offsets_np).to(self.storage_device)
        self.actions_10d = torch.from_numpy(actions_10d_np).to(self.storage_device)
        self.ep_lengths = torch.from_numpy(self.ep_lengths_np.astype(np.int64)).to(
            self.storage_device
        )

    def _collect_e6_valid_starts(self, fold: str) -> Tuple[np.ndarray, np.ndarray]:
        """Exact replica of `SelfPairFeatureBank._collect_valid_starts` so E9 uses
        the identical 1,373,035 train starts and 6,248 val starts as E6 Stage 1."""
        ep_off = self.query_ep_offsets_np
        f_nums = self.query_frame_nums_np
        ep_list: List[np.ndarray] = []
        frame_list: List[np.ndarray] = []

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
            ep_list.append(np.full(len(v_idx), ep_idx, dtype=np.int64))
            frame_list.append(ep_frames[v_idx].astype(np.int64))

        if not ep_list:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        return np.concatenate(ep_list, axis=0), np.concatenate(frame_list, axis=0)

    def _build_wrong_clips(
        self, val_episodes: np.ndarray, val_frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign each val sample a deterministic clip from a different-task val episode."""
        val_eps_by_task: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.episodes):
            if e["fold"] == "val":
                val_eps_by_task[e["task"]].append(i)
        tasks = sorted(val_eps_by_task.keys())
        task_to_idx = {t: i for i, t in enumerate(tasks)}
        rng = np.random.default_rng(self.seed + 99991)
        wrong_ep = np.empty(len(val_episodes), dtype=np.int64)
        wrong_fr = np.empty(len(val_episodes), dtype=np.int64)
        for idx, (ep, fr) in enumerate(zip(val_episodes, val_frames)):
            ep_int = int(ep)
            task = self.episodes[ep_int]["task"]
            if len(tasks) > 1:
                t_i = task_to_idx[task]
                other_t = tasks[
                    (t_i + 1 + int(rng.integers(len(tasks) - 1))) % len(tasks)
                ]
                cands = val_eps_by_task[other_t]
                w_ep = cands[int(rng.integers(len(cands)))]
            else:
                cands = [i for i in val_eps_by_task[task] if i != ep_int]
                w_ep = cands[0] if cands else ep_int
            wrong_ep[idx] = w_ep
            w_len = int(self.ep_lengths_np[w_ep])
            wrong_fr[idx] = min(int(fr), max(w_len - 1, 0))
        return wrong_ep, wrong_fr

    def num_train_samples(self) -> int:
        return len(self.train_episodes)

    def num_val_samples(self) -> int:
        return len(self.val_episodes)

    def build_train_epoch_schedule(
        self, epoch: int, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact E6 Stage 1 deterministic epoch shuffle RNG `[seed, epoch, 6060]`."""
        rng = np.random.default_rng([self.seed, int(epoch), 6060])
        n = len(self.train_episodes)
        order = rng.permutation(n)
        if max_samples is not None:
            order = order[: min(int(max_samples), n)]
        return self.train_episodes[order], self.train_frames[order]

    def gather_batch(
        self,
        clip_episodes: np.ndarray | torch.Tensor,
        clip_frames: np.ndarray | torch.Tensor,
        action_episodes: Optional[np.ndarray | torch.Tensor] = None,
        action_frames: Optional[np.ndarray | torch.Tensor] = None,
        *,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather local 16-frame clips `V_{t:t+16}`, tail padding masks, and 16-step 10D actions.

        When `action_episodes` / `action_frames` are None, actions come from the same
        `(clip_episodes, clip_frames)` (self-reconstruction). Passing different clip vs
        action coordinates supports the read-only wrong-clip sensitivity check.
        """
        dev = self.storage_device
        c_ep = torch.as_tensor(clip_episodes, dtype=torch.long, device=dev)
        c_fr = torch.as_tensor(clip_frames, dtype=torch.long, device=dev)

        # 1. Local video clip V_{t:t+16} -> (B, 16, 2, 768) + tail pad mask (B, 16)
        c_len = self.ep_lengths[c_ep]  # (B,)
        c_off = self.frame_offsets[c_ep]  # (B,)
        win_offsets = torch.arange(LOCAL_WINDOW_LEN, device=dev, dtype=torch.long)[None, :]
        raw_steps = c_fr[:, None] + win_offsets  # (B, 16)
        clip_pad_mask = raw_steps >= c_len[:, None]  # True where out-of-bounds tail
        clamped_steps = torch.minimum(raw_steps, (c_len - 1).clamp_min(0)[:, None])
        clip_rgb = self.frame_features[c_off[:, None] + clamped_steps].to(dtype=out_dtype)
        clip_rgb = clip_rgb.masked_fill(clip_pad_mask[:, :, None, None], 0.0)

        # 2. 16-step 10D GT action chunk with exact E6 BPP repeat-last padding -> (B, 16, 10)
        if action_episodes is None:
            a_ep = c_ep
            a_fr = c_fr
            a_len = c_len
            a_off = c_off
            a_rel = clamped_steps if LOCAL_WINDOW_LEN == ACTION_HORIZON else torch.minimum(
                a_fr[:, None]
                + torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :],
                (a_len - 1).clamp_min(0)[:, None],
            )
        else:
            a_ep = torch.as_tensor(action_episodes, dtype=torch.long, device=dev)
            a_fr = torch.as_tensor(action_frames, dtype=torch.long, device=dev)
            a_len = self.ep_lengths[a_ep]
            a_off = self.frame_offsets[a_ep]
            act_offsets = torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :]
            a_rel = torch.minimum(a_fr[:, None] + act_offsets, (a_len - 1).clamp_min(0)[:, None])

        actions = self.actions_10d[a_off[:, None] + a_rel]

        if dev != self.device:
            clip_rgb = clip_rgb.to(self.device, non_blocking=True)
            clip_pad_mask = clip_pad_mask.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)

        return clip_rgb, clip_pad_mask, actions

    def iter_train_batches(
        self,
        epoch: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        eps, frames = self.build_train_epoch_schedule(epoch, max_samples=max_samples)
        eps = eps[rank::world_size]
        frames = frames[rank::world_size]
        for start in range(0, len(eps), batch_size):
            end = min(start + batch_size, len(eps))
            yield self.gather_batch(eps[start:end], frames[start:end])

    def iter_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate validation split:
        - 'self': encode `V_{t:t+16}` of the sample and reconstruct its own `a_{t:t+16}`
        - 'wrong_clip': read-only sensitivity diagnostic (encode different-task val clip)
        """
        n = (
            len(self.val_episodes)
            if max_samples is None
            else min(int(max_samples), len(self.val_episodes))
        )
        idx = np.arange(n)[rank::world_size]
        act_eps = self.val_episodes[idx]
        act_frames = self.val_frames[idx]

        if mode == "self":
            clip_eps, clip_frames = act_eps, act_frames
        elif mode == "wrong_clip":
            clip_eps = self.val_wrong_episodes[idx]
            clip_frames = self.val_wrong_frames[idx]
        else:
            raise ValueError(
                f"Unknown val mode: {mode} (expected 'self' or 'wrong_clip')"
            )

        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            yield self.gather_batch(
                clip_eps[start:end],
                clip_frames[start:end],
                action_episodes=act_eps[start:end],
                action_frames=act_frames[start:end],
            )
