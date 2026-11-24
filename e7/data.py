"""E7 Data Bank: strictly reuses E6 Stage 2.1's exact train/val splits, valid starts,
epoch Cross/Self pairing schedule (`seed=42`, `97.5%` Cross / `2.5%` Self), and
deterministic validation assignments (`cross_a`, `cross_b`, `self`, `wrong_demo`),
while serving padded Behavior Memory `(h_mem, b_mem, mem_pad_mask)` with zero disk I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from e6.stage2_data import DEFAULT_STAGE2_CACHE_DIR, Stage2TransferBank

from . import ACTION_HORIZON, DEFAULT_BEHAVIOR_MEM_DIR, E5_CACHE_DIR


class E7TransferBank(Stage2TransferBank):
    """Subclass of `Stage2TransferBank` that inherits bit-for-bit identical splits,
    epoch schedules, and validation donor assignments, and additionally loads & gathers
    ragged stride-16 Behavior Memory `(h_mem, b_mem, mem_pad_mask)`."""

    def __init__(
        self,
        *,
        clip_cache_dir: Path = E5_CACHE_DIR,
        z_self_cache_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
        behavior_mem_dir: Path = DEFAULT_BEHAVIOR_MEM_DIR,
        device: torch.device = torch.device("cpu"),
        resident_on_device: bool = True,
        z_self_on_device: bool = True,
        self_ratio: float = 0.025,
        seed: int = 42,
    ):
        super().__init__(
            clip_cache_dir=clip_cache_dir,
            z_self_cache_dir=z_self_cache_dir,
            device=device,
            resident_on_device=resident_on_device,
            z_self_on_device=z_self_on_device,
            self_ratio=self_ratio,
            seed=seed,
        )
        self.behavior_mem_dir = Path(behavior_mem_dir)
        self.behavior_mem_meta = json.loads(
            (self.behavior_mem_dir / "behavior_memory_meta.json").read_text()
        )

        mem_h_np = np.load(self.behavior_mem_dir / "mem_h.npy", allow_pickle=False)
        mem_b_np = np.load(self.behavior_mem_dir / "mem_b.npy", allow_pickle=False)
        mem_offsets_np = np.load(
            self.behavior_mem_dir / "mem_offsets.npy", allow_pickle=False
        )
        mem_lengths_np = np.load(
            self.behavior_mem_dir / "mem_lengths.npy", allow_pickle=False
        )

        if len(mem_lengths_np) != self.num_episodes:
            raise ValueError(
                f"Behavior memory episodes {len(mem_lengths_np)} != {self.num_episodes}"
            )

        self.mem_offsets_np = mem_offsets_np
        self.mem_lengths_np = mem_lengths_np
        self.mem_h = torch.from_numpy(mem_h_np).to(self.storage_device)
        self.mem_b = torch.from_numpy(mem_b_np).to(self.storage_device)
        self.mem_offsets = torch.from_numpy(mem_offsets_np).to(self.storage_device)
        self.mem_lengths = torch.from_numpy(mem_lengths_np.astype(np.int64)).to(
            self.storage_device
        )

    def _gather_ragged_behavior_memory(
        self, ep_ids_t: torch.Tensor, *, out_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather padded `(h_mem, b_mem, mem_pad_mask)` of shapes `(B, M, 4, 768)`,
        `(B, M, 4, 768)`, `(B, M)` where `mem_pad_mask == True` indicates padding."""
        dev = self.storage_device
        lengths = self.mem_lengths[ep_ids_t]
        max_m = int(lengths.max().item())
        starts = self.mem_offsets[ep_ids_t]
        pos = torch.arange(max_m, device=dev, dtype=torch.long)[None, :]
        pad_mask = pos >= lengths[:, None]
        valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
        flat_idx = starts[:, None] + valid_pos

        h_mem = self.mem_h[flat_idx].to(dtype=out_dtype)
        b_mem = self.mem_b[flat_idx].to(dtype=out_dtype)
        h_mem = h_mem.masked_fill(pad_mask[:, :, None, None], 0.0)
        b_mem = b_mem.masked_fill(pad_mask[:, :, None, None], 0.0)
        return h_mem, b_mem, pad_mask

    def gather_e7_batch(
        self,
        *,
        split: str,
        sample_indices: np.ndarray | torch.Tensor,
        demo_episodes: np.ndarray | torch.Tensor,
        is_self: np.ndarray | torch.Tensor,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gather `(h_mem, b_mem, mem_pad_mask, query_rgb, z_self, actions_10d, is_self)`."""
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

        # 1. Behavior Memory `(h_mem, b_mem, mem_pad_mask)` for Demo episodes
        h_mem, b_mem, mem_pad_mask = self._gather_ragged_behavior_memory(
            demos_t, out_dtype=out_dtype
        )

        # 2. Query 2-step CLIP features -> (B, 2, 2, 768)
        query_rgb = self.query_features[q_rows_t].to(dtype=out_dtype)

        # 3. Cached Teacher z_self -> (B, 4, 768)
        idx_z = torch.as_tensor(idx_np, dtype=torch.long, device=self.z_self_device)
        z_self = z_self_bank[idx_z].to(
            device=self.device, dtype=out_dtype, non_blocking=True
        )

        # 4. 16-step 10D GT action chunk with repeat-last tail padding -> (B, 16, 10)
        ep_len = self.ep_lengths[queries_t]
        act_off = self.action_offsets[queries_t]
        step_offsets = torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :]
        rel_steps = torch.minimum(frames_t[:, None] + step_offsets, (ep_len - 1)[:, None])
        actions = self.actions_10d[act_off[:, None] + rel_steps]

        if dev != self.device:
            h_mem = h_mem.to(self.device, non_blocking=True)
            b_mem = b_mem.to(self.device, non_blocking=True)
            mem_pad_mask = mem_pad_mask.to(self.device, non_blocking=True)
            query_rgb = query_rgb.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            is_self_t = is_self_t.to(self.device, non_blocking=True)

        return h_mem, b_mem, mem_pad_mask, query_rgb, z_self, actions, is_self_t

    def iter_e7_train_batches(
        self,
        epoch: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> Iterator[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:
        sample_idx, demos, is_self = self.build_train_epoch_schedule(
            epoch, max_samples=max_samples
        )
        sample_idx = sample_idx[rank::world_size]
        demos = demos[rank::world_size]
        is_self = is_self[rank::world_size]
        for start in range(0, len(sample_idx), batch_size):
            end = min(start + batch_size, len(sample_idx))
            yield self.gather_e7_batch(
                split="train",
                sample_indices=sample_idx[start:end],
                demo_episodes=demos[start:end],
                is_self=is_self[start:end],
            )

    def iter_e7_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        require_cross: bool = True,
    ) -> Iterator[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:
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
            yield self.gather_e7_batch(
                split="val",
                sample_indices=sel[start:end],
                demo_episodes=demos[start:end],
                is_self=is_self[start:end],
            )
