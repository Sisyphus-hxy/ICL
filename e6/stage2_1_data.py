"""Stage 2.1 Transfer Bank: reuses exact Stage 2 cached features, `z_self`, and Cross/Self
pairing (97.5% Cross / 2.5% Self), additionally gathering the Query's own demo memory
`(M_Q, query_pad_mask)` for the Self Anchor Loss `L_anchor = MSE((R_S(M_Q, h_q) - z_s) / s_z)`.

Zero modifications to datasets, cached files, or Stage 1 / Stage 2 data splits.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from . import ACTION_HORIZON
from .stage2_data import DEFAULT_STAGE2_CACHE_DIR, Stage2TransferBank


class Stage2_1TransferBank(Stage2TransferBank):
    """Subclass of `Stage2TransferBank` that also returns `(query_mem, query_pad_mask)`
    from the exact same frozen `demo_memories` bank."""

    def _gather_ragged_memories(
        self, ep_ids_t: torch.Tensor, *, out_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.storage_device
        lengths = self.demo_lengths[ep_ids_t]
        max_p = int(lengths.max().item())
        starts = self.demo_offsets[ep_ids_t]
        pos = torch.arange(max_p, device=dev, dtype=torch.long)[None, :]
        pad_mask = pos >= lengths[:, None]
        valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
        flat_idx = starts[:, None] + valid_pos
        mem = self.demo_memories[flat_idx].to(dtype=out_dtype)
        mem = mem.masked_fill(pad_mask[:, :, None], 0.0)
        return mem, pad_mask

    def gather_batch_with_query_memory(
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
        torch.Tensor,
    ]:
        """Gather `(demo_mem, demo_pad_mask, query_mem, query_pad_mask, query_rgb,
        z_self, actions_10d, is_self)` with zero disk I/O."""
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

        # 1. Demo memory M_D + pad mask
        demo_mem, demo_pad_mask = self._gather_ragged_memories(demos_t, out_dtype=out_dtype)

        # 2. Query's own demo memory M_Q + pad mask (for Self Anchor Loss L_anchor)
        query_mem, query_pad_mask = self._gather_ragged_memories(queries_t, out_dtype=out_dtype)

        # 3. Query CLIP features -> (B, 2, 2, 768)
        query_rgb = self.query_features[q_rows_t].to(dtype=out_dtype)

        # 4. Pre-cached Teacher z_self -> (B, 4, 768)
        idx_z = torch.as_tensor(idx_np, dtype=torch.long, device=self.z_self_device)
        z_self = z_self_bank[idx_z].to(device=self.device, dtype=out_dtype, non_blocking=True)

        # 5. 16-step 10D GT action chunk -> (B, 16, 10)
        ep_len = self.ep_lengths[queries_t]
        act_off = self.action_offsets[queries_t]
        step_offsets = torch.arange(ACTION_HORIZON, device=dev, dtype=torch.long)[None, :]
        rel_steps = torch.minimum(frames_t[:, None] + step_offsets, (ep_len - 1)[:, None])
        actions = self.actions_10d[act_off[:, None] + rel_steps]

        if dev != self.device:
            demo_mem = demo_mem.to(self.device, non_blocking=True)
            demo_pad_mask = demo_pad_mask.to(self.device, non_blocking=True)
            query_mem = query_mem.to(self.device, non_blocking=True)
            query_pad_mask = query_pad_mask.to(self.device, non_blocking=True)
            query_rgb = query_rgb.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            is_self_t = is_self_t.to(self.device, non_blocking=True)

        return (
            demo_mem,
            demo_pad_mask,
            query_mem,
            query_pad_mask,
            query_rgb,
            z_self,
            actions,
            is_self_t,
        )

    def iter_train_batches_joint(
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
            yield self.gather_batch_with_query_memory(
                split="train",
                sample_indices=sample_idx[start:end],
                demo_episodes=demos[start:end],
                is_self=is_self[start:end],
            )

    def iter_val_batches_joint(
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
            yield self.gather_batch_with_query_memory(
                split="val",
                sample_indices=sel[start:end],
                demo_episodes=demos[start:end],
                is_self=is_self[start:end],
            )


__all__ = ["DEFAULT_STAGE2_CACHE_DIR", "Stage2_1TransferBank"]
