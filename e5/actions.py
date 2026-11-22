"""BPP 10D action representation, normalization, and 7D OSC control interface conversion."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn

from . import ACTION_DIM_10D, ACTION_DIM_7D, EXEC_HORIZON
from .bpp_compat import import_bpp_components

_BPP = import_bpp_components()
_ROT_TF = _BPP["RotationTransformer"](from_rep="axis_angle", to_rep="rotation_6d")


def action_7d_to_10d(
    actions_7d: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Convert 7D LIBERO delta OSC actions [dx, dy, dz, ax, ay, az, gripper]
    into BPP's 10D format [dx, dy, dz, rot6d(6), gripper]."""
    if isinstance(actions_7d, np.ndarray):
        if actions_7d.shape[-1] != ACTION_DIM_7D:
            raise ValueError(f"Expected last dim 7, got {actions_7d.shape}")
        arr = actions_7d.astype(np.float32, copy=False)
        pos = arr[..., :3]
        rot6d = _ROT_TF.forward(arr[..., 3:6]).astype(np.float32, copy=False)
        grip = arr[..., 6:7]
        return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32, copy=False)

    if actions_7d.shape[-1] != ACTION_DIM_7D:
        raise ValueError(f"Expected last dim 7, got {tuple(actions_7d.shape)}")
    with torch.autocast(actions_7d.device.type, enabled=False):
        t = actions_7d.float()
        pos = t[..., :3]
        rot6d = _ROT_TF.forward(t[..., 3:6])
        grip = t[..., 6:7]
        return torch.cat([pos, rot6d, grip], dim=-1)


def action_10d_to_7d(
    actions_10d: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Convert BPP's 10D action representation [dx, dy, dz, rot6d(6), gripper]
    back to LIBERO's 7D delta OSC control format [dx, dy, dz, ax, ay, az, gripper]."""
    if isinstance(actions_10d, np.ndarray):
        if actions_10d.shape[-1] != ACTION_DIM_10D:
            raise ValueError(f"Expected last dim 10, got {actions_10d.shape}")
        arr = actions_10d.astype(np.float32, copy=False)
        pos = arr[..., :3]
        rotvec = _ROT_TF.inverse(arr[..., 3:9]).astype(np.float32, copy=False)
        grip = arr[..., 9:10]
        return np.concatenate([pos, rotvec, grip], axis=-1).astype(np.float32, copy=False)

    if actions_10d.shape[-1] != ACTION_DIM_10D:
        raise ValueError(f"Expected last dim 10, got {tuple(actions_10d.shape)}")
    with torch.autocast(actions_10d.device.type, enabled=False):
        t = actions_10d.float()
        pos = t[..., :3]
        rotvec = _ROT_TF.inverse(t[..., 3:9])
        grip = t[..., 9:10]
        return torch.cat([pos, rotvec, grip], dim=-1)


def extract_exec_actions(
    actions_10d: Union[np.ndarray, torch.Tensor],
    exec_horizon: int = EXEC_HORIZON,
) -> Union[np.ndarray, torch.Tensor]:
    """Slice the first `exec_horizon` (default 12 of 16) predicted steps and convert to 7D OSC."""
    if actions_10d.ndim < 2 or actions_10d.shape[-2] < exec_horizon:
        raise ValueError(
            f"Expected at least {exec_horizon} steps in dim -2, got shape {tuple(actions_10d.shape)}"
        )
    return action_10d_to_7d(actions_10d[..., :exec_horizon, :])


class ActionNormalizer(nn.Module):
    """Exact replica of BPP's `robomimic_abs_action_only_normalizer_from_stat`:
    maps delta position [:3] linearly to [-1, 1], and leaves rot6d [3:9] and gripper [9]
    at identity (scale=1, offset=0)."""

    def __init__(
        self,
        scale: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ):
        super().__init__()
        if scale is None:
            scale = torch.ones(ACTION_DIM_10D, dtype=torch.float32)
        if offset is None:
            offset = torch.zeros(ACTION_DIM_10D, dtype=torch.float32)
        self.register_buffer("scale", scale.detach().clone().float().view(ACTION_DIM_10D))
        self.register_buffer("offset", offset.detach().clone().float().view(ACTION_DIM_10D))

    @classmethod
    def fit_from_train_actions_10d(
        cls,
        actions_10d: np.ndarray,
        range_eps: float = 1e-7,
    ) -> "ActionNormalizer":
        if actions_10d.shape[-1] != ACTION_DIM_10D:
            raise ValueError(f"Expected 10D actions, got {actions_10d.shape}")
        flat = actions_10d.reshape(-1, ACTION_DIM_10D)
        pos_min = flat[:, :3].min(axis=0).astype(np.float32)
        pos_max = flat[:, :3].max(axis=0).astype(np.float32)
        pos_range = pos_max - pos_min
        ignore = pos_range < range_eps
        pos_range = np.where(ignore, 2.0, pos_range)
        pos_scale = 2.0 / pos_range
        pos_offset = -1.0 - pos_scale * pos_min
        pos_offset = np.where(ignore, -pos_min, pos_offset)

        scale = np.ones(ACTION_DIM_10D, dtype=np.float32)
        offset = np.zeros(ACTION_DIM_10D, dtype=np.float32)
        scale[:3] = pos_scale
        offset[:3] = pos_offset
        return cls(torch.from_numpy(scale), torch.from_numpy(offset))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(device=x.device, dtype=x.dtype) + self.offset.to(
            device=x.device, dtype=x.dtype
        )

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset.to(device=x.device, dtype=x.dtype)) / self.scale.to(
            device=x.device, dtype=x.dtype
        )
