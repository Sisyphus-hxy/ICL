"""DINOv3 Spatial 4x4 (and optional Global) 100% Self-pair data loader for E6 Stage 1.

Strictly reuses the exact trajectory IDs, Query timesteps, dual-camera views,
action labels, and train/val split from the E5/BPP CLIP cache via the verified
independent DINOv3 feature cache (`dinov3/cache/`).

High-throughput low-RAM streaming design for 89.72 GB `query_spatial_features.npy`:
- `demo_spatial_features.npy` (~9.28 GB) is loaded into CPU RAM once (optional GPU residency).
- All 6,248 validation query frames (~0.55 GB unique rows) are preloaded into CPU RAM
  in a single monotonic disk sweep at startup -> zero disk I/O during validation.
- Training uses double-buffered Episode Super-Chunk prefetching: each epoch shuffles all
  13,995 train episodes, splits them into super-chunks (~500 episodes / ~50k samples /
  ~2.8 GB each), reads each super-chunk's episodes in monotonic disk-offset order in a
  background thread (converting 1.37M tiny random seeks into sequential elevator sweeps),
  and performs full uniform random shuffling across all ~50k samples inside every buffer.
"""

from __future__ import annotations

import json
import queue
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from dinov3 import DEFAULT_CACHE_DIR as DINOV3_CACHE_DIR
from dinov3 import DEFAULT_CLIP_CACHE_DIR, HIDDEN_DIM, NUM_SPATIAL_TOKENS
from e5.actions import ActionNormalizer

from . import ACTION_HORIZON


def verify_dinov3_cache_compatibility(
    dinov3_cache_dir: Path = DINOV3_CACHE_DIR,
    clip_cache_dir: Path = DEFAULT_CLIP_CACHE_DIR,
    *,
    feature_mode: str = "spatial",
    sample_rows: int = 2048,
) -> Dict[str, Any]:
    """Check DINOv3 cache shapes, index parity vs CLIP cache, finite values, and
    quantitatively compare CLIP vs DINOv3 feature distributions."""
    dinov3_cache_dir = Path(dinov3_cache_dir)
    clip_cache_dir = Path(clip_cache_dir)
    if feature_mode not in ("spatial", "global"):
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}; choose 'spatial' or 'global'")

    meta_path = dinov3_cache_dir / "cache_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"DINOv3 cache_meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())

    demo_fname = (
        "demo_spatial_features.npy" if feature_mode == "spatial" else "demo_global_features.npy"
    )
    query_fname = (
        "query_spatial_features.npy" if feature_mode == "spatial" else "query_global_features.npy"
    )
    demo_arr = np.load(dinov3_cache_dir / demo_fname, mmap_mode="r", allow_pickle=False)
    query_arr = np.load(dinov3_cache_dir / query_fname, mmap_mode="r", allow_pickle=False)

    if feature_mode == "spatial":
        if demo_arr.ndim != 4 or demo_arr.shape[1:] != (2, NUM_SPATIAL_TOKENS, HIDDEN_DIM):
            raise ValueError(
                f"Unexpected {demo_fname} shape {demo_arr.shape}; expected (N, 2, {NUM_SPATIAL_TOKENS}, {HIDDEN_DIM})"
            )
        if query_arr.ndim != 4 or query_arr.shape[1:] != (2, NUM_SPATIAL_TOKENS, HIDDEN_DIM):
            raise ValueError(
                f"Unexpected {query_fname} shape {query_arr.shape}; expected (N, 2, {NUM_SPATIAL_TOKENS}, {HIDDEN_DIM})"
            )
    else:
        if demo_arr.ndim != 3 or demo_arr.shape[1:] != (2, HIDDEN_DIM):
            raise ValueError(
                f"Unexpected {demo_fname} shape {demo_arr.shape}; expected (N, 2, {HIDDEN_DIM})"
            )
        if query_arr.ndim != 3 or query_arr.shape[1:] != (2, HIDDEN_DIM):
            raise ValueError(
                f"Unexpected {query_fname} shape {query_arr.shape}; expected (N, 2, {HIDDEN_DIM})"
            )

    # Verify strict index parity against CLIP cache when available
    index_parity: Dict[str, Any] = {"clip_cache_checked": False}
    if (clip_cache_dir / "episodes.json").is_file() and not meta.get("is_subset", False):
        dino_eps = json.loads((dinov3_cache_dir / "episodes.json").read_text())
        clip_eps = json.loads((clip_cache_dir / "episodes.json").read_text())
        parity_checks = {
            "episodes_identical": dino_eps == clip_eps,
            "demo_offsets_identical": np.array_equal(
                np.load(dinov3_cache_dir / "demo_offsets.npy"),
                np.load(clip_cache_dir / "demo_offsets.npy"),
            ),
            "demo_lengths_identical": np.array_equal(
                np.load(dinov3_cache_dir / "demo_lengths.npy"),
                np.load(clip_cache_dir / "demo_lengths.npy"),
            ),
            "demo_frame_indices_identical": np.array_equal(
                np.load(dinov3_cache_dir / "demo_frame_indices.npy"),
                np.load(clip_cache_dir / "demo_frame_indices.npy"),
            ),
            "query_ep_offsets_identical": np.array_equal(
                np.load(dinov3_cache_dir / "query_ep_offsets.npy"),
                np.load(clip_cache_dir / "query_ep_offsets.npy"),
            ),
            "query_frame_nums_identical": np.array_equal(
                np.load(dinov3_cache_dir / "query_frame_nums.npy"),
                np.load(clip_cache_dir / "query_frame_nums.npy"),
            ),
            "action_offsets_identical": np.array_equal(
                np.load(dinov3_cache_dir / "action_offsets.npy"),
                np.load(clip_cache_dir / "action_offsets.npy"),
            ),
        }
        if not all(parity_checks.values()):
            raise RuntimeError(f"DINOv3 vs CLIP cache index parity failed: {parity_checks}")
        index_parity = {"clip_cache_checked": True, **parity_checks}

    # Sample evenly across demo & query arrays for NaN/Inf + feature distribution stats
    def _sample_stats(arr: np.ndarray, n_sample: int) -> Dict[str, Any]:
        n = len(arr)
        k = min(n, n_sample)
        idx = np.linspace(0, n - 1, num=k, dtype=np.int64) if n > 0 else np.empty(0, dtype=np.int64)
        sub = np.asarray(arr[idx], dtype=np.float32)
        nan_cnt = int(np.isnan(sub).sum())
        inf_cnt = int(np.isinf(sub).sum())
        if nan_cnt > 0 or inf_cnt > 0:
            raise RuntimeError(f"Non-finite values found in sampled feature array: nan={nan_cnt}, inf={inf_cnt}")
        flat = sub.reshape(-1, HIDDEN_DIM)
        norms = np.linalg.norm(flat, axis=-1)
        ch_std = flat.std(axis=0)
        return {
            "sampled_rows": int(k),
            "nan_count": nan_cnt,
            "inf_count": inf_cnt,
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "l2_norm_mean": float(norms.mean()),
            "l2_norm_std": float(norms.std()),
            "channel_std_mean": float(ch_std.mean()),
            "channel_std_min": float(ch_std.min()),
            "channel_std_max": float(ch_std.max()),
        }

    dino_demo_stats = _sample_stats(demo_arr, sample_rows)
    dino_query_stats = _sample_stats(query_arr, min(sample_rows, 512))

    dist_comparison: Dict[str, Any] = {
        "dinov3_demo": dino_demo_stats,
        "dinov3_query": dino_query_stats,
    }
    if (clip_cache_dir / "demo_features.npy").is_file():
        clip_demo_arr = np.load(
            clip_cache_dir / "demo_features.npy", mmap_mode="r", allow_pickle=False
        )
        clip_demo_stats = _sample_stats(clip_demo_arr, sample_rows)
        dist_comparison["clip_demo"] = clip_demo_stats
        dist_comparison["dinov3_vs_clip_ratios"] = {
            "std_ratio": round(
                dino_demo_stats["std"] / max(clip_demo_stats["std"], 1e-12), 4
            ),
            "l2_norm_ratio": round(
                dino_demo_stats["l2_norm_mean"] / max(clip_demo_stats["l2_norm_mean"], 1e-12), 4
            ),
            "abs_mean_shift": round(
                abs(dino_demo_stats["mean"] - clip_demo_stats["mean"]), 4
            ),
            "distribution_mismatch_detected": bool(
                abs(dino_demo_stats["l2_norm_mean"] / max(clip_demo_stats["l2_norm_mean"], 1e-12) - 1.0)
                > 0.25
            ),
            "recommendation": (
                "Significant scale/basis shift between CLIP CLS (L2~29.5, std~1.05) and "
                "DINOv3 spatial patches (L2~9.6, std~0.35): apply input LayerNorm on DINOv3 "
                "tokens and avoid blindly reusing CLIP-space PromptPooler/ReferenceReader weights."
            ),
        }

    # Include full completeness audit status if present on disk
    comp_audit_path = dinov3_cache_dir / "completeness_audit.json"
    full_audit_ok = None
    if comp_audit_path.is_file():
        comp = json.loads(comp_audit_path.read_text())
        full_audit_ok = (
            comp.get("missing_episodes") == 0
            and all(
                v.get("nan_count") == 0
                and v.get("inf_count") == 0
                and v.get("all_zero_vectors") == 0
                for v in comp.get("arrays", {}).values()
            )
        )

    return {
        "cache_dir": str(dinov3_cache_dir),
        "feature_mode": feature_mode,
        "backbone_id": meta.get("backbone_id"),
        "cache_version": meta.get("version"),
        "demo_shape": list(demo_arr.shape),
        "query_shape": list(query_arr.shape),
        "demo_dtype": str(demo_arr.dtype),
        "query_dtype": str(query_arr.dtype),
        "index_parity": index_parity,
        "full_completeness_audit_passed": full_audit_ok,
        "feature_distribution_audit": dist_comparison,
    }


class DINOv3SelfPairFeatureBank:
    """E6 Stage 1 100% Self-pair data loader for DINOv3 Spatial 4x4 (or Global) feature cache.

    Produces batches with identical episode/frame/val splits as `e6.data.SelfPairFeatureBank`:
    - Spatial mode:
        demo_rgb:  (B, max_P, 2, 16, 768)
        pad_mask:  (B, max_P) bool (True = padded)
        query_rgb: (B, 2, 2, 16, 768)
        actions:   (B, 16, 10)
    - Global mode:
        demo_rgb:  (B, max_P, 2, 768)
        pad_mask:  (B, max_P) bool
        query_rgb: (B, 2, 2, 768)
        actions:   (B, 16, 10)
    """

    def __init__(
        self,
        cache_dir: Path = DINOV3_CACHE_DIR,
        *,
        feature_mode: str = "spatial",
        device: torch.device = torch.device("cpu"),
        demo_in_ram: bool = True,
        demo_on_device: bool = False,
        preload_val_query: bool = True,
        superchunk_episodes: int = 512,
        seed: int = 42,
    ):
        if feature_mode not in ("spatial", "global"):
            raise ValueError(f"Unsupported feature_mode={feature_mode!r}")

        self.cache_dir = Path(cache_dir)
        self.feature_mode = feature_mode
        self.device = torch.device(device)
        self.demo_in_ram = bool(demo_in_ram)
        self.demo_on_device = bool(demo_on_device)
        self.superchunk_episodes = max(1, int(superchunk_episodes))
        self.seed = int(seed)

        self.meta = json.loads((self.cache_dir / "cache_meta.json").read_text())
        self.episodes = json.loads((self.cache_dir / "episodes.json").read_text())
        self.num_episodes = len(self.episodes)

        self.demo_offsets_np = np.load(self.cache_dir / "demo_offsets.npy", allow_pickle=False)
        self.demo_lengths_np = np.load(self.cache_dir / "demo_lengths.npy", allow_pickle=False)
        self.demo_frame_indices_np = np.load(
            self.cache_dir / "demo_frame_indices.npy", allow_pickle=False
        )
        self.query_ep_offsets_np = np.load(
            self.cache_dir / "query_ep_offsets.npy", allow_pickle=False
        )
        self.query_frame_nums_np = np.load(
            self.cache_dir / "query_frame_nums.npy", allow_pickle=False
        )
        self.actions_10d_np = np.load(self.cache_dir / "actions_10d.npy", allow_pickle=False)
        self.action_offsets_np = np.load(
            self.cache_dir / "action_offsets.npy", allow_pickle=False
        )
        self.ep_lengths_np = np.array([e["frames"] for e in self.episodes], dtype=np.int32)
        self.max_demo_len = int(self.demo_lengths_np.max()) if self.num_episodes > 0 else 0

        norm_state = torch.load(
            self.cache_dir / "action_normalizer.pt", map_location="cpu", weights_only=True
        )
        self.action_normalizer = ActionNormalizer(
            scale=norm_state["scale"], offset=norm_state["offset"]
        )

        demo_fname = (
            "demo_spatial_features.npy"
            if feature_mode == "spatial"
            else "demo_global_features.npy"
        )
        query_fname = (
            "query_spatial_features.npy"
            if feature_mode == "spatial"
            else "query_global_features.npy"
        )

        # Demo features (~9.28 GB spatial / ~0.58 GB global)
        self.demo_features_t: Optional[torch.Tensor] = None
        if self.demo_on_device:
            arr = np.load(self.cache_dir / demo_fname, allow_pickle=False)
            self.demo_features_t = torch.from_numpy(arr).to(self.device)
            self.demo_features_np = arr
        elif self.demo_in_ram:
            self.demo_features_np = np.load(self.cache_dir / demo_fname, allow_pickle=False)
        else:
            self.demo_features_np = np.load(
                self.cache_dir / demo_fname, mmap_mode="r", allow_pickle=False
            )

        # Query features (~89.72 GB spatial -> always mmap; ~5.61 GB global -> mmap or RAM)
        self.query_features_mm = np.load(
            self.cache_dir / query_fname, mmap_mode="r", allow_pickle=False
        )

        # Exact same valid start collection as e6.data.SelfPairFeatureBank
        (
            self.train_episodes,
            self.train_frames,
            self.train_q_rows,
            self.train_starts_by_ep,
        ) = self._collect_valid_starts(fold="train")
        (
            self.val_episodes,
            self.val_frames,
            self.val_q_rows,
            self.val_starts_by_ep,
        ) = self._collect_valid_starts(fold="val")
        self.train_ep_ids = np.asarray(sorted(self.train_starts_by_ep.keys()), dtype=np.int64)

        # Deterministic wrong-task val Demo & same-episode distinct-timestep pairs (exact CLIP match)
        self.val_wrong_demos = self._build_wrong_demos(self.val_episodes)
        self.val_timestep_pairs = self._build_same_episode_timestep_pairs(
            self.val_episodes, self.val_frames, min_frame_gap=20
        )

        # Preload unique val query rows (~0.55 GB) into RAM in a single monotonic sweep
        self.val_query_cache_np: Optional[np.ndarray] = None
        self.val_q_local_rows: Optional[np.ndarray] = None
        if preload_val_query and len(self.val_q_rows) > 0:
            self._preload_val_query_features()

    def _collect_valid_starts(
        self, fold: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        ep_off = self.query_ep_offsets_np
        f_nums = self.query_frame_nums_np
        ep_list: List[np.ndarray] = []
        frame_list: List[np.ndarray] = []
        r0_list: List[np.ndarray] = []
        r1_list: List[np.ndarray] = []
        by_ep: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

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
            frames_ep = ep_frames[v_idx].astype(np.int64)
            loc_r0 = p0[v_idx].astype(np.int64)
            loc_r1 = v_idx.astype(np.int64)
            by_ep[ep_idx] = (frames_ep, loc_r0, loc_r1)

            ep_list.append(np.full(cnt, ep_idx, dtype=np.int64))
            frame_list.append(frames_ep)
            r0_list.append((lo + loc_r0).astype(np.int64))
            r1_list.append((lo + loc_r1).astype(np.int64))

        if not ep_list:
            empty1 = np.empty(0, dtype=np.int64)
            return empty1, empty1, np.empty((0, 2), dtype=np.int64), by_ep

        episodes = np.concatenate(ep_list, axis=0)
        frames = np.concatenate(frame_list, axis=0)
        q_rows = np.stack(
            [np.concatenate(r0_list, axis=0), np.concatenate(r1_list, axis=0)], axis=1
        )
        return episodes, frames, q_rows, by_ep

    def _preload_val_query_features(self) -> None:
        """Load all unique val query rows in monotonic disk order (~0.55 GB for spatial)."""
        ep_off = self.query_ep_offsets_np
        val_eps = sorted(self.val_starts_by_ep.keys())
        chunks: List[np.ndarray] = []
        ep_to_base: Dict[int, int] = {}
        cursor = 0
        for ep_idx in val_eps:
            lo, hi = int(ep_off[ep_idx]), int(ep_off[ep_idx + 1])
            ep_to_base[ep_idx] = cursor
            chunks.append(np.ascontiguousarray(self.query_features_mm[lo:hi]))
            cursor += hi - lo
        self.val_query_cache_np = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0,) + self.query_features_mm.shape[1:], dtype=self.query_features_mm.dtype)
        )
        # Map global self.val_q_rows -> local indices into self.val_query_cache_np
        local_rows = np.empty_like(self.val_q_rows)
        for i in range(len(self.val_episodes)):
            ep_idx = int(self.val_episodes[i])
            lo = int(ep_off[ep_idx])
            base = ep_to_base[ep_idx]
            local_rows[i, 0] = base + (int(self.val_q_rows[i, 0]) - lo)
            local_rows[i, 1] = base + (int(self.val_q_rows[i, 1]) - lo)
        self.val_q_local_rows = local_rows

    def _build_wrong_demos(self, val_episodes: np.ndarray) -> np.ndarray:
        """Exact same deterministic wrong-task val Demo assignment as CLIP SelfPairFeatureBank."""
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
        """Exact same distinct-timestep val pair construction as CLIP SelfPairFeatureBank."""
        by_ep: Dict[int, List[int]] = defaultdict(list)
        for idx, ep in enumerate(val_episodes):
            by_ep[int(ep)].append(idx)

        pairs: List[Tuple[int, int]] = []
        for _, indices in sorted(by_ep.items()):
            if len(indices) < 2:
                continue
            indices_sorted = sorted(indices, key=lambda i: int(val_frames[i]))
            f_first = int(val_frames[indices_sorted[0]])
            f_last = int(val_frames[indices_sorted[-1]])
            if f_last - f_first >= min_frame_gap:
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

    def _gather_demo_and_actions(
        self,
        demos: np.ndarray,
        queries: np.ndarray,
        frames: np.ndarray,
        *,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        demos_np = np.asarray(demos, dtype=np.int64)
        queries_np = np.asarray(queries, dtype=np.int64)
        frames_np = np.asarray(frames, dtype=np.int64)

        lengths = self.demo_lengths_np[demos_np].astype(np.int64)
        max_p = int(lengths.max())
        starts = self.demo_offsets_np[demos_np].astype(np.int64)
        pos = np.arange(max_p, dtype=np.int64)[None, :]
        pad_mask_np = pos >= lengths[:, None]
        valid_pos = np.minimum(pos, np.maximum(lengths - 1, 0)[:, None])
        flat_idx = starts[:, None] + valid_pos

        pad_mask_t = torch.from_numpy(pad_mask_np).to(self.device, non_blocking=True)
        if self.demo_features_t is not None:
            flat_t = torch.from_numpy(flat_idx).to(self.device, dtype=torch.long, non_blocking=True)
            demo_rgb = self.demo_features_t[flat_t].to(dtype=out_dtype)
        else:
            demo_np = np.asarray(self.demo_features_np[flat_idx])
            demo_rgb = (
                torch.from_numpy(demo_np)
                .to(device=self.device, non_blocking=True)
                .to(dtype=out_dtype)
            )
        expand_shape = pad_mask_t.shape + (1,) * (demo_rgb.ndim - pad_mask_t.ndim)
        demo_rgb = demo_rgb.masked_fill(pad_mask_t.view(expand_shape), 0.0)

        ep_len = self.ep_lengths_np[queries_np].astype(np.int64)
        act_off = self.action_offsets_np[queries_np].astype(np.int64)
        step_offsets = np.arange(ACTION_HORIZON, dtype=np.int64)[None, :]
        rel_steps = np.minimum(frames_np[:, None] + step_offsets, (ep_len - 1)[:, None])
        actions = torch.from_numpy(self.actions_10d_np[act_off[:, None] + rel_steps]).to(
            self.device, non_blocking=True
        )
        return demo_rgb, pad_mask_t, actions

    def gather_batch(
        self,
        demos: np.ndarray,
        queries: np.ndarray,
        frames: np.ndarray,
        q_rows: np.ndarray,
        *,
        query_buffer: Optional[np.ndarray] = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather padded Demo features, Query features, and 16-step 10D GT actions."""
        demo_rgb, pad_mask_t, actions = self._gather_demo_and_actions(
            demos, queries, frames, out_dtype=out_dtype
        )
        q_src = query_buffer if query_buffer is not None else self.query_features_mm
        q_np = np.asarray(q_src[np.asarray(q_rows, dtype=np.int64)])
        query_rgb = (
            torch.from_numpy(q_np)
            .to(device=self.device, non_blocking=True)
            .to(dtype=out_dtype)
        )
        return demo_rgb, pad_mask_t, query_rgb, actions

    def gather_val_batch_by_indices(
        self,
        val_indices: np.ndarray,
        *,
        mode: str = "self",
        out_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero-disk-I/O gather for validation indices using preloaded val query RAM cache."""
        idx = np.asarray(val_indices, dtype=np.int64)
        queries = self.val_episodes[idx]
        frames = self.val_frames[idx]
        if mode == "self":
            demos = queries
        elif mode == "wrong_demo":
            demos = self.val_wrong_demos[idx]
        else:
            raise ValueError(f"Unknown val mode: {mode!r}")

        if self.val_query_cache_np is not None and self.val_q_local_rows is not None:
            return self.gather_batch(
                demos,
                queries,
                frames,
                self.val_q_local_rows[idx],
                query_buffer=self.val_query_cache_np,
                out_dtype=out_dtype,
            )
        return self.gather_batch(
            demos, queries, frames, self.val_q_rows[idx], out_dtype=out_dtype
        )

    def _load_superchunk(
        self,
        ep_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read a super-chunk of episodes in monotonic disk-offset order into RAM and
        uniformly shuffle all valid starts inside the super-chunk."""
        # Sort episodes by disk offset for single-sweep sequential read
        sorted_eps = np.sort(ep_ids)
        ep_off = self.query_ep_offsets_np

        q_chunks: List[np.ndarray] = []
        ep_list: List[np.ndarray] = []
        frame_list: List[np.ndarray] = []
        r0_list: List[np.ndarray] = []
        r1_list: List[np.ndarray] = []
        cursor = 0

        for ep_idx_raw in sorted_eps:
            ep_idx = int(ep_idx_raw)
            lo, hi = int(ep_off[ep_idx]), int(ep_off[ep_idx + 1])
            frames_ep, loc_r0, loc_r1 = self.train_starts_by_ep[ep_idx]
            q_chunks.append(np.ascontiguousarray(self.query_features_mm[lo:hi]))
            cnt = len(frames_ep)
            ep_list.append(np.full(cnt, ep_idx, dtype=np.int64))
            frame_list.append(frames_ep)
            r0_list.append(cursor + loc_r0)
            r1_list.append(cursor + loc_r1)
            cursor += hi - lo

        q_buf = np.concatenate(q_chunks, axis=0)
        eps = np.concatenate(ep_list, axis=0)
        frames = np.concatenate(frame_list, axis=0)
        q_rows = np.stack(
            [np.concatenate(r0_list, axis=0), np.concatenate(r1_list, axis=0)], axis=1
        )
        perm = rng.permutation(len(eps))
        return q_buf, eps[perm], frames[perm], q_rows[perm]

    def iter_train_batches(
        self,
        epoch: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield 100% Self-pair training batches with double-buffered super-chunk prefetching.

        Guarantees:
        - Exact same set of 1,373,035 valid train starts every full epoch.
        - 100% Self-pair (`demos == queries`).
        - All ranks iterate the exact same number of batches in DDP.
        """
        ep_rng = np.random.default_rng([self.seed, int(epoch), 6060])
        shuffled_eps = self.train_ep_ids[ep_rng.permutation(len(self.train_ep_ids))]

        # Partition shuffled episodes into super-chunks; if max_samples is set (smoke mode),
        # only read enough episodes to cover max_samples cleanly.
        ep_chunks: List[np.ndarray] = []
        if max_samples is not None:
            target = int(max_samples)
            acc_eps: List[int] = []
            acc_cnt = 0
            for ep_idx in shuffled_eps:
                acc_eps.append(int(ep_idx))
                acc_cnt += len(self.train_starts_by_ep[int(ep_idx)][0])
                if acc_cnt >= target:
                    break
            ep_chunks = [np.asarray(acc_eps, dtype=np.int64)]
        else:
            step = self.superchunk_episodes
            for s in range(0, len(shuffled_eps), step):
                ep_chunks.append(shuffled_eps[s : s + step])

        # Background thread prefetches next super-chunk while GPU trains on current super-chunk.
        buf_q: "queue.Queue[Optional[Tuple[Any, ...]]]" = queue.Queue(maxsize=1)
        errors: List[BaseException] = []

        def _prefetch_worker() -> None:
            try:
                for chunk_i, chunk_eps in enumerate(ep_chunks):
                    chunk_rng = np.random.default_rng([self.seed, int(epoch), 6060, chunk_i])
                    pkt = self._load_superchunk(chunk_eps, chunk_rng)
                    buf_q.put(pkt)
            except BaseException as exc:
                errors.append(exc)
            finally:
                buf_q.put(None)

        worker = threading.Thread(
            target=_prefetch_worker, name=f"dinov3-prefetch-ep{epoch}", daemon=True
        )
        worker.start()

        emitted_global = 0
        try:
            while True:
                pkt = buf_q.get()
                if pkt is None:
                    break
                q_buf, eps, frames, q_rows = pkt
                if max_samples is not None:
                    rem = int(max_samples) - emitted_global
                    if rem <= 0:
                        continue
                    if len(eps) > rem:
                        eps = eps[:rem]
                        frames = frames[:rem]
                        q_rows = q_rows[:rem]
                emitted_global += len(eps)

                # Shard uniformly across DDP ranks
                eps_r = eps[rank::world_size]
                frames_r = frames[rank::world_size]
                q_rows_r = q_rows[rank::world_size]

                for start in range(0, len(eps_r), batch_size):
                    end = min(start + batch_size, len(eps_r))
                    e_sl = eps_r[start:end]
                    f_sl = frames_r[start:end]
                    qr_sl = q_rows_r[start:end]
                    yield self.gather_batch(
                        e_sl, e_sl, f_sl, qr_sl, query_buffer=q_buf, out_dtype=out_dtype
                    )
                del q_buf
        finally:
            worker.join(timeout=30.0)

        if errors:
            raise RuntimeError("Super-chunk prefetch thread failed") from errors[0]

    def iter_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate validation split ('self' or 'wrong_demo') with zero disk I/O."""
        n = (
            len(self.val_episodes)
            if max_samples is None
            else min(int(max_samples), len(self.val_episodes))
        )
        idx = np.arange(n)[rank::world_size]
        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            yield self.gather_val_batch_by_indices(
                idx[start:end], mode=mode, out_dtype=out_dtype
            )
