"""Extract and cache frozen E7 & E9 Behavior Memory + trajectories for unseen LIBERO-Gen Chain tasks.

Uses strictly frozen CLIP ViT-B/16, frozen E7 Stage 1 Teacher, and frozen E9 Stage 1 Local Teacher
in a SINGLE shared CLIP encoding pass across 200 demos (10 unseen tasks x 20 demos).
Reuses the exact encoding convention of `e7/diagnose_behavior_space.py` and `e9/diagnose_stage2_d1_d4.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch

from e5 import BPP_CKPT, DEMO_STRIDE
from e5.actions import action_7d_to_10d
from e5.policy_server import CHAIN_HDF5_DIR, get_chain_tasks
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder
from e7.model import compute_behavior_anchor_steps, load_e7_policy
from e9.stage2_direct_model import load_e9_stage2_direct_policy

from . import (
    BEHAVIOR_STRIDE,
    E7_STAGE1_CKPT,
    E7_STAGE2_CKPT,
    E9_STAGE1_CKPT,
    E9_STAGE2_CKPT,
    OUTPUT_ROOT,
)

DEFAULT_UNSEEN_CACHE_DIR = OUTPUT_ROOT / "unseen_cache"


@torch.inference_mode()
def build_unseen_chain_cache(
    *,
    out_dir: Path = DEFAULT_UNSEEN_CACHE_DIR,
    demos_per_task: int = 20,
    device: torch.device,
    use_bf16: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    meta_path = out_dir / "unseen_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if int(meta.get("demos_per_task", -1)) == int(demos_per_task):
            print(f"[P0 UnseenCache] Reusing existing cache at {out_dir}", flush=True)
            return out_dir

    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_chain_tasks()

    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    e7_policy, _ = load_e7_policy(E7_STAGE2_CKPT, stage1_ckpt=E7_STAGE1_CKPT, device=device)
    e9_policy, _ = load_e9_stage2_direct_policy(
        E9_STAGE2_CKPT, stage1_ckpt=E9_STAGE1_CKPT, device=device
    )
    for mod in (visual, e7_policy, e9_policy):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)

    trainable = sum(
        p.numel()
        for mod in (visual, e7_policy, e9_policy)
        for p in mod.parameters()
        if p.requires_grad
    )
    if trainable != 0:
        raise RuntimeError(f"Expected 0 trainable params, found {trainable}")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    demos_meta: List[Dict[str, Any]] = []
    mem_h_list: List[np.ndarray] = []
    mem_b_e7_list: List[np.ndarray] = []
    mem_b_e9_list: List[np.ndarray] = []
    mem_anchors_list: List[np.ndarray] = []
    mem_offsets: List[int] = [0]
    mem_lengths: List[int] = []

    actions_10d_list: List[np.ndarray] = []
    ee_pos_list: List[np.ndarray] = []
    action_offsets: List[int] = [0]

    max_h_diff = 0.0
    total_demos = len(tasks) * demos_per_task
    done = 0

    for ti, task in enumerate(tasks):
        h5_path = CHAIN_HDF5_DIR / f"{task}_demo.hdf5"
        with h5py.File(h5_path, "r") as f:
            conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
            for di in range(demos_per_task):
                grp = f[f"data/demo_{di}"]
                av_all = grp["obs/agentview_rgb"][:]
                eh_all = grp["obs/eye_in_hand_rgb"][:]
                actions_7d = grp["actions"][:].astype(np.float32)
                ee_pos = grp["obs/ee_pos"][:].astype(np.float32)
                T = len(av_all)

                # Single CLIP encoding pass for all T frames (reused by both E7 and E9)
                all_feats = visual.encode_dual_camera_uint8(
                    np.ascontiguousarray(av_all),
                    np.ascontiguousarray(eh_all),
                    convention=conv,
                    batch_size=256,
                    autocast_dtype=torch.float16 if device.type == "cuda" else None,
                )  # (T, 2, 768)

                d_frames = np.arange(0, T, DEMO_STRIDE, dtype=np.int32)
                anchors = compute_behavior_anchor_steps(T, stride=BEHAVIOR_STRIDE)
                prev_anchors = np.maximum(anchors - 1, 0)

                d_pos = torch.as_tensor(d_frames, dtype=torch.long, device=device)
                p0_pos = torch.as_tensor(prev_anchors, dtype=torch.long, device=device)
                p1_pos = torch.as_tensor(anchors, dtype=torch.long, device=device)
                demo_rgb_1hz = all_feats[d_pos]
                anchor_query_rgb = torch.stack([all_feats[p0_pos], all_feats[p1_pos]], dim=1)

                with amp_ctx:
                    h_mem_e7, b_mem_e7 = e7_policy.extract_behavior_memory_from_single_demo(
                        demo_rgb_1hz, anchor_query_rgb
                    )
                    h_mem_e9, b_mem_e9, anchors_e9 = (
                        e9_policy.teacher.extract_behavior_memory_from_demo_features(
                            all_feats, stride=BEHAVIOR_STRIDE
                        )
                    )

                assert np.array_equal(anchors, anchors_e9), "Anchor mismatch E7 vs E9"
                h_diff = float((h_mem_e7.float() - h_mem_e9.float()).abs().max().item())
                max_h_diff = max(max_h_diff, h_diff)

                M = len(anchors)
                mem_h_list.append(h_mem_e9.float().cpu().numpy().astype(np.float16))
                mem_b_e7_list.append(b_mem_e7.float().cpu().numpy().astype(np.float16))
                mem_b_e9_list.append(b_mem_e9.float().cpu().numpy().astype(np.float16))
                mem_anchors_list.append(anchors.astype(np.int32))
                mem_lengths.append(M)
                mem_offsets.append(mem_offsets[-1] + M)

                actions_10d = action_7d_to_10d(actions_7d).astype(np.float32)
                actions_10d_list.append(actions_10d)
                ee_pos_list.append(ee_pos)
                action_offsets.append(action_offsets[-1] + T)

                demos_meta.append(
                    {
                        "ep_idx": done,
                        "task_idx": ti,
                        "task": task,
                        "demo_idx": di,
                        "demo_label": f"demo_{di}",
                        "frames": int(T),
                        "mem_len": int(M),
                    }
                )
                done += 1
                if done % 40 == 0 or done == total_demos:
                    dt = time.monotonic() - t0
                    print(
                        f"[P0 UnseenCache] {done}/{total_demos} demos encoded in {dt:.1f}s "
                        f"({done / max(dt, 1e-6):.1f} demos/s)",
                        flush=True,
                    )

    mem_h = np.concatenate(mem_h_list, axis=0)
    mem_b_e7 = np.concatenate(mem_b_e7_list, axis=0)
    mem_b_e9 = np.concatenate(mem_b_e9_list, axis=0)
    mem_anchor_steps = np.concatenate(mem_anchors_list, axis=0)
    actions_10d_all = np.concatenate(actions_10d_list, axis=0)
    ee_pos_all = np.concatenate(ee_pos_list, axis=0)

    np.save(out_dir / "mem_h.npy", mem_h)
    np.save(out_dir / "mem_b_e7.npy", mem_b_e7)
    np.save(out_dir / "mem_b_e9.npy", mem_b_e9)
    np.save(out_dir / "mem_anchor_steps.npy", mem_anchor_steps)
    np.save(out_dir / "mem_offsets.npy", np.asarray(mem_offsets, dtype=np.int64))
    np.save(out_dir / "mem_lengths.npy", np.asarray(mem_lengths, dtype=np.int32))
    np.save(out_dir / "actions_10d.npy", actions_10d_all)
    np.save(out_dir / "ee_pos.npy", ee_pos_all)
    np.save(out_dir / "action_offsets.npy", np.asarray(action_offsets, dtype=np.int64))

    meta = {
        "num_tasks": len(tasks),
        "tasks": tasks,
        "demos_per_task": int(demos_per_task),
        "num_demos": len(demos_meta),
        "total_anchors": int(len(mem_anchor_steps)),
        "total_steps": int(len(actions_10d_all)),
        "max_h_mem_diff_e7_vs_e9": max_h_diff,
        "e7_b_rms": float(np.sqrt(np.mean(mem_b_e7.astype(np.float64) ** 2))),
        "e9_b_rms": float(np.sqrt(np.mean(mem_b_e9.astype(np.float64) ** 2))),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "demos": demos_meta,
    }
    atomic_write_json(meta, meta_path)
    print(
        f"[P0 UnseenCache DONE] {len(demos_meta)} demos / {len(mem_anchor_steps):,} anchors "
        f"(max_h_diff={max_h_diff:.2e}, E7_rms={meta['e7_b_rms']:.4f}, E9_rms={meta['e9_b_rms']:.4f}) "
        f"-> {out_dir} in {meta['elapsed_seconds']}s",
        flush=True,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_UNSEEN_CACHE_DIR)
    parser.add_argument("--demos-per-task", type=int, default=20)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    build_unseen_chain_cache(
        out_dir=args.output,
        demos_per_task=args.demos_per_task,
        device=device,
        use_bf16=not args.no_bf16,
    )


if __name__ == "__main__":
    main()
