"""Batched GPU policy server over Unix domain socket for E8 closed-loop LIBERO-Gen Goal Chain
evaluation (Default 500 trials, Oracle-GT 331 paired trials, Shifted 500 paired trials).

Reuses E7's exact video-only Behavior Memory construction and `UnixBatchedServer`,
serving `E8Policy` (`z_c = sum w_i b_i`, `h_ref = sum w_i h_i^D`, `z_q = G_theta(z_c, h_ref, h_q)`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from e5 import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    BPP_CKPT,
    DEMO_STRIDE,
    EMBED_DIM,
)
from e5.policy_server import (
    CHAIN_HDF5_DIR,
    DEFAULT_PROMPT_SLOTS,
    get_chain_tasks,
)
from e5.vision import BPPVisualEncoder
from e7.model import compute_attention_stats, compute_behavior_anchor_steps
from e7.policy_server import ORACLE_GT_EXPERTS_DIR, SHIFT_OFFSET, UnixBatchedServer

from . import BEHAVIOR_STRIDE, DEFAULT_STAGE1_CKPT, NUM_QUERY_TOKENS
from .model import load_e8_policy


class E8BatchedEngine:
    """Pre-constructs Behavior Memory `(h_mem, b_mem)` from raw Demo RGB videos once on GPU
    and serves batched closed-loop inference for `e8` (and optional `e8_no_transfer`)."""

    def __init__(
        self,
        *,
        mode: str,
        prompt_mode: str,
        checkpoint: Path,
        stage1_ckpt: Optional[Path],
        device: torch.device,
        use_bf16: bool = True,
        behavior_stride: int = BEHAVIOR_STRIDE,
        tasks: Optional[List[str]] = None,
    ):
        if mode not in ("e8", "e8_no_transfer"):
            raise ValueError(f"Unsupported mode: {mode} (expected e8|e8_no_transfer)")
        if prompt_mode not in ("default", "default_paired", "shifted", "oracle_gt", "shuffle"):
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        self.mode = mode
        self.prompt_mode = prompt_mode
        self.apply_transfer = mode == "e8"
        self.behavior_stride = int(behavior_stride)
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
        self.policy, payload = load_e8_policy(
            checkpoint, stage1_ckpt=stage1_ckpt, device=device
        )
        self.ckpt_epoch = payload.get("epoch")

        # Online reader attention stats: (norm_entropy, max_weight, eff_pos_exp, valid_pos)
        self.online_attn_records: List[Tuple[float, float, float, float]] = []

        self.prompt_memories: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._preencode_all_prompts()

        mem_lens = [v[0].shape[0] for v in self.prompt_memories.values()]
        print(
            f"[E8Server] Ready: mode={mode} (apply_transfer={self.apply_transfer}) "
            f"prompt_mode={prompt_mode} ckpt={checkpoint} ep={self.ckpt_epoch} "
            f"prompts={len(self.prompt_memories)} "
            f"(entries min={min(mem_lens)}, max={max(mem_lens)}) "
            f"device={device} bf16={self.use_bf16} startup={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    @torch.inference_mode()
    def _encode_single_demo_video(
        self,
        d: h5py.Group,
        conv: str,
        amp_ctx,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        av_all = d["obs/agentview_rgb"][:]
        eh_all = d["obs/eye_in_hand_rgb"][:]
        T = len(av_all)
        if len(eh_all) != T:
            raise ValueError(f"Camera frame count mismatch in demo: {T} vs {len(eh_all)}")

        d_frames = np.arange(0, T, DEMO_STRIDE, dtype=np.int32)
        anchors = compute_behavior_anchor_steps(T, stride=self.behavior_stride)
        prev_anchors = np.maximum(anchors - 1, 0)

        union_frames = np.unique(np.concatenate([d_frames, prev_anchors, anchors]))
        feats_union = self.visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all[union_frames]),
            np.ascontiguousarray(eh_all[union_frames]),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if self.device.type == "cuda" else None,
        )

        d_pos = torch.as_tensor(
            np.searchsorted(union_frames, d_frames), dtype=torch.long, device=self.device
        )
        p0_pos = torch.as_tensor(
            np.searchsorted(union_frames, prev_anchors), dtype=torch.long, device=self.device
        )
        p1_pos = torch.as_tensor(
            np.searchsorted(union_frames, anchors), dtype=torch.long, device=self.device
        )

        demo_rgb_1hz = feats_union[d_pos]
        anchor_query_rgb = torch.stack([feats_union[p0_pos], feats_union[p1_pos]], dim=1)

        with amp_ctx:
            h_mem, b_mem = self.policy.extract_behavior_memory_from_single_demo(
                demo_rgb_1hz, anchor_query_rgb
            )
        return h_mem.float().contiguous(), b_mem.float().contiguous()

    @torch.inference_mode()
    def _preencode_all_prompts(self) -> None:
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.use_bf16
            else torch.autocast(device_type=self.device.type, enabled=False)
        )

        if self.prompt_mode in ("default", "default_paired"):
            for ti, task in enumerate(self.tasks):
                h5_path = CHAIN_HDF5_DIR / f"{task}_demo.hdf5"
                with h5py.File(h5_path, "r") as f:
                    conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                    for di in range(DEFAULT_PROMPT_SLOTS):
                        d = f[f"data/demo_{di}"]
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_video(
                            d, conv, amp_ctx
                        )
        elif self.prompt_mode == "shifted":
            for ti, task in enumerate(self.tasks):
                h5_path = CHAIN_HDF5_DIR / f"{task}_demo.hdf5"
                with h5py.File(h5_path, "r") as f:
                    conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                    for di in range(DEFAULT_PROMPT_SLOTS):
                        src_di = SHIFT_OFFSET + di
                        d = f[f"data/demo_{src_di}"]
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_video(
                            d, conv, amp_ctx
                        )
        elif self.prompt_mode == "shuffle":
            n_tasks = len(self.tasks)
            for ti in range(n_tasks):
                donor_task = self.tasks[(ti + 1) % n_tasks]
                h5_path = CHAIN_HDF5_DIR / f"{donor_task}_demo.hdf5"
                with h5py.File(h5_path, "r") as f:
                    conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                    for di in range(DEFAULT_PROMPT_SLOTS):
                        d = f[f"data/demo_{di}"]
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_video(
                            d, conv, amp_ctx
                        )
        elif self.prompt_mode == "oracle_gt":
            for ti, task in enumerate(self.tasks):
                task_dir = ORACLE_GT_EXPERTS_DIR / task
                if not task_dir.is_dir():
                    continue
                for qi in range(50):
                    h5_path = task_dir / f"query_{qi:02d}" / "demo.hdf5"
                    if not h5_path.is_file():
                        continue
                    with h5py.File(h5_path, "r") as f:
                        conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                        d = f["data/demo_0"]
                        self.prompt_memories[(ti, qi)] = self._encode_single_demo_video(
                            d, conv, amp_ctx
                        )

    @torch.inference_mode()
    def predict_batch(
        self,
        keys: List[Tuple[int, int]],
        seeds: List[int],
        rgb_batch: np.ndarray,
    ) -> np.ndarray:
        B = len(keys)
        dev = self.device

        av = rgb_batch[:, :, 0].reshape(B * 2, 128, 128, 3)
        eh = rgb_batch[:, :, 1].reshape(B * 2, 128, 128, 3)
        q_feats = self.visual.encode_dual_camera_uint8(
            av,
            eh,
            convention="opengl",
            batch_size=max(64, B * 2),
            autocast_dtype=torch.float16 if dev.type == "cuda" else None,
        ).reshape(B, 2, 2, EMBED_DIM)

        mems = [self.prompt_memories[k] for k in keys]
        max_m = max(hm.shape[0] for hm, _ in mems)
        h_mem = torch.zeros(
            (B, max_m, NUM_QUERY_TOKENS, EMBED_DIM), device=dev, dtype=torch.float32
        )
        b_mem = torch.zeros(
            (B, max_m, NUM_QUERY_TOKENS, EMBED_DIM), device=dev, dtype=torch.float32
        )
        mem_mask = torch.ones((B, max_m), device=dev, dtype=torch.bool)
        for i, (hm, bm) in enumerate(mems):
            L = hm.shape[0]
            h_mem[i, :L] = hm
            b_mem[i, :L] = bm
            mem_mask[i, :L] = False

        noises = []
        for s in seeds:
            gen = torch.Generator(device="cpu").manual_seed(int(s))
            noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
        init_noise = torch.stack(noises, dim=0).to(device=dev, dtype=torch.float32)

        amp_ctx = (
            torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
            if self.use_bf16
            else torch.autocast(device_type=dev.type, enabled=False)
        )
        with amp_ctx:
            out = self.policy.predict_action_from_behavior_memory(
                h_mem,
                b_mem,
                mem_mask,
                q_feats,
                apply_transfer=self.apply_transfer,
                init_noise=init_noise,
            )

        ast = compute_attention_stats(out["attn_weights"], mem_mask)
        ne = ast["norm_entropy"].cpu().numpy()
        mw = ast["max_weight"].cpu().numpy()
        ep = ast["eff_pos_exp"].cpu().numpy()
        vp = ast["valid_pos"].cpu().numpy()
        for i in range(B):
            self.online_attn_records.append((float(ne[i]), float(mw[i]), float(ep[i]), float(vp[i])))

        exec_7d = out["exec_action_7d"]
        return np.ascontiguousarray(exec_7d.float().cpu().numpy(), dtype=np.float32)

    def summarize_online_reader_stats(self) -> Optional[Dict[str, object]]:
        if not self.online_attn_records:
            return None
        arr = np.asarray(self.online_attn_records, dtype=np.float64)
        return {
            "mode": self.mode,
            "prompt_mode": self.prompt_mode,
            "total_replan_calls": int(len(arr)),
            "normalized_entropy_mean": float(arr[:, 0].mean()),
            "max_weight_mean": float(arr[:, 1].mean()),
            "effective_positions_exp_entropy": float(arr[:, 2].mean()),
            "valid_positions_mean": float(arr[:, 3].mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("e8", "e8_no_transfer"), required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("default", "default_paired", "shifted", "oracle_gt", "shuffle"),
        default="default",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--coalesce-ms", type=float, default=4.0)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = E8BatchedEngine(
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        checkpoint=args.checkpoint,
        stage1_ckpt=args.stage1_ckpt,
        device=device,
        use_bf16=not args.no_bf16,
    )
    server = UnixBatchedServer(
        engine=engine,
        socket_path=args.socket,
        max_batch=args.max_batch,
        coalesce_ms=args.coalesce_ms,
    )
    try:
        server.serve(ready_file=args.ready_file)
    finally:
        if args.stats_out is not None:
            st = engine.summarize_online_reader_stats()
            if st is not None:
                args.stats_out.parent.mkdir(parents=True, exist_ok=True)
                args.stats_out.write_text(json.dumps(st, indent=2) + "\n")


if __name__ == "__main__":
    main()
