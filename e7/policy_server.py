"""Batched GPU policy server over Unix domain socket for E7 closed-loop LIBERO-Gen Goal Chain
evaluation (Default 500 trials, Oracle-GT 331 paired trials, Shifted 500 paired trials).

At inference time, Behavior Memory `(h_i^D, b_i)` is constructed STRICTLY from the provided
Demo RGB video (`agentview_rgb`, `eye_in_hand_rgb`) using frozen CLIP + frozen Stage 1 Teacher:
  1. Encode full Demo 1Hz RGB sequence -> `M_D` (complete Demo context).
  2. At stride-16 steps `t_i = 0, 16, 32, ... < T`, encode `[max(0, t_i - 1), t_i]` RGB -> `h_i^D`
     and extract `b_i = R_T(M_D, h_i^D)`.
  3. At each online query step, retrieve `z_c = sum_i w_i b_i` via `SimpleBehaviorReader`,
     apply single-step Delta correction (`e7`) or zero Delta (`e7_no_delta`), and sample
     16-step 10D -> 12-step 7D actions from frozen Stage 1 UNet.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import socket
import threading
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
    ROOT,
)
from e5.policy_server import (
    CHAIN_HDF5_DIR,
    DEFAULT_PROMPT_SLOTS,
    REQ_HDR,
    REQ_TOTAL_BYTES,
    RGB_BYTES,
    RGB_SHAPE,
    _recv_exact,
    get_chain_tasks,
)
from e5.vision import BPPVisualEncoder

from . import BEHAVIOR_STRIDE, DEFAULT_STAGE1_CKPT, NUM_QUERY_TOKENS
from .adjacent_model import load_e7_adjacent_policy
from .model import compute_behavior_anchor_steps, load_e7_policy

ORACLE_GT_EXPERTS_DIR = (
    ROOT / "bpp/experiments/oracle_gt/outputs/liberogen_goal_chain/experts"
)
SHIFT_OFFSET = 10


class E7BatchedEngine:
    """Pre-constructs Behavior Memory from provided Demo videos once on GPU and serves
    batched closed-loop inference for `e7` (single-step Delta) and `e7_no_delta`."""

    def __init__(
        self,
        *,
        mode: str,
        prompt_mode: str,
        checkpoint: Path,
        stage1_ckpt: Optional[Path],
        device: torch.device,
        use_bf16: bool = True,
        delta_scale: float = 1.0,
        behavior_stride: int = BEHAVIOR_STRIDE,
        tasks: Optional[List[str]] = None,
    ):
        if mode not in ("e7", "e7_no_delta", "e7_adjacent", "e7_adjacent_no_delta"):
            raise ValueError(
                f"Unsupported mode: {mode} (expected e7|e7_no_delta|e7_adjacent|e7_adjacent_no_delta)"
            )
        if prompt_mode not in ("default", "default_paired", "shifted", "oracle_gt", "shuffle"):
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        self.mode = mode
        self.prompt_mode = prompt_mode
        self.apply_delta = mode in ("e7", "e7_adjacent")
        self.is_adjacent = mode in ("e7_adjacent", "e7_adjacent_no_delta")
        self.delta_scale = float(delta_scale)
        self.behavior_stride = int(behavior_stride)
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
        if self.is_adjacent:
            self.policy, payload = load_e7_adjacent_policy(
                checkpoint, stage1_ckpt=stage1_ckpt, device=device
            )
        else:
            self.policy, payload = load_e7_policy(
                checkpoint, stage1_ckpt=stage1_ckpt, device=device
            )
        self.ckpt_epoch = payload.get("epoch")
        # Online closed-loop reader diagnostics (replan_idx -> list of (norm_pos, beta))
        self.online_records: List[Tuple[int, float, float, int, int]] = []

        self.prompt_memories: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._preencode_all_prompts()

        mem_lens = [v[0].shape[0] for v in self.prompt_memories.values()]
        pred_mode = getattr(self.policy, "prediction_mode", "residual")
        print(
            f"[E7Server] Ready: mode={mode} (prediction_mode={pred_mode}, "
            f"apply_delta={self.apply_delta}, delta_scale={self.delta_scale}) "
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
        """Construct `(h_mem, b_mem)` strictly from raw Demo RGB video streams."""
        av_all = d["obs/agentview_rgb"][:]
        eh_all = d["obs/eye_in_hand_rgb"][:]
        T = len(av_all)
        if len(eh_all) != T:
            raise ValueError(f"Camera frame count mismatch in demo: {T} vs {len(eh_all)}")

        # 1. Full Demo 1Hz frames (stride 20) + stride-16 anchor frames [max(0, t_i - 1), t_i]
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
        )  # (U, 2, 768) float32 on GPU

        d_pos = torch.as_tensor(
            np.searchsorted(union_frames, d_frames), dtype=torch.long, device=self.device
        )
        p0_pos = torch.as_tensor(
            np.searchsorted(union_frames, prev_anchors), dtype=torch.long, device=self.device
        )
        p1_pos = torch.as_tensor(
            np.searchsorted(union_frames, anchors), dtype=torch.long, device=self.device
        )

        demo_rgb_1hz = feats_union[d_pos]  # (P, 2, 768)
        anchor_query_rgb = torch.stack(
            [feats_union[p0_pos], feats_union[p1_pos]], dim=1
        )  # (M, 2, 2, 768)

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
                apply_delta=self.apply_delta,
                delta_scale=self.delta_scale,
                init_noise=init_noise,
            )
        if self.is_adjacent and "reader_info" in out:
            ri = out["reader_info"]
            pidx_np = ri["pair_idx"].detach().cpu().numpy()
            beta_np = ri["beta_selected"].detach().cpu().numpy()
            npairs_np = ri["pair_valid_mask"].sum(dim=-1).detach().cpu().numpy()
            for i in range(B):
                replan_idx = int(seeds[i]) % 100
                np_i = max(int(npairs_np[i]), 1)
                idx_i = int(pidx_np[i])
                b_i = float(beta_np[i])
                npos = min(1.0, max(0.0, (idx_i + b_i) / float(np_i)))
                self.online_records.append((replan_idx, npos, b_i, idx_i, np_i))
        exec_7d = out["exec_action_7d"]
        return np.ascontiguousarray(exec_7d.float().cpu().numpy(), dtype=np.float32)

    def summarize_online_reader_stats(self) -> Optional[Dict[str, object]]:
        if not self.is_adjacent or not self.online_records:
            return None
        arr = np.asarray(self.online_records, dtype=np.float64)
        replans = arr[:, 0].astype(np.int64)
        npos = arr[:, 1]
        betas = arr[:, 2]
        pidx = arr[:, 3].astype(np.int64)
        npairs = arr[:, 4].astype(np.int64)

        bins = np.linspace(0.0, 1.0, 11)
        pos_hist, _ = np.histogram(npos, bins=bins)
        beta_hist, _ = np.histogram(betas, bins=bins)

        # Track selected normalized position across execution time (replan_idx = 0..45)
        by_replan = []
        for r in range(0, int(replans.max()) + 1):
            m = replans == r
            cnt = int(m.sum())
            if cnt == 0:
                continue
            by_replan.append(
                {
                    "replan_idx": int(r),
                    "env_step": int(r * 12),
                    "count": cnt,
                    "mean_selected_pos": round(float(npos[m].mean()), 4),
                    "std_selected_pos": round(float(npos[m].std()), 4),
                    "mean_beta": round(float(betas[m].mean()), 4),
                }
            )

        xm = replans.astype(np.float64) - float(replans.mean())
        ym = npos - float(npos.mean())
        denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
        pearson_r = float((xm * ym).sum() / denom) if denom > 1e-12 else 0.0

        return {
            "mode": self.mode,
            "prompt_mode": self.prompt_mode,
            "total_replan_calls": int(len(arr)),
            "selected_pos_mean": float(npos.mean()),
            "selected_pos_std": float(npos.std()),
            "first_pair_frac": float((pidx == 0).mean()),
            "last_pair_frac": float((pidx == np.maximum(npairs - 1, 0)).mean()),
            "pos_hist_10bins_frac": [round(float(x), 5) for x in (pos_hist / len(arr))],
            "beta_mean": float(betas.mean()),
            "beta_std": float(betas.std()),
            "beta_p25": float(np.quantile(betas, 0.25)),
            "beta_p50": float(np.quantile(betas, 0.50)),
            "beta_p75": float(np.quantile(betas, 0.75)),
            "beta_hist_10bins_frac": [round(float(x), 5) for x in (beta_hist / len(arr))],
            "replan_vs_selected_pos_pearson_r": pearson_r,
            "by_replan_step": by_replan,
        }


class UnixBatchedServer:
    def __init__(
        self,
        engine: E7BatchedEngine,
        socket_path: Path,
        max_batch: int = 20,
        coalesce_ms: float = 4.0,
    ):
        self.engine = engine
        self.socket_path = Path(socket_path)
        self.max_batch = max_batch
        self.coalesce_s = coalesce_ms / 1000.0
        self.req_q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.stats = {"requests": 0, "batches": 0}

    def _client_thread(self, conn: socket.socket) -> None:
        try:
            while not self.stop_event.is_set():
                raw = _recv_exact(conn, REQ_TOTAL_BYTES)
                if raw is None:
                    break
                ti, di, seed, _flags = REQ_HDR.unpack_from(raw, 0)
                rgb = np.frombuffer(
                    raw, dtype=np.uint8, count=RGB_BYTES, offset=REQ_HDR.size
                ).reshape(RGB_SHAPE)
                reply_q: queue.Queue = queue.Queue(maxsize=1)
                self.req_q.put(((ti, di), seed, rgb.copy(), reply_q))
                resp = reply_q.get()
                if isinstance(resp, BaseException):
                    raise resp
                conn.sendall(resp)
        except (BrokenPipeError, ConnectionResetError, EOFError):
            pass
        finally:
            conn.close()

    def _batch_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                first = self.req_q.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.coalesce_s
            while len(batch) < self.max_batch:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                try:
                    batch.append(self.req_q.get(timeout=rem))
                except queue.Empty:
                    break

            keys = [item[0] for item in batch]
            seeds = [item[1] for item in batch]
            rgbs = np.stack([item[2] for item in batch], axis=0)
            try:
                actions = self.engine.predict_batch(keys, seeds, rgbs)
                self.stats["requests"] += len(batch)
                self.stats["batches"] += 1
                for i, item in enumerate(batch):
                    item[3].put(actions[i].tobytes())
            except BaseException as exc:
                for item in batch:
                    item[3].put(exc)

    def serve(self, ready_file: Optional[Path] = None) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        worker = threading.Thread(target=self._batch_worker, daemon=True)
        worker.start()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        srv.listen(64)
        srv.settimeout(0.5)

        def _handle_sig(_signum, _frame):
            self.stop_event.set()

        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)

        if ready_file is not None:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.write_text(f"pid={os.getpid()} socket={self.socket_path}\n")

        try:
            while not self.stop_event.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self._client_thread, args=(conn,), daemon=True)
                t.start()
        finally:
            self.stop_event.set()
            srv.close()
            if self.socket_path.exists():
                self.socket_path.unlink(missing_ok=True)
            print(
                f"[E7Server] Stopped after {self.stats['requests']} requests in "
                f"{self.stats['batches']} batches",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("e7", "e7_no_delta", "e7_adjacent", "e7_adjacent_no_delta"),
        required=True,
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("default", "default_paired", "shifted", "oracle_gt", "shuffle"),
        default="default",
    )
    parser.add_argument("--delta-scale", type=float, default=1.0)
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
    engine = E7BatchedEngine(
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        checkpoint=args.checkpoint,
        stage1_ckpt=args.stage1_ckpt,
        device=device,
        use_bf16=not args.no_bf16,
        delta_scale=args.delta_scale,
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
                import json

                args.stats_out.parent.mkdir(parents=True, exist_ok=True)
                args.stats_out.write_text(json.dumps(st, indent=2) + "\n")


if __name__ == "__main__":
    main()
