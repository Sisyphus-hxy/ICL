"""Batched GPU policy server over Unix domain socket for E6 Stage 2 closed-loop
LIBERO-Gen Goal Chain evaluation (10 tasks x 50 layouts = 500 trials).

Wire-compatible with `e5/eval_libero_chain.py` (read-only client). Supports:
  - e6_stage2          : Frozen Stage 1 Teacher + Stage 2 ResidualTransformer (`z_cross + delta_z`)
  - e6_stage2_no_delta : Frozen Stage 1 Teacher only (`z_cross`, `delta_z = 0` ablation)
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import socket
import struct
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
    EXEC_HORIZON,
)
from e5.actions import extract_exec_actions
from e5.policy_server import (
    CHAIN_HDF5_DIR,
    DEFAULT_PROMPT_SLOTS,
    PROPRIO_BYTES,
    REQ_HDR,
    REQ_TOTAL_BYTES,
    RGB_BYTES,
    RGB_SHAPE,
    _recv_exact,
    get_chain_tasks,
)
from e5.vision import BPPVisualEncoder

from .stage2_1_model import load_stage2_1_policy
from .stage2_model import DEFAULT_STAGE1_CKPT, load_stage2_policy


class E6Stage2BatchedEngine:
    """Pre-encodes all 10x10 Chain prompts via frozen CLIP + Stage 1 PromptPooler
    and serves batched closed-loop inference for E6 Stage 2."""

    def __init__(
        self,
        *,
        mode: str,
        stage2_ckpt: Path,
        stage1_ckpt: Optional[Path],
        device: torch.device,
        use_bf16: bool = True,
        delta_steps: Optional[int] = None,
        delta_scale: float = 1.0,
        tasks: Optional[List[str]] = None,
    ):
        if mode not in (
            "e6_stage2",
            "e6_stage2_no_delta",
            "e6_stage2_1",
            "e6_stage2_1_no_delta",
            "e6_stage2_1_2step",
        ):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.apply_delta = mode in ("e6_stage2", "e6_stage2_1", "e6_stage2_1_2step")
        self.num_delta_steps = (
            int(delta_steps)
            if delta_steps is not None
            else (2 if mode == "e6_stage2_1_2step" else (1 if self.apply_delta else 0))
        )
        self.delta_scale = float(delta_scale)
        self.is_stage2_1 = mode in ("e6_stage2_1", "e6_stage2_1_no_delta", "e6_stage2_1_2step")
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
        if self.is_stage2_1:
            self.policy, payload = load_stage2_1_policy(
                stage2_ckpt, stage1_ckpt=stage1_ckpt, device=device
            )
        else:
            self.policy, payload = load_stage2_policy(
                stage2_ckpt, stage1_ckpt=stage1_ckpt, device=device
            )
        self.ckpt_epoch = payload.get("epoch")
        self.prompt_memories: Dict[Tuple[int, int], torch.Tensor] = {}
        self._preencode_all_prompts()
        print(
            f"[E6Stage2Server] Ready: mode={mode} (apply_delta={self.apply_delta}, "
            f"delta_steps={self.num_delta_steps}, delta_scale={self.delta_scale}) "
            f"ckpt={stage2_ckpt} ep={self.ckpt_epoch} prompts={len(self.prompt_memories)} "
            f"device={device} bf16={self.use_bf16} startup={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    @torch.inference_mode()
    def _preencode_all_prompts(self) -> None:
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.use_bf16
            else torch.autocast(device_type=self.device.type, enabled=False)
        )
        for ti, task in enumerate(self.tasks):
            h5_path = CHAIN_HDF5_DIR / f"{task}_demo.hdf5"
            if not h5_path.is_file():
                raise FileNotFoundError(f"Missing Chain prompt HDF5: {h5_path}")
            with h5py.File(h5_path, "r") as f:
                conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                for di in range(DEFAULT_PROMPT_SLOTS):
                    d = f[f"data/demo_{di}"]
                    av_1hz = d["obs/agentview_rgb"][::DEMO_STRIDE]
                    eh_1hz = d["obs/eye_in_hand_rgb"][::DEMO_STRIDE]
                    demo_rgb = self.visual.encode_dual_camera_uint8(
                        av_1hz,
                        eh_1hz,
                        convention=conv,
                        batch_size=256,
                        autocast_dtype=torch.float16 if self.device.type == "cuda" else None,
                    )  # (P, 2, 768) float32 on GPU
                    with amp_ctx:
                        mem = self.policy.encode_demo_memory(demo_rgb[None])  # (1, P, 768)
                    self.prompt_memories[(ti, di)] = mem[0].float().contiguous()

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
        max_m = max(m.shape[0] for m in mems)
        memory = torch.zeros((B, max_m, EMBED_DIM), device=dev, dtype=torch.float32)
        mem_mask = torch.ones((B, max_m), device=dev, dtype=torch.bool)
        for i, m in enumerate(mems):
            L = m.shape[0]
            memory[i, :L] = m
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
            if self.is_stage2_1:
                out = self.policy.predict_action_from_memory(
                    memory,
                    mem_mask,
                    q_feats,
                    apply_delta=self.apply_delta,
                    num_delta_steps=self.num_delta_steps,
                    delta_scale=self.delta_scale,
                    init_noise=init_noise,
                )
            else:
                out = self.policy.predict_action_from_memory(
                    memory,
                    mem_mask,
                    q_feats,
                    apply_delta=self.apply_delta,
                    init_noise=init_noise,
                )
        exec_7d = out["exec_action_7d"]
        return np.ascontiguousarray(exec_7d.float().cpu().numpy(), dtype=np.float32)


class UnixBatchedServer:
    def __init__(
        self,
        engine: E6Stage2BatchedEngine,
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
                f"[E6Stage2Server] Stopped after {self.stats['requests']} requests in "
                f"{self.stats['batches']} batches",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "e6_stage2",
            "e6_stage2_no_delta",
            "e6_stage2_1",
            "e6_stage2_1_no_delta",
            "e6_stage2_1_2step",
        ),
        default="e6_stage2",
    )
    parser.add_argument("--delta-steps", type=int, default=None)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--coalesce-ms", type=float, default=4.0)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = E6Stage2BatchedEngine(
        mode=args.mode,
        stage2_ckpt=args.checkpoint,
        stage1_ckpt=args.stage1_ckpt,
        device=device,
        use_bf16=not args.no_bf16,
        delta_steps=args.delta_steps,
        delta_scale=args.delta_scale,
    )
    server = UnixBatchedServer(
        engine=engine,
        socket_path=args.socket,
        max_batch=args.max_batch,
        coalesce_ms=args.coalesce_ms,
    )
    server.serve(ready_file=args.ready_file)


if __name__ == "__main__":
    main()
