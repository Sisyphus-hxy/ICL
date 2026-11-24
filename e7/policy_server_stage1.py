"""Batched GPU policy server over Unix domain socket for Stage 1 Teacher Closed-Loop Diagnosis.

Bypasses E7's Reader and Direct Transfer completely and runs frozen E6 Stage 1 Teacher directly:
    z_t = R_T(M_D, h_t),    a_hat_t = U(z_t)

where:
  - M_D is the full 1Hz Demo memory encoded by frozen CLIP + Stage 1 PromptPooler (+ prompt_pos_emb).
  - h_t is the ACTUAL closed-loop online RGB observation history [t-1, t] (4 tokens) from MuJoCo.
  - Strictly reuses the exact LIBERO-Gen Goal Chain initial states, Demos, DDIM noise seeds,
    12-step warmup, 12-step execution horizon, and termination criteria of E7-Direct evaluation.
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
    EXEC_HORIZON,
)
from e5.actions import extract_exec_actions
from e5.model import E5Policy
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
from e6.stage1_model import load_frozen_stage1_teacher

from . import DEFAULT_STAGE1_CKPT
from .policy_server import ORACLE_GT_EXPERTS_DIR, SHIFT_OFFSET


class Stage1TeacherBatchedEngine:
    """Pre-encodes full 1Hz Demo memories `M_D + prompt_pos_emb` once on GPU and serves
    batched closed-loop Stage 1 Teacher inference: z_t = R_T(M_D, h_t), a_hat_t = U(z_t)."""

    def __init__(
        self,
        *,
        prompt_mode: str,
        stage1_ckpt: Path,
        device: torch.device,
        use_bf16: bool = True,
        tasks: Optional[List[str]] = None,
    ):
        if prompt_mode not in ("default", "default_paired", "shifted", "oracle_gt"):
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        self.prompt_mode = prompt_mode
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)

        self.teacher = load_frozen_stage1_teacher(stage1_ckpt, device=device)
        if self.teacher.reference_reader.attention_sink_tokens is not None:
            raise RuntimeError("Unexpected attention_sink_tokens in Stage 1 Teacher")

        self.prompt_memories: Dict[Tuple[int, int], torch.Tensor] = {}
        self._preencode_all_prompts()

        mem_lens = [v.shape[0] for v in self.prompt_memories.values()]
        print(
            f"[Stage1Server] Ready: mode=stage1_teacher prompt_mode={prompt_mode} "
            f"ckpt={stage1_ckpt} prompts={len(self.prompt_memories)} "
            f"(1Hz len min={min(mem_lens)}, max={max(mem_lens)}) "
            f"device={device} bf16={self.use_bf16} startup={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    @torch.inference_mode()
    def _encode_single_demo_1hz(
        self,
        d: h5py.Group,
        conv: str,
        amp_ctx,
    ) -> torch.Tensor:
        """Encode full Demo 1Hz RGB stream -> `M_D + prompt_pos_emb` of shape `(P, 768)`."""
        av_1hz = np.ascontiguousarray(d["obs/agentview_rgb"][::DEMO_STRIDE])
        eh_1hz = np.ascontiguousarray(d["obs/eye_in_hand_rgb"][::DEMO_STRIDE])
        if len(av_1hz) != len(eh_1hz):
            raise ValueError(f"Camera frame count mismatch: {len(av_1hz)} vs {len(eh_1hz)}")

        demo_rgb = self.visual.encode_dual_camera_uint8(
            av_1hz,
            eh_1hz,
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if self.device.type == "cuda" else None,
        )  # (P, 2, 768) float32 on GPU

        rr = self.teacher.reference_reader
        P = demo_rgb.shape[0]
        if P > rr.max_prompt_tokens:
            demo_rgb = demo_rgb[: rr.max_prompt_tokens]
            P = rr.max_prompt_tokens

        with amp_ctx:
            M_d = self.teacher.prompt_pooler(demo_rgb.unsqueeze(0), demo_pad_mask=None)
            mem = M_d + rr.prompt_pos_emb[:, :P, :]
        return mem[0].float().contiguous()

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
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_1hz(
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
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_1hz(
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
                        self.prompt_memories[(ti, qi)] = self._encode_single_demo_1hz(
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
        max_p = max(m.shape[0] for m in mems)
        memory = torch.zeros((B, max_p, EMBED_DIM), device=dev, dtype=torch.float32)
        mem_mask = torch.ones((B, max_p), device=dev, dtype=torch.bool)
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
        rr = self.teacher.reference_reader
        with amp_ctx:
            h_t = E5Policy.format_query_tokens(q_feats)  # (B, 4, 768)
            z_t = rr.prompt_with_obs_decoder(
                tgt=h_t + rr.prompt_current_obs_pos_emb,
                memory=memory,
                memory_key_padding_mask=mem_mask,
                need_weights=False,
            )
            action_10d = self.teacher.sample_from_z_self(z_t, init_noise=init_noise)
            exec_7d = extract_exec_actions(action_10d, exec_horizon=EXEC_HORIZON)

        return np.ascontiguousarray(exec_7d.float().cpu().numpy(), dtype=np.float32)


class UnixBatchedServer:
    def __init__(
        self,
        engine: Stage1TeacherBatchedEngine,
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
                f"[Stage1Server] Stopped after {self.stats['requests']} requests in "
                f"{self.stats['batches']} batches",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-mode",
        choices=("default", "default_paired", "shifted", "oracle_gt"),
        required=True,
    )
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--coalesce-ms", type=float, default=4.0)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = Stage1TeacherBatchedEngine(
        prompt_mode=args.prompt_mode,
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
    server.serve(ready_file=args.ready_file)


if __name__ == "__main__":
    main()
