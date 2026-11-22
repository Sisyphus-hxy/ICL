"""Batched GPU policy server over Unix domain socket for fast LIBERO-Gen Goal Chain evaluation.

Runs inside `env/train/e4_chain` (PyTorch CUDA BF16 + timm + einops) while parallel LIBERO
simulation workers run inside `env/libero/.conda` (MuJoCo + robosuite + EGL) and send raw
dual-camera uint8 frames + 11D proprio over a local AF_UNIX stream socket.

Supports all E5 / BPP models under the exact same server & deterministic DDIM seeding:
  - e5            (E5Policy, use_residual=True)
  - bpp_rgb_only  (E5Policy, use_residual=False)
  - bpp_full      (BPPFullPolicy, RGB + 11D proprio + 20-step prompt actions)
  - bpp_official  (Official liberogen_goal_chain_behavior_prompting.ckpt, sanity-check only)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    BPP_CKPT,
    DEMO_STRIDE,
    EMBED_DIM,
    EXEC_HORIZON,
    ROOT,
)
from .actions import action_7d_to_10d, extract_exec_actions
from .bpp_full import BPPFullPolicy, FULL_QUERY_TOKENS
from .model import E5Policy
from .vision import BPPVisualEncoder

CHAIN_SPLIT = "libero_goal_chain_selected_view"
CHAIN_BDDL_DIR = (
    ROOT
    / "bpp/src/behavior_prompting/train_network/env/libero/bddl_files"
    / CHAIN_SPLIT
)
CHAIN_HDF5_DIR = ROOT / "bpp/datasets/libero" / CHAIN_SPLIT
DEFAULT_PROMPT_SLOTS = 10

# Binary framing over Unix socket:
#   Request:  <iiii (task_idx, donor_idx, seed, flags)
#             + rgb uint8 (2, 2, 128, 128, 3) = 196,608 bytes
#             + proprio float32 (2, 11)       = 88 bytes
#   Response: exec_actions_7d float32 (12, 7) = 336 bytes
REQ_HDR = struct.Struct("<iiii")
RGB_SHAPE = (2, 2, 128, 128, 3)
RGB_BYTES = int(np.prod(RGB_SHAPE))
PROPRIO_SHAPE = (2, 11)
PROPRIO_BYTES = int(np.prod(PROPRIO_SHAPE)) * 4
REQ_TOTAL_BYTES = REQ_HDR.size + RGB_BYTES + PROPRIO_BYTES
RESP_BYTES = EXEC_HORIZON * 7 * 4


def get_chain_tasks() -> List[str]:
    tasks = sorted(p.stem for p in CHAIN_BDDL_DIR.glob("*.bddl"))
    if len(tasks) != 10:
        raise RuntimeError(f"Expected 10 Chain tasks under {CHAIN_BDDL_DIR}, found {len(tasks)}")
    return tasks


def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        chunk = conn.recv_into(mv[got:], n - got)
        if chunk == 0:
            if got == 0:
                return None
            raise EOFError(f"Truncated socket read: {got}/{n} bytes")
        got += chunk
    return bytes(buf)


class BatchedPolicyEngine:
    """Pre-encodes all 10x10 Chain prompts once on GPU and serves batched closed-loop inference."""

    def __init__(
        self,
        *,
        mode: str,
        checkpoint: Optional[Path],
        device: torch.device,
        use_bf16: bool = True,
        ref_only: bool = False,
        tasks: Optional[List[str]] = None,
    ):
        self.mode = mode
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.ref_only = ref_only
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()

        if mode in ("e5", "bpp_rgb_only"):
            if checkpoint is None:
                raise ValueError(f"--checkpoint is required for mode={mode}")
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.policy = E5Policy(
                use_residual=(mode == "e5"),
                load_pretrained_reader=False,
            )
            self.policy.load_state_dict(ck["model"], strict=True)
            self.ckpt_epoch = ck.get("epoch")
        elif mode == "bpp_full":
            if checkpoint is None:
                raise ValueError("--checkpoint is required for mode=bpp_full")
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.policy = BPPFullPolicy(load_pretrained_reader=False)
            self.policy.load_state_dict(ck["model"], strict=True)
            self.ckpt_epoch = ck.get("epoch")
        elif mode == "bpp_official":
            self.policy = BPPFullPolicy(load_pretrained_reader=False)
            self.policy.load_full_official_bpp_checkpoint(BPP_CKPT)
            self.ckpt_epoch = "official"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.policy = self.policy.to(device).eval()
        self.prompt_memories: Dict[Tuple[int, int], torch.Tensor] = {}
        self._preencode_all_prompts()
        print(
            f"[PolicyServer] Ready: mode={mode} ckpt={checkpoint} ep={self.ckpt_epoch} "
            f"prompts={len(self.prompt_memories)} device={device} bf16={self.use_bf16} "
            f"startup={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    @torch.inference_mode()
    def _preencode_all_prompts(self) -> None:
        """Encode all 10 tasks x 10 default demos (demo_0..demo_9) once into GPU memory tokens."""
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
                    P = len(av_1hz)
                    demo_rgb = self.visual.encode_dual_camera_uint8(
                        av_1hz,
                        eh_1hz,
                        convention=conv,
                        batch_size=256,
                        autocast_dtype=torch.float16 if self.device.type == "cuda" else None,
                    )  # (P, 2, 768) float32 on GPU

                    with amp_ctx:
                        if self.mode in ("e5", "bpp_rgb_only"):
                            assert isinstance(self.policy, E5Policy)
                            M_d = self.policy.prompt_pooler(demo_rgb[None])  # (1, P, 768)
                            rr = self.policy.reference_reader
                            mem = M_d + rr.prompt_pos_emb[:, :P, :]
                            if rr.attention_sink_tokens is not None:
                                mem = torch.cat([rr.attention_sink_tokens, mem], dim=1)
                        else:
                            assert isinstance(self.policy, BPPFullPolicy)
                            ee_pos = d["obs/ee_pos"][::DEMO_STRIDE].astype(np.float32)
                            ee_ori_rv = d["obs/ee_ori"][::DEMO_STRIDE].astype(np.float32)
                            ee_ori = (
                                Rotation.from_rotvec(ee_ori_rv)
                                .as_matrix()[:, :2, :]
                                .reshape(-1, 6)
                                .astype(np.float32)
                            )
                            grip = d["obs/gripper_states"][::DEMO_STRIDE].astype(np.float32)
                            demo_prop = torch.from_numpy(
                                np.concatenate([ee_pos, ee_ori, grip], axis=-1)
                            ).to(device=self.device, dtype=torch.float32)[None]  # (1, P, 11)

                            acts_10d = action_7d_to_10d(d["actions"][:].astype(np.float32))
                            target_len = P * DEMO_STRIDE
                            if len(acts_10d) < target_len:
                                pad = np.zeros(
                                    (target_len - len(acts_10d), ACTION_DIM_10D), dtype=np.float32
                                )
                                acts_10d = np.concatenate([acts_10d, pad], axis=0)
                            else:
                                acts_10d = acts_10d[:target_len]
                            demo_acts = torch.from_numpy(
                                acts_10d.reshape(1, P, DEMO_STRIDE, ACTION_DIM_10D)
                            ).to(device=self.device, dtype=torch.float32)

                            d_ori, d_pos, d_grip = self.policy._project_proprio_tokens(demo_prop)
                            d_act_norm = self.policy.action_normalizer.normalize(demo_acts)
                            d_act_tok = self.policy.action_proj(d_act_norm.reshape(1, P, -1))
                            prompt_6 = torch.stack(
                                [
                                    demo_rgb[None, :, 0, :],
                                    demo_rgb[None, :, 1, :],
                                    d_ori,
                                    d_pos,
                                    d_grip,
                                    d_act_tok,
                                ],
                                dim=2,
                            )  # (1, P, 6, 768)
                            M_d = self.policy.prompt_attention_pool(
                                prompt_6.reshape(P, 6, EMBED_DIM)
                            ).reshape(1, P, EMBED_DIM)
                            mem = M_d + self.policy.prompt_pos_emb[:, :P, :]
                            mem = torch.cat([self.policy.attention_sink_tokens, mem], dim=1)

                    self.prompt_memories[(ti, di)] = mem[0].float().contiguous()

    @torch.inference_mode()
    def predict_batch(
        self,
        keys: List[Tuple[int, int]],
        seeds: List[int],
        rgb_batch: np.ndarray,
        proprio_batch: np.ndarray,
    ) -> np.ndarray:
        """Run one batched closed-loop inference step -> (B, 12, 7) float32 OSC actions."""
        B = len(keys)
        dev = self.device

        # 1. Dual-camera CLIP encoding (matches offline cache fp16 autocast)
        av = rgb_batch[:, :, 0].reshape(B * 2, 128, 128, 3)
        eh = rgb_batch[:, :, 1].reshape(B * 2, 128, 128, 3)
        q_feats = self.visual.encode_dual_camera_uint8(
            av,
            eh,
            convention="opengl",
            batch_size=max(64, B * 2),
            autocast_dtype=torch.float16 if dev.type == "cuda" else None,
        ).reshape(B, 2, 2, EMBED_DIM)

        # 2. Assemble padded prompt memory tensors
        mems = [self.prompt_memories[k] for k in keys]
        max_m = max(m.shape[0] for m in mems)
        memory = torch.zeros((B, max_m, EMBED_DIM), device=dev, dtype=torch.float32)
        mem_mask = torch.ones((B, max_m), device=dev, dtype=torch.bool)
        for i, m in enumerate(mems):
            L = m.shape[0]
            memory[i, :L] = m
            mem_mask[i, :L] = False

        # 3. Deterministic per-request DDIM initial noise (invariant to batching order)
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
            if self.mode in ("e5", "bpp_rgb_only"):
                assert isinstance(self.policy, E5Policy)
                h_q = self.policy.format_query_tokens(q_feats)  # (B, 4, 768)
                rr = self.policy.reference_reader
                z_ref = rr.prompt_with_obs_decoder(
                    tgt=h_q + rr.prompt_current_obs_pos_emb,
                    memory=memory,
                    memory_key_padding_mask=mem_mask,
                )
                if self.policy.residual_transformer is not None and not self.ref_only:
                    z_cond = z_ref + self.policy.residual_transformer(z_ref, h_q)
                else:
                    z_cond = z_ref
                gcond = self.policy._build_global_cond(h_q, z_cond)
                action_10d = self.policy.sample_from_global_cond(gcond, init_noise=init_noise)
            else:
                assert isinstance(self.policy, BPPFullPolicy)
                q_prop = torch.from_numpy(proprio_batch).to(device=dev, dtype=torch.float32)
                q_ori, q_pos, q_grip = self.policy._project_proprio_tokens(q_prop)
                h_q = torch.stack(
                    [q_feats[:, :, 0], q_feats[:, :, 1], q_ori, q_pos, q_grip], dim=2
                ).reshape(B, FULL_QUERY_TOKENS, EMBED_DIM)
                z_ref = self.policy.prompt_with_obs_decoder(
                    tgt=h_q + self.policy.prompt_current_obs_pos_emb,
                    memory=memory,
                    memory_key_padding_mask=mem_mask,
                )
                gcond = torch.cat([h_q, z_ref], dim=1).reshape(B, -1)
                traj = init_noise.clone()
                self.policy.noise_scheduler.set_timesteps(
                    self.policy.num_inference_steps, device=dev
                )
                for t in self.policy.noise_scheduler.timesteps:
                    out = self.policy.action_decoder(
                        traj.to(dtype=gcond.dtype), t, local_cond=None, global_cond=gcond
                    ).float()
                    traj = self.policy.noise_scheduler.step(out, t, traj).prev_sample
                action_10d = self.policy.action_normalizer.unnormalize(traj)

        exec_7d = extract_exec_actions(action_10d, exec_horizon=EXEC_HORIZON)
        return np.ascontiguousarray(exec_7d.float().cpu().numpy(), dtype=np.float32)


class UnixBatchedServer:
    def __init__(
        self,
        engine: BatchedPolicyEngine,
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
                prop = np.frombuffer(
                    raw,
                    dtype=np.float32,
                    count=PROPRIO_BYTES // 4,
                    offset=REQ_HDR.size + RGB_BYTES,
                ).reshape(PROPRIO_SHAPE)
                done_evt = threading.Event()
                slot: List[Optional[bytes]] = [None]
                self.req_q.put(((ti, di), seed, rgb, prop, done_evt, slot))
                done_evt.wait()
                resp = slot[0]
                if resp is None:
                    break
                conn.sendall(resp)
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _gpu_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                first = self.req_q.get(timeout=0.05)
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

            keys = [x[0] for x in batch]
            seeds = [x[1] for x in batch]
            rgb_b = np.stack([x[2] for x in batch], axis=0)
            prop_b = np.stack([x[3] for x in batch], axis=0)
            actions = self.engine.predict_batch(keys, seeds, rgb_b, prop_b)
            self.stats["requests"] += len(batch)
            self.stats["batches"] += 1
            for i, item in enumerate(batch):
                item[5][0] = actions[i].tobytes()
                item[4].set()

    def serve(self, ready_file: Optional[Path] = None) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        srv.listen(64)
        srv.settimeout(0.2)

        gpu_t = threading.Thread(target=self._gpu_loop, daemon=True)
        gpu_t.start()

        if ready_file is not None:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.write_text(
                json.dumps(
                    {
                        "socket": str(self.socket_path),
                        "mode": self.engine.mode,
                        "tasks": self.engine.tasks,
                    }
                )
                + "\n"
            )

        def _handle_sig(_signum, _frame):
            self.stop_event.set()

        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)

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
            avg_b = self.stats["requests"] / max(self.stats["batches"], 1)
            print(
                f"[PolicyServer] Shutdown: requests={self.stats['requests']} "
                f"batches={self.stats['batches']} avg_batch={avg_b:.2f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("e5", "bpp_rgb_only", "bpp_full", "bpp_official"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--coalesce-ms", type=float, default=4.0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--ref-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = BatchedPolicyEngine(
        mode=args.mode,
        checkpoint=args.checkpoint,
        device=device,
        use_bf16=(not args.fp32),
        ref_only=args.ref_only,
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
