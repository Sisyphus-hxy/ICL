"""Batched GPU policy server over Unix domain socket for E9 Stage 2 closed-loop
LIBERO-Gen Goal Chain evaluation (Default 500, Oracle-GT 331 paired, Shifted 500 paired).

Strict Inference-Time Invariants:
  1. Behavior Memory `(h_i^D, b_i)` is pre-encoded STRICTLY from the provided Demo RGB video
     (`agentview_rgb`, `eye_in_hand_rgb`) using frozen CLIP + frozen E9 Stage 1 Local Encoder
     (`b_i = E_local(V_{i:i+16}^D)` at stride-16 anchors `0, 16, 32, ... < T`).
  2. Online query inference uses ONLY Query current & 1-step past RGB observations `[t-1, t]`.
  3. ZERO Demo actions, ZERO proprioception, ZERO Query future frames are ever accessed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from e5 import ACTION_DIM_10D, ACTION_HORIZON, BPP_CKPT, EMBED_DIM
from e5.policy_server import CHAIN_HDF5_DIR, DEFAULT_PROMPT_SLOTS, get_chain_tasks
from e5.vision import BPPVisualEncoder
from e7.policy_server import ORACLE_GT_EXPERTS_DIR, SHIFT_OFFSET, UnixBatchedServer

from . import BEHAVIOR_STRIDE, DEFAULT_STAGE1_LOCAL_CKPT, NUM_QUERY_TOKENS
from .reader_temperature import set_policy_reader_temperature
from .stage2_direct_model import load_e9_stage2_direct_policy


class E9Stage2BatchedEngine:
    """Pre-constructs Local Behavior Memory `(h_mem, b_mem)` from provided Demo videos once
    on GPU and serves batched closed-loop inference matching `e7/policy_server.py`."""

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
        reader_temperature: float = 1.0,
        behavior_stride: int = BEHAVIOR_STRIDE,
        tasks: Optional[List[str]] = None,
    ):
        if mode not in ("e9_direct", "e9_no_transfer"):
            raise ValueError(
                f"Unsupported mode: {mode} (expected e9_direct|e9_no_transfer)"
            )
        if prompt_mode not in ("default", "default_paired", "shifted", "oracle_gt", "shuffle"):
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        self.mode = mode
        self.prompt_mode = prompt_mode
        self.apply_delta = mode == "e9_direct"
        self.delta_scale = float(delta_scale)
        self.reader_temperature = float(reader_temperature)
        self.behavior_stride = int(behavior_stride)
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
        self.policy, payload = load_e9_stage2_direct_policy(
            checkpoint, stage1_ckpt=stage1_ckpt, device=device
        )
        if self.reader_temperature != 1.0:
            set_policy_reader_temperature(self.policy, self.reader_temperature)
        self.ckpt_epoch = payload.get("epoch")

        self.prompt_memories: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._preencode_all_prompts()

        mem_lens = [v[0].shape[0] for v in self.prompt_memories.values()]
        print(
            f"[E9Stage2Server] Ready: mode={mode} (prediction_mode={self.policy.prediction_mode}, "
            f"apply_delta={self.apply_delta}, delta_scale={self.delta_scale}, "
            f"reader_tau={self.reader_temperature}) "
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
        """Construct `(h_mem, b_mem)` strictly from raw Demo RGB video streams (NO actions/proprio)."""
        av_all = d["obs/agentview_rgb"][:]
        eh_all = d["obs/eye_in_hand_rgb"][:]
        T = len(av_all)
        if len(eh_all) != T:
            raise ValueError(f"Camera frame count mismatch in demo: {T} vs {len(eh_all)}")

        demo_all_feats = self.visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if self.device.type == "cuda" else None,
        )  # (T, 2, 768) float32 on GPU

        with amp_ctx:
            h_mem, b_mem, _ = self.policy.teacher.extract_behavior_memory_from_demo_features(
                demo_all_feats, stride=self.behavior_stride
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
        """Predict 12-step 7D actions strictly from Query `[t-1, t]` RGB + Demo Behavior Memory."""
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
        exec_7d = out["exec_action_7d"]
        return np.ascontiguousarray(exec_7d.float().cpu().numpy(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("e9_direct", "e9_no_transfer"),
        default="e9_direct",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("default", "default_paired", "shifted", "oracle_gt", "shuffle"),
        default="default",
    )
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--reader-temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--coalesce-ms", type=float, default=4.0)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = E9Stage2BatchedEngine(
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        checkpoint=args.checkpoint,
        stage1_ckpt=args.stage1_ckpt,
        device=device,
        use_bf16=not args.no_bf16,
        delta_scale=args.delta_scale,
        reader_temperature=args.reader_temperature,
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
