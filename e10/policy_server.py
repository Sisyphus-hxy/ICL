"""Batched GPU policy server over Unix domain socket for E10 closed-loop
LIBERO-Gen Goal Chain evaluation (Default 500, Shifted 500 paired, Oracle-GT 331 paired).

Identical online interface & invariants to `e9/policy_server_stage2.py` (RGB-only, zero
demo actions/proprio, zero query future frames).
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

from . import BEHAVIOR_STRIDE, E9_STAGE1_CKPT, E9_STAGE2_CKPT, LATENT_TOKENS
from .model import load_e10_policy


class E10BatchedEngine:
    """Pre-constructs frozen E9 Local Behavior Memory `(h_mem, b_mem)` from provided Demo
    videos and serves E10 Low-Rank Transfer closed-loop inference."""

    def __init__(
        self,
        *,
        mode: str,
        prompt_mode: str,
        checkpoint: Path,
        stage1_ckpt: Optional[Path] = None,
        e9_stage2_ckpt: Optional[Path] = None,
        device: torch.device,
        use_bf16: bool = True,
        delta_scale: float = 1.0,
        behavior_stride: int = BEHAVIOR_STRIDE,
        tasks: Optional[List[str]] = None,
    ):
        if mode not in ("e10_lowrank", "e10_no_transfer"):
            raise ValueError(f"Unsupported mode: {mode}")
        if prompt_mode not in ("default", "default_paired", "shifted", "oracle_gt", "shuffle"):
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        self.mode = mode
        self.prompt_mode = prompt_mode
        self.apply_delta = mode == "e10_lowrank"
        self.delta_scale = float(delta_scale)
        self.behavior_stride = int(behavior_stride)
        self.device = device
        self.use_bf16 = use_bf16 and (device.type == "cuda")
        self.tasks = tasks if tasks is not None else get_chain_tasks()

        t0 = time.monotonic()
        self.visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
        self.policy, payload = load_e10_policy(
            checkpoint, stage1_ckpt=stage1_ckpt, e9_stage2_ckpt=e9_stage2_ckpt, device=device
        )
        self.ckpt_epoch = payload.get("epoch")

        self.prompt_memories: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._preencode_all_prompts()

        mem_lens = [v[0].shape[0] for v in self.prompt_memories.values()]
        print(
            f"[E10Server] Ready: mode={mode} variant={payload.get('variant')} "
            f"(transfer_mode={self.policy.transfer_mode}, K={self.policy.rank_k}) "
            f"prompt_mode={prompt_mode} ckpt={checkpoint} ep={self.ckpt_epoch} "
            f"prompts={len(self.prompt_memories)} (entries min={min(mem_lens)}, max={max(mem_lens)}) "
            f"device={device} bf16={self.use_bf16} startup={time.monotonic() - t0:.1f}s",
            flush=True,
        )

    @torch.inference_mode()
    def _encode_single_demo_video(self, d: h5py.Group, conv: str, amp_ctx) -> Tuple[torch.Tensor, torch.Tensor]:
        av_all = d["obs/agentview_rgb"][:]
        eh_all = d["obs/eye_in_hand_rgb"][:]
        demo_all_feats = self.visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if self.device.type == "cuda" else None,
        )
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
                with h5py.File(CHAIN_HDF5_DIR / f"{task}_demo.hdf5", "r") as f:
                    conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                    for di in range(DEFAULT_PROMPT_SLOTS):
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_video(
                            f[f"data/demo_{di}"], conv, amp_ctx
                        )
        elif self.prompt_mode == "shifted":
            for ti, task in enumerate(self.tasks):
                with h5py.File(CHAIN_HDF5_DIR / f"{task}_demo.hdf5", "r") as f:
                    conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                    for di in range(DEFAULT_PROMPT_SLOTS):
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_video(
                            f[f"data/demo_{SHIFT_OFFSET + di}"], conv, amp_ctx
                        )
        elif self.prompt_mode == "shuffle":
            n_tasks = len(self.tasks)
            for ti in range(n_tasks):
                donor_task = self.tasks[(ti + 1) % n_tasks]
                with h5py.File(CHAIN_HDF5_DIR / f"{donor_task}_demo.hdf5", "r") as f:
                    conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
                    for di in range(DEFAULT_PROMPT_SLOTS):
                        self.prompt_memories[(ti, di)] = self._encode_single_demo_video(
                            f[f"data/demo_{di}"], conv, amp_ctx
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
                        self.prompt_memories[(ti, qi)] = self._encode_single_demo_video(
                            f["data/demo_0"], conv, amp_ctx
                        )

    @torch.inference_mode()
    def predict_batch(self, keys: List[Tuple[int, int]], seeds: List[int], rgb_batch: np.ndarray) -> np.ndarray:
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
        h_mem = torch.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), device=dev, dtype=torch.float32)
        b_mem = torch.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), device=dev, dtype=torch.float32)
        pad_mask = torch.ones((B, max_m), device=dev, dtype=torch.bool)
        for bi, (hm, bm) in enumerate(mems):
            m = hm.shape[0]
            h_mem[bi, :m] = hm
            b_mem[bi, :m] = bm
            pad_mask[bi, :m] = False

        noises = []
        for s in seeds:
            gen = torch.Generator(device="cpu").manual_seed(int(s) & 0x7FFFFFFF)
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
                pad_mask,
                q_feats,
                apply_delta=self.apply_delta,
                delta_scale=self.delta_scale,
                init_noise=init_noise,
            )
        return np.ascontiguousarray(out["exec_action_7d"].float().cpu().numpy(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("e10_lowrank", "e10_no_transfer"), default="e10_lowrank")
    parser.add_argument("--prompt-mode", choices=("default", "default_paired", "shifted", "oracle_gt", "shuffle"), default="default")
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, default=None)
    parser.add_argument("--e9-stage2-ckpt", type=Path, default=None)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--coalesce-ms", type=float, default=4.0)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = E10BatchedEngine(
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        checkpoint=args.checkpoint,
        stage1_ckpt=args.stage1_ckpt,
        e9_stage2_ckpt=args.e9_stage2_ckpt,
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
    server.serve(ready_file=args.ready_file)


if __name__ == "__main__":
    main()
