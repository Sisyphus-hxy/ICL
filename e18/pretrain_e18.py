"""E18 Unified Pure-Video Pretraining Engine (ZERO GT Actions, Parameter & Budget Controlled).

Supports ALL 4 Pillars of E18 with strict scientific discipline:
  - Hard assertion `assert batch["actions_10d"] is None` every single training step.
  - Identical `LocalBehaviorEncoder` (42.55M params) initialized from scratch every run.
  - Matched total pretraining capacity (~80M params) across URBE, CompactLAM, LAPA-VQ, CycleMimic.
  - Exact Relation Scaling controls:
      * Axis 3A: Donor pool size M_pool in {0, 8, 32, 128, 255, 1024 (FIFO queue)}
      * Axis 3B: Exact cross-task coverage p_cov in {0.0, 0.25, 0.50, 1.0} at fixed M_pool=48
                 using identical `TaskGroupedBalancedSampler` (4 tasks x 64 clips/task = 256)
      * Axis 3C: Behavioral diversity at FIXED video count (N_eps=2700: 60x45 vs 150x18 vs 300x9)
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import DEFAULT_DENSE_CACHE_DIR, E5_CACHE_DIR, EMBED_DIM
from e12.eval_suite import _effective_rank_stats
from e15.train_e15 import compute_linear_cka_penalty
from e16 import MOTION_CACHE_DIR
from e16.video_relations import (
    RichPairedBatchSampler,
    VideoMotionLocalBank,
    compute_clip_state_change_targets,
    extract_video_relation_features,
)

from . import CV_GEO_DIR
from .geometry_relations import CVGeoMotionEncoder, LinearCVCCAEncoder, compute_weighted_displacement_loss
from .models import CycleMimicModel, LAPAVQModel, URBEModel


class FixedDiversityVideoBank(VideoMotionLocalBank):
    """Extends `VideoMotionLocalBank` with exact `(num_tasks, eps_per_task)` subsampling for Axis 3C."""

    def __init__(
        self,
        *args: Any,
        diversity_tasks: Optional[int] = None,
        diversity_eps_per_task: Optional[int] = None,
        diversity_seed: int = 1803,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if diversity_tasks is not None and diversity_eps_per_task is not None:
            rng = np.random.default_rng(diversity_seed)
            by_task: Dict[int, List[int]] = defaultdict(list)
            for ep_idx, row in enumerate(self.episodes):
                if row["fold"] == "train":
                    by_task[int(self.ep_task_ids_np[ep_idx])].append(ep_idx)
            eligible_tasks = sorted([tid for tid, eps in by_task.items() if len(eps) >= diversity_eps_per_task])
            if len(eligible_tasks) < diversity_tasks:
                raise RuntimeError(
                    f"Requested {diversity_tasks} tasks with >= {diversity_eps_per_task} eps, "
                    f"only {len(eligible_tasks)} eligible!"
                )
            chosen_tasks = sorted(rng.choice(eligible_tasks, size=diversity_tasks, replace=False).tolist())
            kept_eps: set = set()
            for tid in chosen_tasks:
                arr = np.array(by_task[tid], dtype=np.int64)
                rng.shuffle(arr)
                kept_eps.update(arr[:diversity_eps_per_task].tolist())
            mask = np.array([int(ep) in kept_eps for ep in self.train_episodes], dtype=bool)
            self.train_episodes = self.train_episodes[mask]
            self.train_frames = self.train_frames[mask]
            self.kept_train_episode_ids = sorted(kept_eps)
            self.kept_diversity_tasks = chosen_tasks


class TaskGroupedBalancedSampler:
    """Axis 3B sampler: puts exactly `tasks_per_batch=4` tasks x `clips_per_task=64` (=256) in each batch.

    Ensures EVERY query has ~62 same-task diff-ep donors AND 192 cross-task diff-ep donors,
    so exact cross-task fraction `p_cov in {0.0, 0.25, 0.50, 1.0}` at fixed `M_pool=48` has ZERO
    fallback and IDENTICAL batch statistics across all coverage runs!
    """

    def __init__(
        self,
        bank: VideoMotionLocalBank,
        *,
        tasks_per_batch: int = 4,
        seed: int = 42,
        include_actions: bool = False,
    ):
        self.bank = bank
        self.tasks_per_batch = int(tasks_per_batch)
        self.seed = int(seed)
        self.include_actions = bool(include_actions)

        task_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, ep in enumerate(bank.train_episodes):
            tid = int(bank.ep_task_ids_np[int(ep)])
            task_to_indices[tid].append(idx)

        self.task_ids = sorted([t for t, l in task_to_indices.items() if len(l) >= 64])
        self.task_arrays = {t: np.array(task_to_indices[t], dtype=np.int64) for t in self.task_ids}

    def iter_batches(self, epoch: int, batch_size: int) -> Iterator[Dict[str, Optional[torch.Tensor]]]:
        assert batch_size % self.tasks_per_batch == 0
        per_task = batch_size // self.tasks_per_batch
        rng = np.random.default_rng([self.seed, int(epoch), 18303])
        steps = math.ceil(self.bank.num_train_samples() / batch_size)
        for _ in range(steps):
            chosen_t = rng.choice(self.task_ids, size=self.tasks_per_batch, replace=False)
            chunks = []
            for tid in chosen_t:
                arr = self.task_arrays[int(tid)]
                sel = rng.choice(arr, size=per_task, replace=(len(arr) < per_task))
                chunks.append(sel)
            batch_idx = np.concatenate(chunks, axis=0)
            # Keep paired ordering for displacement losses (adjacent pairs share task)
            eps = self.bank.train_episodes[batch_idx]
            frs = self.bank.train_frames[batch_idx]
            yield self.bank.gather_rich_batch(eps, frs, include_actions=self.include_actions)


@torch.inference_mode()
def evaluate_e18_video_validation(
    model: nn.Module,
    bank: VideoMotionLocalBank,
    device: torch.device,
    batch_size: int = 256,
    max_samples: int = 2048,
) -> Dict[str, Any]:
    model.eval()
    n = min(max_samples, bank.num_val_samples())
    latents = []
    dv_sse = dv_sst = 0.0
    dv_cnt = 0
    vq_perps = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = bank.gather_rich_batch(bank.val_episodes[s:e], bank.val_frames[s:e], include_actions=False)
        assert batch["actions_10d"] is None
        b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])
        tgts = compute_clip_state_change_targets(batch["clip_rgb"], batch["clip_pad_mask"])
        if isinstance(model, LAPAVQModel):
            b_q, _, vqs = model.quantize_b(b_t)
            pred_dv, _ = model.fdm(tgts["v0"], b_q.reshape(b_t.shape[0], -1))
            vq_perps.append(float(vqs["vq_perplexity"].item()))
        else:
            pred_dv, _ = model.predict_state_change_from_b(tgts["v0"], b_t)
        w = tgts["seg_valid"][:, :, None, None]
        dv_sse += float(((pred_dv - tgts["delta_v_seg"]).square() * w).sum().item())
        dv_sst += float((tgts["delta_v_seg"].square() * w).sum().item())
        dv_cnt += int(w.sum().item())
        latents.append(b_t.float().cpu().numpy())

    Z = np.concatenate(latents, axis=0)
    op_d = getattr(model, "op_dim", 384)
    op_d_eff = min(op_d, 384)
    er_full = _effective_rank_stats(Z.reshape(len(Z), -1), device=device)
    er_op = _effective_rank_stats(Z[:, :, :op_d_eff].reshape(len(Z), -1), device=device)
    er_res = _effective_rank_stats(Z[:, :, op_d_eff:].reshape(len(Z), -1), device=device)
    out = {
        "val_samples": int(n),
        "val_dv_mse": float(dv_sse / max(dv_cnt * 2 * 768, 1)),
        "val_dv_r2": float(1.0 - dv_sse / max(dv_sst, 1e-8)),
        "val_b_rms": float(np.sqrt(np.mean(Z ** 2))),
        "val_op_rms": float(np.sqrt(np.mean(Z[:, :, :op_d_eff] ** 2))),
        "val_res_rms": float(np.sqrt(np.mean(Z[:, :, op_d_eff:] ** 2))),
        "erank_full": float(er_full["erank_energy"]),
        "erank_op": float(er_op["erank_energy"]),
        "erank_res": float(er_res["erank_energy"]),
    }
    if vq_perps:
        out["val_vq_perplexity"] = float(np.mean(vq_perps))
    return out


def _save_snapshot(model: nn.Module, config: Dict[str, Any], step: int, val_m: Dict[str, Any], out_path: Path) -> None:
    payload: Dict[str, Any] = {
        "stage": f"e18_{config['run_name']}",
        "step": int(step),
        "local_encoder": {k: v.detach().cpu() for k, v in model.local_encoder.state_dict().items()},
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "val_metrics": val_m,
    }
    if hasattr(model, "reuse_module") and model.reuse_module is not None:
        payload["reuse_module"] = {
            k: v.detach().cpu()
            for k, v in model.reuse_module.state_dict().items()
            if not k.startswith("queue_")
        }
    atomic_torch_save(payload, out_path)


def build_e18_model(args: argparse.Namespace) -> nn.Module:
    v = args.variant
    if v == "lapa_vq":
        return LAPAVQModel(codebook_size=args.vq_codebook_size, code_dim=args.vq_code_dim, fdm_hidden=1920)
    if v == "cyclemimic":
        return CycleMimicModel(op_dim=384, fdm_hidden=1408, compact_bottleneck=True)
    if v == "compact_lam":
        return URBEModel(compact_manifold=True, op_dim=384, use_reuse=False, predict_motion=False)
    if v == "pure_reuse_min":
        return URBEModel(
            compact_manifold=True, op_dim=384, use_reuse=True, top_k_donors=args.top_k_donors,
            top1_reuse_weight=0.40, init_temp=0.07, coverage_mode=args.coverage_mode,
            donor_pool_size=args.donor_pool_size, memory_queue_size=args.memory_queue_size,
            exact_cross_task_frac=args.exact_cross_task_frac, predict_motion=False,
        )
    if v in ("reuse_cycle", "urbe_full", "g2_crossview_geo"):
        return URBEModel(
            compact_manifold=True, op_dim=384, use_reuse=True, top_k_donors=args.top_k_donors,
            top1_reuse_weight=0.40, init_temp=0.07, coverage_mode=args.coverage_mode,
            donor_pool_size=args.donor_pool_size, memory_queue_size=args.memory_queue_size,
            exact_cross_task_frac=args.exact_cross_task_frac, predict_motion=bool(args.predict_motion),
        )
    if v == "g3_indep_mot_subspace":
        return URBEModel(
            compact_manifold=True, op_dim=384, use_reuse=True, top_k_donors=args.top_k_donors,
            top1_reuse_weight=0.40, init_temp=0.07, coverage_mode=args.coverage_mode,
            predict_motion=False, indep_motion_subspace=True, mot_subspace_dim=128,
        )
    if v == "no_opres_split":
        return URBEModel(
            compact_manifold=True, op_dim=768, use_reuse=True, top_k_donors=args.top_k_donors,
            top1_reuse_weight=0.40, init_temp=0.07, coverage_mode=args.coverage_mode,
            predict_motion=False,
        )
    if v == "soft_reuse_only":
        return URBEModel(
            compact_manifold=True, op_dim=384, use_reuse=True, top_k_donors=None,
            top1_reuse_weight=0.0, init_temp=0.12, coverage_mode=args.coverage_mode,
            predict_motion=False,
        )
    if v == "nocompact_sharp":
        return URBEModel(
            compact_manifold=False, op_dim=384, use_reuse=True, top_k_donors=args.top_k_donors,
            top1_reuse_weight=0.40, init_temp=0.07, coverage_mode=args.coverage_mode,
            predict_motion=False,
        )
    raise ValueError(f"Unknown E18 variant: {v}")


def run_e18_pretraining(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    need_motion = args.variant in ("g2_crossview_geo", "g3_indep_mot_subspace") or bool(args.predict_motion)
    t0 = time.monotonic()
    bank = FixedDiversityVideoBank(
        e5_cache_dir=args.e5_cache_dir,
        dense_cache_dir=args.dense_cache_dir,
        motion_cache_dir=args.motion_cache_dir if need_motion else None,
        device=device,
        resident_on_device=bool(args.gpu_resident_cache),
        seed=args.seed,
        train_episode_fraction=args.train_episode_fraction,
        subset_seed=args.seed,
        diversity_tasks=args.diversity_tasks,
        diversity_eps_per_task=args.diversity_eps_per_task,
        diversity_seed=1803,
    )
    print(
        f"[E18 Pretrain:{args.run_name}] ZERO-ACTION bank ready (variant={args.variant}, "
        f"eps={len(bank.kept_train_episode_ids):,}, starts={bank.num_train_samples():,}, "
        f"pool={args.donor_pool_size}, queue={args.memory_queue_size}, cov_frac={args.exact_cross_task_frac}) "
        f"in {time.monotonic()-t0:.1f}s",
        flush=True,
    )

    cv_geo_enc: Optional[nn.Module] = None
    if args.variant in ("g2_crossview_geo", "g3_indep_mot_subspace"):
        cv_ckpt = Path(args.cv_geo_ckpt)
        if not cv_ckpt.is_file():
            raise FileNotFoundError(f"CV-Geo encoder not found: {cv_ckpt}")
        cv_pl = torch.load(cv_ckpt, map_location="cpu", weights_only=False)
        if "hidden_dim" in cv_pl["config"]:
            cv_geo_enc = CVGeoMotionEncoder(
                **{k: cv_pl["config"][k] for k in ("in_dim", "hidden_dim", "out_dim")}
            ).to(device).eval()
        else:
            cv_geo_enc = LinearCVCCAEncoder(**cv_pl["config"]).to(device).eval()
        cv_geo_enc.load_state_dict(cv_pl["state_dict"], strict=True)
        for p in cv_geo_enc.parameters():
            p.requires_grad_(False)

    seed_everything(args.seed)
    model = build_e18_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay
    )
    total_steps = int(args.total_steps)
    warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMAModel(
        model.parameters(),
        decay=args.ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    if args.exact_cross_task_frac is not None:
        sampler: Any = TaskGroupedBalancedSampler(bank, tasks_per_batch=4, seed=args.seed, include_actions=False)
    else:
        sampler = RichPairedBatchSampler(bank, within_ep_frac=0.5, seed=args.seed, include_actions=False)

    uses_task_meta = (args.exact_cross_task_frac is not None) or (
        args.coverage_mode in ("strict_cross_task", "same_task_only")
    )
    save_steps = sorted({int(x) for x in args.save_steps.split(",") if x.strip()}) if args.save_steps else []

    config = {
        "regime": "pure_video_pretraining_task_meta_controlled" if uses_task_meta else "pure_video_only_pretraining_zero_actions_zero_task_meta",
        "uses_task_metadata": bool(uses_task_meta),
        "action_supervision": False,
        "init_mode": "from_scratch_zero_action_init",
        "run_name": args.run_name,
        "variant": args.variant,
        "lambda_vid": float(args.lambda_vid),
        "lambda_reuse": float(args.lambda_reuse),
        "lambda_orth": float(args.lambda_orth),
        "predict_motion": bool(args.predict_motion),
        "coverage_mode": args.coverage_mode,
        "top_k_donors": args.top_k_donors,
        "donor_pool_size": args.donor_pool_size,
        "memory_queue_size": int(args.memory_queue_size),
        "exact_cross_task_frac": args.exact_cross_task_frac,
        "diversity_tasks": args.diversity_tasks,
        "diversity_eps_per_task": args.diversity_eps_per_task,
        "train_episode_fraction": float(args.train_episode_fraction),
        "num_train_episodes": len(bank.kept_train_episode_ids),
        "num_train_starts": bank.num_train_samples(),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "encoder_params": sum(p.numel() for p in model.local_encoder.parameters()),
        "total_pretrain_params": sum(p.numel() for p in model.parameters()),
    }
    atomic_write_json(config, run_dir / "config.json")
    print(
        f"[E18 Pretrain:{args.run_name}] Params: encoder={config['encoder_params']/1e6:.2f}M, "
        f"total_pretrain={config['total_pretrain_params']/1e6:.2f}M | seed={args.seed}",
        flush=True,
    )

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def _inf_batches() -> Iterator[Dict[str, Optional[torch.Tensor]]]:
        ep = 1
        while True:
            yield from sampler.iter_batches(ep, args.batch_size)
            ep += 1

    batch_iter = _inf_batches()
    step = 0
    best_dv_mse = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []
    wt0 = time.monotonic()
    acc_lam = acc_vid = acc_reu = acc_tot = 0.0
    acc_n = 0

    use_cycle_flag = args.variant not in ("pure_reuse_min", "compact_lam")
    use_equiv_flag = args.variant not in ("pure_reuse_min", "reuse_cycle", "compact_lam")

    while step < total_steps:
        model.train()
        batch = next(batch_iter)
        assert batch["actions_10d"] is None, "Action tensor leaked into Pure Video-Only Pretraining!"
        step += 1

        warm = (step / float(args.rel_warmup_steps)) if (args.rel_warmup_steps > 0 and step <= args.rel_warmup_steps) else 1.0
        cur_lvid = float(args.lambda_vid) * warm
        cur_lreu = float(args.lambda_reuse) * warm

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])

        vid_loss = b_t.new_zeros(())
        reu_loss = b_t.new_zeros(())
        orth_loss = b_t.new_zeros(())
        rel_stats: Dict[str, torch.Tensor] = {}

        if isinstance(model, LAPAVQModel):
            lam_loss, rel_stats = model.compute_loss(b_t, batch["clip_rgb"], batch["clip_pad_mask"])
            total_loss = lam_loss
        elif isinstance(model, CycleMimicModel):
            lam_loss, rel_stats = model.compute_loss(
                b_t, batch["clip_rgb"], batch["clip_pad_mask"], batch["ep_ids"], do_cycle=(step > 500)
            )
            total_loss = lam_loss
        else:
            assert isinstance(model, URBEModel)
            lam_loss, lam_stats = model.compute_lam_loss(
                b_t, batch["clip_rgb"], batch["clip_pad_mask"], batch["clip_motion"]
            )
            rel_stats.update(lam_stats)

            # Pillar 2: Geometry-Aware Cross-View Motion Relation (`g2`) OR Isolated 2D Motion Subspace (`g3`)
            if args.variant == "g2_crossview_geo" and cur_lvid > 0.0:
                assert cv_geo_enc is not None
                with torch.no_grad():
                    u_cv, agree = cv_geo_enc.encode_canonical(batch["clip_motion"], batch["clip_pad_mask"])
                    pw = torch.sigmoid((agree - 0.15) / 0.20)
                z_op = b_t[:, :, : model.op_dim]
                vid_loss, vs = compute_weighted_displacement_loss(z_op, u_cv, pair_weights=pw)
                rel_stats.update({f"geo_{k}": v for k, v in vs.items()})
            elif args.variant == "g3_indep_mot_subspace" and cur_lvid > 0.0:
                u_mot = extract_video_relation_features(
                    "motion_flow", batch["clip_rgb"], batch["clip_pad_mask"], batch["clip_motion"]
                )
                z_mot = b_t[:, :, model.reuse_dim : model.op_dim]
                vid_loss, vs = compute_weighted_displacement_loss(z_mot, u_mot, pair_weights=None)
                rel_stats.update({f"mot_{k}": v for k, v in vs.items()})

            # Unified Sharp Cross-Demo Reuse
            if model.reuse_module is not None and cur_lreu > 0.0:
                do_cyc = use_cycle_flag and (step > 500)
                do_eq = use_equiv_flag and (step > 500)
                z_reuse = b_t[:, :, : model.reuse_dim]
                reu_loss, rs = model.reuse_module.forward_loss(
                    z_reuse,
                    z_reuse,
                    batch["clip_rgb"],
                    batch["clip_pad_mask"],
                    batch["ep_ids"],
                    task_ids=batch["task_ids"] if uses_task_meta else None,
                    clip_motion=batch["clip_motion"] if model.reuse_module.forward_pred.predict_motion else None,
                    encoder_fn=model.encode_local_behavior if do_cyc else None,
                    extract_pred_fn=(lambda b: b[:, :, : model.reuse_dim]),
                    use_cycle=do_cyc,
                    use_equiv=do_eq,
                )
                rel_stats.update(rs)

            # Orthogonal isolation for G3 independent motion subspace (stop-grad on z_reuse so motion NEVER corrupts z_reuse!)
            if args.variant == "g3_indep_mot_subspace" and args.lambda_orth > 0.0 and warm > 0.0:
                z_reu_det = b_t[:, :, : model.reuse_dim].detach().float().reshape(b_t.shape[0], -1)
                z_mot_f = b_t[:, :, model.reuse_dim : model.op_dim].float().reshape(b_t.shape[0], -1)
                orth_loss = compute_linear_cka_penalty(z_reu_det, z_mot_f)
                rel_stats["cka_reuse_mot"] = orth_loss.detach()

            total_loss = lam_loss + cur_lvid * vid_loss + cur_lreu * reu_loss + (float(args.lambda_orth) * warm) * orth_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        ema.step(model.parameters())

        acc_lam += float(lam_loss.item())
        acc_vid += float(vid_loss.item())
        acc_reu += float(reu_loss.item())
        acc_tot += float(total_loss.item())
        acc_n += 1

        need_eval = (step % args.eval_every == 0) or (step == total_steps) or (step in save_steps)
        if need_eval:
            dt = time.monotonic() - wt0
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            val_m = evaluate_e18_video_validation(model, bank, device)
            if val_m["val_dv_mse"] < best_dv_mse or best_step == 0:
                best_dv_mse = val_m["val_dv_mse"]
                best_step = step
                _save_snapshot(model, config, step, val_m, run_dir / "best.pt")
            if step in save_steps and step != total_steps:
                _save_snapshot(model, config, step, val_m, run_dir / f"step_{step:05d}_ema.pt")
            if step == total_steps:
                _save_snapshot(model, config, step, val_m, run_dir / "last_ema.pt")
            ema.restore(model.parameters())

            rec = {
                "step": step,
                "train": {
                    "lam_loss": acc_lam / max(acc_n, 1),
                    "vid_loss": acc_vid / max(acc_n, 1),
                    "reuse_loss": acc_reu / max(acc_n, 1),
                    "total_loss": acc_tot / max(acc_n, 1),
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "steps_per_sec": round(acc_n / max(dt, 1e-6), 2),
                    "rel_stats": {k: float(v.item()) for k, v in rel_stats.items()},
                },
                "val": val_m,
            }
            history.append(rec)
            atomic_write_json(
                {"config": config, "best_step": best_step, "best_dv_mse": best_dv_mse, "history": history},
                run_dir / "metrics.json",
            )
            extra = f" perp={val_m['val_vq_perplexity']:.1f}" if "val_vq_perplexity" in val_m else ""
            print(
                f"[E18:{args.run_name} step {step:5d}/{total_steps}] "
                f"lam={rec['train']['lam_loss']:.4f} vid={rec['train']['vid_loss']:.4f} reu={rec['train']['reuse_loss']:.4f} | "
                f"dv_r2={val_m['val_dv_r2']:.4f} erank(f/op/rs)={val_m['erank_full']:.1f}/{val_m['erank_op']:.1f}/{val_m['erank_res']:.1f}{extra} | "
                f"{rec['train']['steps_per_sec']:.1f}it/s",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_lam = acc_vid = acc_reu = acc_tot = 0.0
            acc_n = 0

    print(f"[E18:{args.run_name}] Done. Saved to {run_dir / 'last_ema.pt'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=[
            "compact_lam",
            "pure_reuse_min",
            "reuse_cycle",
            "urbe_full",
            "no_opres_split",
            "soft_reuse_only",
            "nocompact_sharp",
            "g2_crossview_geo",
            "g3_indep_mot_subspace",
            "lapa_vq",
            "cyclemimic",
        ],
    )
    p.add_argument("--lambda-vid", type=float, default=0.0)
    p.add_argument("--lambda-reuse", type=float, default=0.25)
    p.add_argument("--lambda-orth", type=float, default=0.05)
    p.add_argument("--rel-warmup-steps", type=int, default=500)
    p.add_argument("--predict-motion", action="store_true")
    p.add_argument("--coverage-mode", type=str, default="cross_demo_unmasked")
    p.add_argument("--top-k-donors", type=int, default=16)
    p.add_argument("--donor-pool-size", type=int, default=None)
    p.add_argument("--memory-queue-size", type=int, default=0)
    p.add_argument("--exact-cross-task-frac", type=float, default=None)
    p.add_argument("--diversity-tasks", type=int, default=None)
    p.add_argument("--diversity-eps-per-task", type=int, default=None)
    p.add_argument("--train-episode-fraction", type=float, default=1.0)
    p.add_argument("--vq-codebook-size", type=int, default=256)
    p.add_argument("--vq-code-dim", type=int, default=128)
    p.add_argument("--cv-geo-ckpt", type=Path, default=CV_GEO_DIR / "cv_geo_encoder.pt")
    p.add_argument("--save-steps", type=str, default="16092,32184")
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--motion-cache-dir", type=Path, default=MOTION_CACHE_DIR)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=32184)
    p.add_argument("--eval-every", type=int, default=4023)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_e18_pretraining(parse_args())
