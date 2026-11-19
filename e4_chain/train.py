"""Stage A action reconstruction, B RGB alignment, C layout-conditioned latent residual."""

import argparse
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from e2.common import sha, write
from e2.phase1 import save_pt
from .archives import OUT
from .cache import SIGNATURE
from .data import Bank
from .model import Chain, MODULES, osc

EPOCHS = {"A": 32, "B": 24, "C": 24}


def objective(result, target):
    prediction, z, teacher, delta = result
    action = F.mse_loss(osc(prediction), osc(target))
    auxiliary = (
        F.mse_loss(prediction[:, :, :7], target[:, :, :7])
        + F.mse_loss(prediction[:, :, 14:], target[:, :, 14:])
    ) / 2
    alignment = F.mse_loss(z.float(), teacher.float())
    total = action + 0.25 * auxiliary + alignment
    if delta is not None:
        total = total + 0.02 * delta.float().square().mean()
    return total


def publish_best(last, best):
    # Also preserves the optimizer state without duplicating checkpoint bytes.
    tmp = best.with_suffix(".link")
    tmp.unlink(missing_ok=True)
    os.link(last, tmp)
    os.replace(tmp, best)


def frozen_hash(net):
    h = hashlib.sha256()
    for name, par in net.named_parameters():
        if not par.requires_grad:
            h.update(name.encode())
            h.update(par.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_gradients(net, stage):
    norms = {}
    for name, module in net.named_children():
        active = name in MODULES[stage]
        grads = []
        for p in module.parameters():
            if p.requires_grad != active or (p.grad is not None) != active:
                raise RuntimeError(f"Gradient/freeze contract failed in {name}")
            if active:
                grads.append(p.grad.detach().float().norm())
        norms[name] = torch.stack(grads).norm().item() if grads else 0.0
        if active and not 0 < norms[name] < float("inf"):
            raise RuntimeError(f"Invalid gradient in {name}")
    return norms


@torch.no_grad()
def evaluate(net, bank, stage, batch_size):
    net.eval()
    ids = bank.ids["val"]
    rng = np.random.default_rng(1024)
    sums = {}
    for start in range(0, len(ids), batch_size):
        batch = bank.batch(ids[start : start + batch_size], "val", rng)
        target = batch[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = net(*batch)
            zero_delta = (
                net.decoder(net.behavior_encoder(batch[1])) if stage == "C" else None
            )
        pred, z, teacher, delta = result
        error = (pred - target) * bank.std
        values = dict(
            loss=objective(result, target),
            osc6_mae=error[:, :, 8:14].abs().mean(),
            late_osc6_mae=error[:, 96:, 8:14].abs().mean(),
            grip_accuracy=((pred[:, :, 7] > 0) == (target[:, :, 7] > 0)).float().mean(),
            position_mm=1000 * error[:, :, :3].norm(dim=-1).mean(),
            latent_rmse=(z.float() - teacher.float()).square().mean().sqrt(),
        )
        if delta is not None:
            values["residual_to_donor_rms"] = delta.float().norm() / (
                z - delta
            ).float().norm().clamp_min(1e-8)
            values["uncorrected_osc6_mae"] = (
                ((zero_delta - target) * bank.std)[:, :, 8:14].abs().mean()
            )
            values["uncorrected_latent_rmse"] = (
                (z.float() - delta.float() - teacher.float()).square().mean().sqrt()
            )
        for key, v in values.items():
            sums[key] = sums.get(key, 0.0) + len(target) * v.item()
    return {k: v / len(ids) for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=MODULES, required=True)
    p.add_argument("--index-dir", type=Path, default=OUT)
    p.add_argument("--run", type=Path, default=OUT / "training")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, default=64, help="per GPU")
    p.add_argument("--lr", type=float, default=0.0004)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    stage = args.stage
    epochs = EPOCHS[stage] if args.epochs is None else args.epochs
    if min(epochs, args.batch_size, args.lr) <= 0:
        p.error("Epochs, batch size and lr must be positive")
    rank, local, world = (
        int(os.environ.get(k, d))
        for k, d in (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1))
    )
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    folder = args.run / stage
    folder.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        if dist.is_initialized():
            stack.callback(dist.destroy_process_group)
        if rank == 0:
            handle = stack.enter_context((folder / "train.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank = Bank(stage, device, args.index_dir)
        model = Chain(len(bank.names)).to(device)
        previous = args.run / ({"B": "A", "C": "B"}.get(stage, "")) / "best.pt"
        lineage = dict(
            data=bank.fingerprint, geometry=bank.names, feature_recipe=SIGNATURE
        )
        parent_sha = None
        if stage != "A":
            ck = torch.load(previous, map_location="cpu", weights_only=True)
            if ck["lineage"] != lineage or ck["stage"] != previous.parent.name:
                raise ValueError(f"Incompatible parent {previous}")
            model.load_state_dict(ck["model"])
            parent_sha = sha(previous)
            del ck
        model.configure(stage)
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=0.01,
        )
        config = dict(
            batch_size=args.batch_size,
            world=world,
            seed=args.seed,
            lr=args.lr,
            parent_sha=parent_sha,
        )
        start, best, history, grad = 0, float("inf"), [], {}
        last = folder / "last.pt"
        if args.resume:
            ck = torch.load(last, map_location=device, weights_only=True)
            if (
                ck["config"] != config
                or ck["lineage"] != lineage
                or ck["stage"] != stage
            ):
                raise ValueError(
                    "Resume checkpoint has different lineage/training settings"
                )
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            start, best, history, grad = (
                ck[k] for k in ("epoch", "best", "history", "gradients")
            )
            if rank == 0 and history[-1]["loss"] == best:
                publish_best(last, folder / "best.pt")
            del ck
        elif last.exists() or (folder / "best.pt").exists():
            raise FileExistsError(
                f"{folder} already contains weights; use --resume or a fresh --run"
            )
        frozen = frozen_hash(model)
        net = DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
        ids = bank.ids["train"]
        if rank == 0:
            print(
                stage,
                "training donors/episodes",
                len(ids),
                "validation",
                len(bank.ids["val"]),
                flush=True,
            )
        for epoch in range(start + 1, epochs + 1):
            model.configure(stage)
            rng = np.random.default_rng([args.seed, epoch])
            order = rng.permutation(ids)
            order = np.concatenate([order, order[: (-len(order)) % world]])[rank::world]
            partner_rng = np.random.default_rng([args.seed, epoch, rank])
            train_loss = torch.zeros(2, device=device)
            for k in range(0, len(order), args.batch_size):
                batch = bank.batch(order[k : k + args.batch_size], "train", partner_rng)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = net(*batch)
                    loss = objective(result, batch[0])
                loss.backward()
                if not grad:
                    grad = audit_gradients(model, stage)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                n = len(batch[0])
                train_loss += torch.stack((loss.detach() * n, loss.new_tensor(n)))
            if world > 1:
                dist.all_reduce(train_loss)
            if rank == 0:
                metrics = evaluate(model, bank, stage, args.batch_size)
                if not all(
                    np.isfinite(v) for v in metrics.values()
                ) or frozen != frozen_hash(model):
                    raise ValueError("Validation/parameter freeze check failed")
                record = dict(
                    epoch=epoch,
                    train_loss=(train_loss[0] / train_loss[1]).item(),
                    **metrics,
                )
                history.append(record)
                improved = metrics["loss"] < best
                best = min(best, metrics["loss"])
                ck = dict(
                    stage=stage,
                    config=config,
                    lineage=lineage,
                    model=model.state_dict(),
                    mean=bank.mean,
                    std=bank.std,
                    epoch=epoch,
                    best=best,
                    gradients=grad,
                    history=history,
                    frozen_sha256=frozen,
                )
                save_pt(last, dict(ck, optimizer=opt.state_dict()))
                if improved:
                    publish_best(last, folder / "best.pt")
                write(
                    folder / "metrics.json",
                    dict(
                        history=history,
                        gradients=grad,
                        frozen_sha256=frozen,
                        parameters={
                            n: sum(p.numel() for p in m.parameters())
                            for n, m in model.named_children()
                        },
                    ),
                )
                print(stage, record, flush=True)
            if world > 1:
                dist.barrier()


if __name__ == "__main__":
    main()
