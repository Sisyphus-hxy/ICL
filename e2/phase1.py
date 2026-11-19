"""Fit a shared RGB-only encoder and decoder and accept on training-side held-out tasks."""

import argparse
from collections import defaultdict
import os
import numpy as np
import torch
from torch.nn import functional as F
from .common import OUT, read, sha, write, table
from .data import Data, balanced, relative_labels
from .model import Autoencoder, loss

P = OUT / "phase1"


def save_pt(path, state):
    tmp = path.with_suffix(".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def metrics(y, t):
    dp = (y[:, :, :3] - t[:, :, :3]).norm(dim=-1)
    dots = (y[:, :, 3:7] * t[:, :, 3:7]).sum(-1).abs().clamp(0, 1)
    return dict(
        pos_mm=dp.mean().item() * 1000,
        rot_deg=torch.rad2deg(2 * dots.acos()).mean().item(),
        grip_acc=((y[:, :, 7] > 0) == (t[:, :, 7] > 0)).float().mean().item(),
        osc6_mae=(y[:, :, 8:14] - t[:, :, 8:14]).abs().mean().item(),
    )


def unnorm(x, ck):
    return x * ck["std"] + ck["mean"]


def event_f1(y, t):
    matched = pred = ref = 0
    for a, b in zip(y, t):
        p = list(np.where(np.diff(a > 0))[0] + 1)
        q = list(np.where(np.diff(b > 0))[0] + 1)
        pred += len(p)
        ref += len(q)
        for i in p:
            ok = [j for j in q if (a[i] > 0) == (b[j] > 0) and abs(j - i) <= 6]
            if ok:
                q.remove(min(ok, key=lambda j: abs(j - i)))
                matched += 1
    return 2 * matched / max(pred + ref, 1)


@torch.inference_mode()
def validation(net, D):
    net.eval()
    rows = []
    outs = []
    for k, ii in D.val.items():
        with amp():
            y = torch.cat(
                [net(D.x[ids]) for ids in np.array_split(ii, (len(ii) + 15) // 16)]
            )
        rows.append((k, loss(y, D.y[ii]).item()))
        outs.append((k, ii, y))
    return float(np.mean([score for _, score in rows])), outs


def plot_paths(path, gt, pred, title):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 5), layout="constrained")
    axes = fig.subplots(2, 3)
    for a, i, label in zip(
        axes.flat,
        [0, 1, 2, 3, 7, 14],
        ["x (m)", "y (m)", "z (m)", "quaternion x", "gripper close", "finger qpos (m)"],
    ):
        a.plot(gt[:, i], label="GT")
        a.plot(pred[:, i], label="decoded")
        a.set_title(label)
    axes[0, 0].legend()
    fig.suptitle(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def curve(path, history):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 3), layout="constrained")
    ax = fig.subplots()
    for k in ("train", "validation"):
        ax.plot([r["step"] for r in history], [r[k] for r in history], label=k)
    ax.set(xlabel="Optimization step", ylabel="Objective")
    ax.legend()
    fig.savefig(path)


@torch.inference_mode()
def sensitivity(net, ck):
    groups = defaultdict(dict)
    for r in read(OUT / "sensitivity/inventory.json"):
        p = OUT / "sensitivity" / r["layout"] / r["label"]
        with np.load(p / "labels.npz") as labels:
            gt = relative_labels(labels["pose"], labels["action"], labels["fingers"], 1)
        with amp():
            z = net.encoder(torch.from_numpy(np.load(p / "features.npy")).cuda()[None])
            y = unnorm(net.decoder(z), ck)[0].cpu().numpy()
        groups[r["layout"]][r["label"]] = (z.flatten().float().cpu().numpy(), y, gt)
    rows = []
    for layout, pair in groups.items():
        n, g = pair["normal"], pair["gentle"]
        direction = np.mean(
            [v["gentle"][0] - v["normal"][0] for k, v in groups.items() if k != layout],
            axis=0,
        )
        d = g[0] - n[0]
        cos = float(
            direction @ d / (np.linalg.norm(direction) * np.linalg.norm(d) + 1e-12)
        )

        def dist(y, t):
            return float(np.linalg.norm(y[:, :3] - t[:, :3], axis=1).mean() * 1000)

        own = (dist(n[1], n[2]) + dist(g[1], g[2])) / 2
        swap = (dist(n[1], g[2]) + dist(g[1], n[2])) / 2
        rows.append(
            dict(
                layout=layout,
                held_layout_pair_correct=int(cos > 0),
                direction_cosine=cos,
                matched_pos_mm=own,
                swapped_pos_mm=swap,
            )
        )
    table(P / "procedure_validation.csv", rows)
    return sum(r["held_layout_pair_correct"] for r in rows), len(rows)


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--steps", type=int, default=1800)
    pa.add_argument("--batch", type=int, default=24)
    a = pa.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    P.mkdir(parents=True, exist_ok=True)
    D = Data()
    net = Autoencoder().cuda()
    ck = dict(
        mean=D.mean,
        std=D.std,
        split_sha=sha(OUT / "split.json"),
        cache_sha=sha(OUT / "cache/inventory.json"),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=0.0006, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=0.00008)
    history = []
    bucket = []
    best = float("inf")
    for step in range(1, a.steps + 1):
        net.train()
        ii = balanced(D.train, a.batch, rng)
        with amp():
            y = net(D.x[ii])
        objective = loss(y, D.y[ii])
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
        opt.step()
        schedule.step()
        bucket.append(objective.item())
        if step % 100 == 0 or step == a.steps:
            score, _ = validation(net, D)
            row = dict(step=step, train=float(np.mean(bucket)), validation=score)
            bucket.clear()
            history.append(row)
            table(P / "loss.csv", history)
            print(row, flush=True)
            if score < best:
                best = score
                save_pt(P / "model.pt", dict(**ck, model=net.state_dict(), step=step))
    curve(P / "loss.png", history)
    net.load_state_dict(torch.load(P / "model.pt", weights_only=True)["model"])
    _, out = validation(net, D)
    baseline = D.y[: D.nt].mean(0)
    baseline[:, 3:7] = F.normalize(baseline[:, 3:7], dim=-1)
    rows = []
    for k, ii, y in out:
        pr = unnorm(y, ck)
        gt = unnorm(D.y[ii], ck)
        base = unnorm(baseline.expand_as(y), ck)
        m = metrics(pr, gt)
        bm = metrics(base, gt)
        m.update({f"mean_plan_{j}": v for j, v in bm.items()})
        m["event_f1"] = event_f1(pr[:, :, 7].cpu().numpy(), gt[:, :, 7].cpu().numpy())
        rows.append(dict(task=k, episodes=len(ii), **m))
        pp = pr[0].cpu().numpy().copy()
        pp[:, 7] = np.where(pp[:, 7] > 0, 1, -1)
        plot_paths(
            P / "reconstruction" / k / "trajectory.png", gt[0].cpu().numpy(), pp, k
        )
    table(P / "validation_tasks.csv", rows)
    by = {}
    for b in ("cf", "bpp"):
        rs = [r for r in rows if r["task"].startswith(b + "/")]
        by[b] = {k: float(np.mean([r[k] for r in rs])) for k in list(rs[0])[2:]}
    correct, n = sensitivity(net, ck)
    passed = (
        all(
            m["pos_mm"] < m["mean_plan_pos_mm"] and m["grip_acc"] > 0.75
            for m in by.values()
        )
        and correct / n >= 0.8
    )
    result = dict(
        passed=passed,
        task_macro=by,
        procedure_correct_layouts=correct,
        procedure_layouts=n,
        best_step=torch.load(P / "model.pt", weights_only=True)["step"],
        model_sha=sha(P / "model.pt"),
        train_tasks=len(D.train),
        validation_tasks=len(D.val),
        train_episodes=D.nt,
        validation_episodes=len(D.ids) - D.nt,
        parameters=sum(p.numel() for p in net.parameters()),
    )
    write(P / "acceptance.json", result)
    print(result, flush=True)
    if not passed:
        raise SystemExit("Phase 1 gate failed")


if __name__ == "__main__":
    main()
