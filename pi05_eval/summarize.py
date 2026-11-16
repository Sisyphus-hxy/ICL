"""Summarize pi0.5 rollout outputs into per-task CSVs + REPORT.md.

Scans episode result.json files under the two output roots and emits:
  <output_root>/summary.csv          (one row per episode)
  <output_root>/REPORT.md            (per-task success table)
"""
import csv
import dataclasses
import json
import pathlib
import sys


@dataclasses.dataclass
class Args:
    roots: list[str]


def scan(root: pathlib.Path):
    rows = []
    for rj in sorted(root.rglob("result.json")):
        rel = rj.parent.relative_to(root)
        r = json.loads(rj.read_text())
        rows.append({
            "path": str(rel),
            "success": int(bool(r.get("success"))),
            "policy_steps": r.get("policy_steps", ""),
            **{k: r.get(k, "") for k in ("mode", "suite", "benchmark", "task", "task_id", "index")},
            "demo_hdf5": str(rel / "demo.hdf5") if (rj.parent / "demo.hdf5").exists() else "",
        })
    return rows


def main(args: Args):
    for root_s in args.roots:
        root = pathlib.Path(root_s)
        rows = scan(root)
        if not rows:
            print(f"no results under {root}")
            continue
        keys = sorted({k for r in rows for k in r})
        with open(root / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        # per (suite_or_bench, task) stats
        groups = {}
        for r in rows:
            g = (r.get("suite") or r.get("benchmark"), r["task"])
            groups.setdefault(g, []).append(r)
        lines = ["# pi0.5 rollout summary", "",
                 f"root: `{root}`", "",
                 "| group | task | episodes | successes | rate |",
                 "|---|---|---:|---:|---:|"]
        tot_e = tot_s = 0
        for (grp, task), rs in sorted(groups.items()):
            s = sum(r["success"] for r in rs)
            e = len(rs)
            tot_s += s
            tot_e += e
            lines.append(f"| {grp} | {task} | {e} | {s} | {s/e:.2f} |")
        lines += ["", f"**total: {tot_s}/{tot_e} = {tot_s/tot_e:.3f}**", ""]
        (root / "REPORT.md").write_text("\n".join(lines))
        print(f"{root}: {tot_s}/{tot_e} = {tot_s/tot_e:.3f} ({len(groups)} tasks)")


if __name__ == "__main__":
    main(Args(roots=sys.argv[1:]))
