"""Compare same-layout GT-demo eval vs the default eval, per init state.

Default-run per-init outcomes are recovered from the rollout video filenames
(init_states[episode_idx] was the rollout init in main_incontext.py).
"""

import json
import pathlib
import re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "logs" / "eval_results" / "gt_demo_eval"
VDIR = ROOT / "data" / "libero_incontext" / "videos" / "unseen"


def main():
    # (suite, task_slug, init_idx) -> success, from default-eval video names
    default = {}
    for mp4 in VDIR.glob("*/*.mp4"):
        m = re.match(r"rollout_(.+)_ep(\d+)_(success|failure)\.mp4", mp4.name)
        if m:
            default[(mp4.parent.name, m.group(1), int(m.group(2)))] = m.group(3) == "success"

    rows = []
    missing_default = []
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        d = json.loads((RES / f"{suite}.json").read_text())
        for r in d["per_episode_results"]:
            key = (suite, r["task"].replace(" ", "_"), r["init_index"])
            if key not in default:
                missing_default.append(key)
                continue
            rows.append(
                {
                    "suite": suite,
                    "task_id": r["task_id"],
                    "task": r["task"],
                    "init_index": r["init_index"],
                    "demo_episode": r["demo_episode"],
                    "gt_demo_success": r["success"],
                    "default_success": default[key],
                }
            )

    per_task = defaultdict(lambda: {"n": 0, "gt": 0, "def": 0, "flips_to_win": 0, "flips_to_lose": 0})
    for r in rows:
        key = f"{r['suite']}/{r['task_id']}"
        t = per_task[key]
        t["task"] = r["task"]
        t["n"] += 1
        t["gt"] += r["gt_demo_success"]
        t["def"] += r["default_success"]
        t["flips_to_win"] += r["gt_demo_success"] and not r["default_success"]
        t["flips_to_lose"] += (not r["gt_demo_success"]) and r["default_success"]

    print(f"{'suite/task':24s} {'n':>3s} {'GTdemo':>7s} {'default':>8s} {'+win':>5s} {'-lose':>6s}")
    tot = {"n": 0, "gt": 0, "def": 0, "flips_to_win": 0, "flips_to_lose": 0}
    for key in sorted(per_task):
        t = per_task[key]
        for k in tot:
            tot[k] += t[k]
        print(
            f"{key:24s} {t['n']:3d} {t['gt']/t['n']:7.1%} {t['def']/t['n']:8.1%} "
            f"{t['flips_to_win']:5d} {t['flips_to_lose']:6d}  {t['task'][:45]}"
        )
    print(
        f"{'TOTAL':24s} {tot['n']:3d} {tot['gt']/tot['n']:7.1%} {tot['def']/tot['n']:8.1%} "
        f"{tot['flips_to_win']:5d} {tot['flips_to_lose']:6d}"
    )
    print(f"\n(inits without a default-run video: {len(missing_default)})")

    out = {
        "per_task": {k: v for k, v in sorted(per_task.items())},
        "total": tot,
        "episodes": rows,
        "missing_default": missing_default,
    }
    (RES / "comparison.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RES/'comparison.json'}")


if __name__ == "__main__":
    main()
