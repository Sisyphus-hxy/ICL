"""Merge GT replay passes (initial run, top-10 retry, 50-init brute force) into
a final per-episode verdict and summary table.

Usage:
    python3 scripts/merge_gt_replay.py
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "logs" / "eval_results" / "gt_replay"
BF_LOG = ROOT / "logs" / "gt_replay_bruteforce.log"

INJECTED = {1, 8, 379, 392, 814, 821, 1278, 1280}


def main():
    base = json.loads((RES / "results.json").read_text())["episodes"]
    retry = {e["episode_index"]: e for e in json.loads((RES / "results_retry.json").read_text())["episodes"]}

    # Parse brute-force logs: lines "ep N: K successful inits [(j, True), ...]"
    bf_success = {1: [2], 8: [23]}  # from the earlier ad-hoc brute-force run
    for bf_log in [BF_LOG, ROOT / "logs" / "bf_a.log", ROOT / "logs" / "bf_b.log", ROOT / "logs" / "bf_c.log"]:
        if not bf_log.exists():
            continue
        for line in bf_log.read_text().splitlines():
            m = re.match(r"ep (\d+): (\d+) successful inits \[", line)
            if m and int(m.group(2)) > 0:
                inits = re.findall(r"\((\d+), True", line)
                bf_success[int(m.group(1))] = [int(j) for j in inits]

    merged = []
    for e in base:
        ep = e["episode_index"]
        out = dict(e)
        out["pass1_success"] = e["success"]
        src = "pass1"
        if not e["success"] and ep in retry:
            r = retry[ep]
            if r["success"]:
                out.update(init_index=r["init_index"], success=True, video=r["video"])
                src = "retry"
        if not out["success"] and ep in bf_success:
            out.update(init_index=bf_success[ep][0], success=True)
            src = "bruteforce"
        out["resolved_by"] = src if out["success"] else "unresolved"
        out["injected"] = ep in INJECTED
        merged.append(out)

    n_ok = sum(m["success"] for m in merged)
    print(f"GT replay final: {n_ok}/{len(merged)} = {n_ok/len(merged):.1%}")
    by_src = {}
    for m in merged:
        by_src.setdefault(m["resolved_by"], 0)
        by_src[m["resolved_by"]] += 1
    print("resolved_by:", by_src)

    per_task = {}
    for m in merged:
        key = f"{m['suite']}/{m['task_id']}"
        agg = per_task.setdefault(key, {"task": m["task"], "eps": 0, "ok": 0, "fail_eps": []})
        agg["eps"] += 1
        agg["ok"] += int(m["success"])
        if not m["success"]:
            agg["fail_eps"].append(m["episode_index"])
    for key, agg in per_task.items():
        agg["rate"] = agg["ok"] / agg["eps"]
    print(json.dumps(per_task, indent=1))

    injected = [m for m in merged if m["injected"]]
    print("\nInjected demos (the ones the policy actually saw):")
    for m in injected:
        print(
            f"  ep={m['episode_index']:5d} {m['suite']:14s} task={m['task_id']} "
            f"init={m['init_index']:2d} success={m['success']} via={m['resolved_by']}"
        )

    out_path = RES / "results_merged.json"
    out_path.write_text(json.dumps({"summary": {"episodes": len(merged), "successes": n_ok,
                                                 "success_rate": n_ok / len(merged),
                                                 "resolved_by": by_src, "per_task": per_task},
                                    "episodes": merged}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
