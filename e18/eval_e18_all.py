"""E18 Comprehensive Evaluation Suite:
  1. E15 Full 6-Axis Evaluation (Fidelity, Editability, Composability, Predictability, Robustness, Zero-Leakage Visual ICL)
  2. E17 Pillar 1 Relation Authenticity & Counterfactual Intervention Audit (for reuse-equipped models)
  3. E17 Pillar 2 Emergent Factors & Subgoal Boundary Detection AUC
  4. Exact Paired Per-Task Statistical Tests + 10,000-Resample 95% Bootstrap CIs
  5. Full-5799 (311-task) Unified Label Efficiency Audit (1%, 10%, 100% frozen decoders)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.train import atomic_write_json
from e12.eval_suite import ValClipCache
from e13.exp3_factor_composition import _select_swap_pairs
from e15.eval_e15 import evaluate_single_model as eval_e15_model
from e16.analyze_discovered_relations import _gather_beyond_e12_targets
from e17.exp1_relation_authenticity import audit_single_model_authenticity
from e17.exp2_emergent_factors import analyze_emergent_factors
from e17.stats_paired import collect_per_task_arrays, paired_bootstrap_stats

from . import OUTPUT_ROOT


def run_eval_manifest(
    manifest_path: Path,
    *,
    do_e15: bool = True,
    do_auth: bool = False,
    do_emergent: bool = False,
    do_paired: bool = False,
    paired_comparisons: Optional[List[Tuple[str, str]]] = None,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest: Dict[str, Dict[str, Any]] = json.loads(manifest_path.read_text())
    keys = list(manifest.keys())
    my_keys = [k for i, k in enumerate(keys) if (i % num_shards) == shard_idx]

    eval_root = OUTPUT_ROOT / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=42)

    beyond = None
    swap_pairs = None
    if do_emergent:
        beyond = _gather_beyond_e12_targets(cache)
        non_tail = np.flatnonzero(~beyond["is_tail"])
        swap_pairs = _select_swap_pairs(
            beyond["val_a10"][non_tail],
            beyond["val_a7"][non_tail],
            cache.val_task_ids[non_tail],
            num_pairs=256,
            seed=42,
        )

    for key in my_keys:
        info = manifest[key]
        dec_ckpt = Path(info["dec_ckpt"]) if info.get("dec_ckpt") else None
        enc_ckpt = Path(info["enc_ckpt"]) if info.get("enc_ckpt") else None
        op_dim = int(info.get("op_dim", 384))
        eval_op_dim = min(op_dim, 384)

        if do_e15 and dec_ckpt is not None and dec_ckpt.is_file():
            out_dir = eval_root / key
            out_json = out_dir / "eval_e15.json"
            if not out_json.is_file():
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"[E18 Eval-6Axis] Running {key}...", flush=True)
                eval_e15_model(
                    model_key=key,
                    ckpt_path=dec_ckpt,
                    out_dir=out_dir,
                    device=device,
                    op_dim=eval_op_dim,
                )

        if do_auth and enc_ckpt is not None and enc_ckpt.is_file() and info.get("has_reuse", False):
            auth_dir = OUTPUT_ROOT / "authenticity"
            auth_dir.mkdir(parents=True, exist_ok=True)
            out_a = auth_dir / f"{key}.json"
            if not out_a.is_file():
                print(f"[E18 Authenticity] Auditing {key}...", flush=True)
                res_a = audit_single_model_authenticity(
                    name=key,
                    ckpt_path=enc_ckpt,
                    regime_tag=info.get("regime_tag", "PureVideo (E18)"),
                    cache=cache,
                    device=device,
                    decoder_ckpt=dec_ckpt if (dec_ckpt and dec_ckpt.is_file()) else None,
                    op_dim=op_dim,
                )
                atomic_write_json(res_a, out_a)

        if do_emergent and dec_ckpt is not None and dec_ckpt.is_file():
            em_dir = OUTPUT_ROOT / "emergent"
            em_dir.mkdir(parents=True, exist_ok=True)
            out_e = em_dir / f"{key}.json"
            if not out_e.is_file():
                print(f"[E18 Emergent] Analyzing {key}...", flush=True)
                assert beyond is not None and swap_pairs is not None
                res_e = analyze_emergent_factors(
                    name=key,
                    ckpt_path=dec_ckpt,
                    cache=cache,
                    beyond=beyond,
                    swap_pairs=swap_pairs,
                    device=device,
                    op_dim=eval_op_dim,
                )
                atomic_write_json(res_e, out_e)

        if do_paired and dec_ckpt is not None and dec_ckpt.is_file():
            ps_dir = OUTPUT_ROOT / "paired_stats" / "per_model"
            ps_dir.mkdir(parents=True, exist_ok=True)
            out_p = ps_dir / f"{key}.json"
            if not out_p.is_file():
                print(f"[E18 PairedArrays] Collecting {key}...", flush=True)
                arrs = collect_per_task_arrays(
                    key, dec_ckpt, cache, device, op_dim=eval_op_dim
                )
                atomic_write_json(arrs, out_p)

    # Aggregate paired bootstrap statistics when requested on shard 0
    if paired_comparisons and shard_idx == 0:
        ps_dir = OUTPUT_ROOT / "paired_stats" / "per_model"
        loaded: Dict[str, Dict[str, Any]] = {}
        for p in sorted(ps_dir.glob("*.json")):
            loaded[p.stem] = json.loads(p.read_text())

        comps_out: Dict[str, Any] = {}
        for b_key, a_key in paired_comparisons:
            if b_key not in loaded or a_key not in loaded:
                print(f"[E18 PairedStats] Skip {b_key} vs {a_key} (missing arrays)", flush=True)
                continue
            A = loaded[a_key]
            B = loaded[b_key]
            t_keys = sorted(set(A["fidelity_per_task"]) & set(B["fidelity_per_task"]))
            fid_st = paired_bootstrap_stats(
                np.array([A["fidelity_per_task"][t] for t in t_keys]),
                np.array([B["fidelity_per_task"][t] for t in t_keys]),
                lower_is_better=True,
            )
            icl_st: Dict[str, Any] = {}
            for mm in ("rel_progress", "visual_knn"):
                for adap in ("lin4k", "rbf4k"):
                    ma = A["icl_per_task_seed"][mm][adap]
                    mb = B["icl_per_task_seed"][mm][adap]
                    ck = sorted(set(ma) & set(mb))
                    icl_st[f"{mm}_{adap}"] = paired_bootstrap_stats(
                        np.array([ma[k] for k in ck]),
                        np.array([mb[k] for k in ck]),
                        lower_is_better=True,
                    )
            ed_full = paired_bootstrap_stats(
                np.array(A["edit_quadruples"]["full_arith_mse10"]),
                np.array(B["edit_quadruples"]["full_arith_mse10"]),
                lower_is_better=True,
            )
            ed_op = paired_bootstrap_stats(
                np.array(A["edit_quadruples"]["op_arith_mse10"]),
                np.array(B["edit_quadruples"]["op_arith_mse10"]),
                lower_is_better=True,
            )
            comps_out[f"{b_key}_vs_{a_key}"] = {
                "B": b_key,
                "A": a_key,
                "fidelity_311tasks": fid_st,
                "visual_icl": icl_st,
                "edit_full_136quads": ed_full,
                "edit_op_136quads": ed_op,
            }
        atomic_write_json(comps_out, OUTPUT_ROOT / "paired_stats" / "paired_comparisons.json")
        print(f"[E18 PairedStats] Saved {len(comps_out)} paired comparisons.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--e15", action="store_true")
    p.add_argument("--auth", action="store_true")
    p.add_argument("--emergent", action="store_true")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--paired-list", type=Path, default=None)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    comps = None
    if args.paired_list is not None and args.paired_list.is_file():
        comps = [tuple(x) for x in json.loads(args.paired_list.read_text())]

    run_eval_manifest(
        args.manifest,
        do_e15=args.e15,
        do_auth=args.auth,
        do_emergent=args.emergent,
        do_paired=args.paired,
        paired_comparisons=comps,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
