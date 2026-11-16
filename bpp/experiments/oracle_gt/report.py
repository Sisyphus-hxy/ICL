#!/usr/bin/env python3
"""Aggregate paired Default/Oracle-GT results into SR tables.

Outputs under <output>/<benchmark>/:
  summary.csv       per task: n_queries, default SR, oracle SR, delta SR,
                    expert yield (GT gen success rate)
  per_query.csv     per (task, query): default_success, oracle_success,
                    transition (fail->success / success->fail / both / none /
                    no_gt / missing), plus steps
  report.json       same data in JSON for downstream use
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as T


def load_result(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def report_benchmark(benchmark, output):
    root = Path(output) / benchmark
    per_query_rows = []
    summary_rows = []
    for task in T.task_names(benchmark):
        gen_results = {}
        gen_path = root / 'experts' / task / 'generation_results.json'
        if gen_path.exists():
            gen_results = {int(k): v for k, v in
                           json.loads(gen_path.read_text()).items()}
        task_rows = []
        for qi in range(T.N_QUERIES):
            row = {'benchmark': benchmark, 'task': task, 'query': qi}
            for arm in ('default', 'oracle', 'shuffle'):
                r = load_result(root / 'eval' / task / f'query_{qi:02d}' / arm
                                / 'result.json')
                row[f'{arm}_success'] = (None if r is None else bool(r['success']))
                row[f'{arm}_steps'] = (None if r is None else r['steps'])
            gen = gen_results.get(qi)
            row['gt_available'] = bool(gen and gen.get('success'))
            if row['default_success'] is None:
                row['transition'] = 'missing'
            elif not row['gt_available'] or row['oracle_success'] is None:
                row['transition'] = 'no_gt'
            else:
                d, o = row['default_success'], row['oracle_success']
                row['transition'] = ('fail->success' if not d and o else
                                     'success->fail' if d and not o else
                                     'both' if d and o else 'none')
            task_rows.append(row)
            per_query_rows.append(row)
        ran_d = [r for r in task_rows if r['default_success'] is not None]
        ran_o = [r for r in task_rows if r['oracle_success'] is not None]
        ran_s = [r for r in task_rows if r['shuffle_success'] is not None]
        paired = [r for r in task_rows if r['transition']
                  in ('fail->success', 'success->fail', 'both', 'none')]
        paired3 = [r for r in task_rows
                   if r['default_success'] is not None
                   and r['oracle_success'] is not None
                   and r['shuffle_success'] is not None]
        gen_ok = sum(1 for v in gen_results.values() if v.get('success'))
        summary_rows.append({
            'benchmark': benchmark, 'task': task,
            'n_default': len(ran_d), 'n_oracle': len(ran_o), 'n_paired': len(paired),
            'n_shuffle': len(ran_s), 'n_paired3': len(paired3),
            'expert_yield': round(gen_ok / T.N_QUERIES, 4),
            'default_sr': (round(sum(r['default_success'] for r in ran_d) / len(ran_d), 4)
                           if ran_d else None),
            'oracle_sr': (round(sum(r['oracle_success'] for r in ran_o) / len(ran_o), 4)
                          if ran_o else None),
            'oracle_sr_paired': (round(sum(r['oracle_success'] for r in paired) / len(paired), 4)
                                 if paired else None),
            'default_sr_paired': (round(sum(r['default_success'] for r in paired) / len(paired), 4)
                                  if paired else None),
            'delta_sr_paired': (round(sum(r['oracle_success'] - r['default_success']
                                          for r in paired) / len(paired), 4)
                                if paired else None),
            'shuffle_sr': (round(sum(r['shuffle_success'] for r in ran_s) / len(ran_s), 4)
                           if ran_s else None),
            'shuffle_sr_paired3': (round(sum(r['shuffle_success'] for r in paired3)
                                         / len(paired3), 4)
                                   if paired3 else None),
            'delta_shuffle_vs_default_paired3': (
                round(sum(r['shuffle_success'] - r['default_success']
                          for r in paired3) / len(paired3), 4) if paired3 else None),
            'delta_shuffle_vs_oracle_paired3': (
                round(sum(r['shuffle_success'] - r['oracle_success']
                          for r in paired3) / len(paired3), 4) if paired3 else None),
        })
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in [('per_query.csv', per_query_rows),
                       ('summary.csv', summary_rows)]:
        if not rows:
            continue
        fields = list(rows[0])
        with (root / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    (root / 'report.json').write_text(json.dumps(
        {'summary': summary_rows, 'per_query': per_query_rows}, indent=1))
    print(f'wrote {root}/summary.csv ({len(summary_rows)} tasks), '
          f'per_query.csv ({len(per_query_rows)} rows)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--output', type=Path, default=T.OUTPUTS)
    args = p.parse_args()
    report_benchmark(args.benchmark, args.output)


if __name__ == '__main__':
    main()
