"""2x2 effect analysis for the crossed prompt-query diagnostic.

Reads outputs/cross_{task}[/grasped]/rollout_results.csv and writes
tables.json + a printed summary: cell means, paired prompt diffs (fixed
visual), paired visual diffs (fixed prompt), and the interaction term.
"""
import csv
import json
import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fnum(v):
    if v in ('', 'None', None):
        return np.nan
    if v in ('True', 'False'):
        return float(v == 'True')
    return float(v)


def table(rows, metric, visuals, prompts):
    out = {}
    for vis in visuals:
        out[vis] = {}
        for p in prompts:
            sel = [fnum(r[metric]) for r in rows
                   if r['visual'] == vis and r['prompt'] == p]
            sel = [x for x in sel if not np.isnan(x)]
            out[vis][p] = dict(mean=float(np.mean(sel)) if sel else None,
                               n=len(sel),
                               sd=float(np.std(sel)) if sel else None)
    return out


def paired(rows, key_a, key_b, metric, visuals, prompts):
    """Mean of (b - a) over complete pairs, split by the held factor.
    key_a/key_b are the two prompt levels; visual levels come from `visuals`."""
    groups = {}
    for r in rows:
        g = (r['layout'], r['diff_seed'])
        groups.setdefault(g, {})[(r['visual'], r['prompt'])] = r
    res = {}
    for held in visuals:  # prompt effect within each visual
        diffs = [fnum(v[(held, key_b)][metric]) - fnum(v[(held, key_a)][metric])
                 for v in groups.values()
                 if (held, key_a) in v and (held, key_b) in v]
        diffs = [d for d in diffs if not np.isnan(d)]
        res[f'prompt[{key_b}-{key_a}]@{held}'] = dict(
            mean=float(np.mean(diffs)) if diffs else None, n=len(diffs),
            sd=float(np.std(diffs)) if diffs else None)
    for held in prompts:  # visual effect within each prompt
        diffs = [fnum(v[(visuals[1], held)][metric]) - fnum(v[(visuals[0], held)][metric])
                 for v in groups.values()
                 if (visuals[0], held) in v and (visuals[1], held) in v]
        diffs = [d for d in diffs if not np.isnan(d)]
        res[f'visual[{visuals[1]}-{visuals[0]}]@{held}'] = dict(
            mean=float(np.mean(diffs)) if diffs else None, n=len(diffs),
            sd=float(np.std(diffs)) if diffs else None)
    # interaction: (b-a prompt diff on visual2) - (b-a prompt diff on visual1)
    d2 = [fnum(v[(visuals[1], prompts[1])][metric]) - fnum(v[(visuals[1], prompts[0])][metric])
          for v in groups.values()
          if (visuals[1], prompts[0]) in v and (visuals[1], prompts[1]) in v]
    d1 = [fnum(v[(visuals[0], prompts[1])][metric]) - fnum(v[(visuals[0], prompts[0])][metric])
          for v in groups.values()
          if (visuals[0], prompts[0]) in v and (visuals[0], prompts[1]) in v]
    d2, d1 = [x for x in d2 if not np.isnan(x)], [x for x in d1 if not np.isnan(x)]
    res['interaction'] = dict(
        mean=(float(np.mean(d2) - np.mean(d1))
              if d2 and d1 else None),
        n=min(len(d2), len(d1)))
    return res


def analyze(task, results_dir, tag=''):
    rows = list(csv.DictReader((results_dir / 'rollout_results.csv').open()))
    T = {'cup': (['empty', 'filled'], ['normal', 'gentle'],
                 ['success', 'lifted', 'near_goal', 'transport_duration_s',
                  'eef_speed_mean_m_s', 'eef_speed_p95_m_s', 'eef_acc_p95_m_s2',
                  'eef_jerk_p95_m_s3', 'tilt_mean_deg', 'transport_steps']),
         'obstacle': (['plastic', 'glass'], ['narrow', 'wide'],
                      ['success', 'lifted', 'near_goal', 'transported',
                       'transport_min_clearance_m', 'post_contact_events',
                       'transport_path_length_m', 'eef_speed_mean_m_s',
                       'steps'])}
    visuals, prompts, metrics_ = T[task]
    out = {'tag': tag, 'n_rollouts': len(rows),
           'cells': {}, 'paired_diffs': {}, 'failures': []}
    for m in metrics_:
        out['cells'][m] = table(rows, m, visuals, prompts)
        out['paired_diffs'][m] = paired(rows, prompts[0], prompts[1], m,
                                        visuals, prompts) if 'prompt' in m or True else None
    # prompt diffs use prompt axis; visual diffs handled inside paired()
    for m in metrics_:
        out['paired_diffs'][m] = paired(rows, prompts[0], prompts[1], m,
                                        visuals, prompts)
    for r in rows:
        if r['success'] != 'True':
            out['failures'].append({k: r[k] for k in
                                    ('layout', 'visual', 'diff_seed', 'prompt',
                                     'failure_stage', 'steps')})
    (results_dir / 'tables.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({m: out['paired_diffs'][m] for m in metrics_},
                     indent=2))
    print('success cells:', json.dumps(out['cells']['success'], indent=1))
    print(f"failures: {len(out['failures'])}/{len(rows)}")


if __name__ == '__main__':
    task = sys.argv[1]
    sub = sys.argv[2] if len(sys.argv) > 2 else f'cross_{task}'
    analyze(task, HERE / 'outputs' / sub, tag=sub)
