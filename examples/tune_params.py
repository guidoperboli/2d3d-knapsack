#!/usr/bin/env python3
"""Systematic parameter tuning for GASP (random search, paired design).

Samples random configurations from the GASPParams space (always
including the paper/default configuration as the baseline to beat),
evaluates every configuration on the same (instance, seed) cells of a
validation subset, and ranks them by mean rank across cells (robust to
heterogeneous instance scales). The best configuration is written as a
JSON file directly usable by run_campaign.py via --config.

Checkpointed and resumable like run_campaign: interrupt and relaunch
with the same command line at any time.

Examples
--------
# 20 configurazioni sul preset BR, 2 seed, 8 processi (~1h a 10 s/run)
python examples/tune_params.py --family br --configs 20 --seeds 2 \
    --time 10 --workers 8

# preset classico 2D, piu' rapido
python examples/tune_params.py --family classic --configs 16 --time 10

# solo la classifica di un tuning gia' eseguito
python examples/tune_params.py --summary results/tuning.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasp import knapsack_upper_bound
from gasp.best_known import OPTIMA
from gasp.io_utils import write_json_atomic
from gasp.readers import load_set

SPACE = {
    "alpha": [0.05, 0.1, 0.2],
    "beta": [0.05, 0.1, 0.2],
    "non_improving_limit": [10, 20, 30, 50],
    "reinit_swaps": [3, 5, 10],
    "k_init": [2, 3, 5],
    "pch_deltas": [[10], [5, 10, 20, 33, 50]],
}
BASELINE = {"alpha": 0.1, "beta": 0.1, "non_improving_limit": 30,
            "reinit_swaps": 5, "k_init": 3, "pch_deltas": [10]}

FAMILIES = {
    "classic": (False, [("gcut", [0, 2, 6, 9, 10]),
                        ("okp", [0, 3]), ("cgcut", [0])]),
    "br":      (True,  [("thpack1", [0, 1, 2]),
                        ("thpack4", [0, 1, 2]),
                        ("thpack7", [0, 1, 2])]),
    "ep3":     (True,  [("ep3", [4, 8, 12, 16, 24, 28, 32, 36])]),
    "mix":     (None,  [("gcut", [2, 9]), ("okp", [3]),
                        ("thpack1", [0]), ("thpack7", [0]),
                        ("ep3", [4, 12])]),
}
ROT_BY_SET = {s: True for s in
              ("ep2", "ep3", "thpack1", "thpack2", "thpack3", "thpack4",
               "thpack5", "thpack6", "thpack7")}


def sample_configs(n: int, space_seed: int):
    rng = random.Random(space_seed)
    configs = [("baseline", dict(BASELINE))]
    seen = {json.dumps(BASELINE, sort_keys=True)}
    while len(configs) < n:
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        key = json.dumps(c, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        configs.append((f"cfg{len(configs):02d}", c))
    return configs


def cell_refs(subsets) -> dict:
    """Per-instance reference for the lower-better metric."""
    refs = {}
    for set_name, idxs in subsets:
        instances = load_set(set_name)
        for i in idxs:
            name, items, ks = instances[i]
            key = f"{set_name}/{i}"
            if set_name.startswith("thpack"):
                refs[key] = {"name": name, "metric": "fill"}
            else:
                opt = OPTIMA.get(name)
                refs[key] = {"name": name,
                             "metric": "opt" if opt else "ub",
                             "ref": opt.value if opt
                             else knapsack_upper_bound(items, ks)}
    return refs


def lower_better(ref: dict, profit: float, fill: float) -> float:
    if ref["metric"] == "fill":
        return round(100.0 - fill, 2)
    return round(100.0 * (ref["ref"] - profit) / ref["ref"], 2)


# ----------------------------------------------------------------------
def run_tuning(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(out.read_text()) if out.exists() else {}

    if args.family:
        fam_rot, subsets = FAMILIES[args.family]
    else:
        subsets = [(s, list(range(args.n_instances or 3)))
                   for s in args.sets]
        fam_rot = None

    if "refs" not in state:
        state["refs"] = cell_refs(subsets)
    refs = state["refs"]
    configs = sample_configs(args.configs, args.space_seed)
    cfgs = state.setdefault("configs", {})
    for cid, params in configs:
        cfgs.setdefault(cid, {"params": params, "results": {}})

    tasks = []
    for cid, params in configs:
        done = cfgs[cid]["results"]
        for set_name, idxs in subsets:
            rot = (fam_rot if fam_rot is not None
                   else ROT_BY_SET.get(set_name, False))
            for i in idxs:
                for seed in range(1, args.seeds + 1):
                    cell = f"{set_name}/{i}/{seed}"
                    if cell in done:
                        continue
                    ov = dict(params)
                    ref = refs[f"{set_name}/{i}"]
                    if ref["metric"] == "opt":
                        ov["known_optimum"] = ref["ref"]
                    tasks.append((cid, set_name, i, rot, seed, ov))

    n_cells = sum(len(i) for _, i in subsets) * args.seeds
    print(f"Tuning: {len(configs)} configurazioni x {n_cells} celle = "
          f"{len(configs) * n_cells} run; {len(tasks)} da eseguire "
          f"(~{len(tasks) * args.time / 3600:.1f} h / {args.workers} "
          f"worker)\nCheckpoint: {out}\n")

    def store(cid, set_name, i, seed, profit, fill, k, t0):
        ref = refs[f"{set_name}/{i}"]
        val = lower_better(ref, profit, fill)
        cfgs[cid]["results"][f"{set_name}/{i}/{seed}"] = val
        write_json_atomic(out, state)
        el = time.time() - t0
        eta = el / (k + 1) * (len(tasks) - k - 1)
        print(f"[{k+1}/{len(tasks)}] {cid} {set_name}/{ref['name']} "
              f"s{seed}: {val:.2f}   (ETA {eta/3600:.1f} h)", flush=True)

    t0 = time.time()
    try:
        if args.workers <= 1:
            from gasp.runner import run_one
            for k, (cid, sn, i, rot, seed, ov) in enumerate(tasks):
                _, profit, fill, *_rest = run_one(sn, i, seed, args.time, rot, ov)
                store(cid, sn, i, seed, profit, fill, k, t0)
        else:
            from concurrent.futures import (ProcessPoolExecutor,
                                            as_completed)
            from gasp.runner import run_one
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_one, sn, i, seed, args.time,
                                  rot, ov): (cid, sn, i, seed)
                        for (cid, sn, i, rot, seed, ov) in tasks}
                for k, fut in enumerate(as_completed(futs)):
                    cid, sn, i, seed = futs[fut]
                    _, profit, fill, *_rest = fut.result()
                    store(cid, sn, i, seed, profit, fill, k, t0)
    except KeyboardInterrupt:
        print("\nInterrotto: checkpoint salvato, rilanciare per riprendere.")
        return

    rank_and_save(state, args.best_out)


# ----------------------------------------------------------------------
def rank_and_save(state: dict, best_out: str) -> None:
    cfgs = state["configs"]
    cells = sorted({c for v in cfgs.values() for c in v["results"]})
    ranks = {cid: [] for cid in cfgs}
    for cell in cells:
        have = [(cid, v["results"][cell]) for cid, v in cfgs.items()
                if cell in v["results"]]
        if len(have) < 2:
            continue
        have.sort(key=lambda t: t[1])
        for pos, (cid, _) in enumerate(have, 1):
            ranks[cid].append(pos)

    rows = []
    for cid, v in cfgs.items():
        vals = list(v["results"].values())
        if not vals or not ranks[cid]:
            continue
        rows.append((sum(ranks[cid]) / len(ranks[cid]),
                     sum(vals) / len(vals), cid, v["params"]))
    rows.sort()

    print("\n" + "=" * 72)
    print(f"{'config':<10} {'mean rank':>9} {'mean value':>11}   parametri")
    print("-" * 72)
    for mr, mv, cid, params in rows:
        pstr = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{cid:<10} {mr:>9.2f} {mv:>10.2f}%   {pstr}")
    print("=" * 72)
    print("mean value: media del 'lower-better' (gap % oppure 100-fill).")

    if rows:
        best = rows[0][3]
        Path(best_out).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(best_out, best, indent=2)
        print(f"\nMigliore configurazione salvata in {best_out}")
        print("Usala con: python examples/run_campaign.py ... "
              f"--config {best_out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES),
                    help="sottoinsieme di validazione predefinito")
    ap.add_argument("--sets", nargs="+", help="in alternativa a --family")
    ap.add_argument("--n-instances", type=int, default=None)
    ap.add_argument("--configs", type=int, default=16,
                    help="configurazioni campionate (inclusa la baseline)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--space-seed", type=int, default=42,
                    help="seed del campionamento (riproducibile)")
    ap.add_argument("--out", default="results/tuning.json")
    ap.add_argument("--best-out", default="results/best_config.json")
    ap.add_argument("--summary", metavar="FILE",
                    help="stampa solo la classifica di un tuning")
    args = ap.parse_args()

    if args.summary:
        state = json.loads(Path(args.summary).read_text())
        rank_and_save(state, args.best_out)
        return
    if not args.family and not args.sets:
        ap.error("specificare --family oppure --sets")
    run_tuning(args)


if __name__ == "__main__":
    main()
