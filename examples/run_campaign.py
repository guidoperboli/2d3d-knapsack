#!/usr/bin/env python3
"""Campaign runner with checkpointing, for long local benchmark runs.

Reproduces (and extends) the experimental campaigns of the project:
classic 2D sets, Egeblad-Pisinger 2D/3D, BR container loading and the
large ngcutfs sets, with multi-seed protocols, automatic metric choice
(gap vs proven optimum, gap vs exact 1D bound, or fill % for CLP) and
resumable JSON checkpoints: interrupt with Ctrl+C at any time and
relaunch with the same command line to continue.

Examples
--------
# elenco dei set disponibili e numero di istanze
python examples/run_campaign.py --list-sets

# best-of-10 completo su BR7 (100 istanze x 10 seed x 10 s ~ 2h45)
python examples/run_campaign.py --sets thpack7 --seeds 10 --time 10

# tutte le BR, single run (700 istanze x 10 s ~ 2h)
python examples/run_campaign.py --sets thpack1 thpack2 thpack3 thpack4 \
    thpack5 thpack6 thpack7 --time 10

# le 630 ngcutfs della Tabella 1 (30 s ciascuna ~ 5h15)
python examples/run_campaign.py --sets ngcutfs1 ngcutfs2 ngcutfs3 --time 30

# classiche 2D con sonda colonnare e riparazione esatta, best-of-5
python examples/run_campaign.py --sets ngcut cgcut gcut okp --seeds 5 \
    --time 30 --basin-probe --exact-repair

# solo riepilogo di una campagna gia' (anche parzialmente) eseguita
python examples/run_campaign.py --summary results/campaign.json

# parallelo: best-of-10 su BR7 con 8 processi (~20 min invece di 2h45)
python examples/run_campaign.py --sets thpack7 --seeds 10 --time 10 --workers 8

# con i parametri vincitori del tuning (vedi tune_params.py)
python examples/run_campaign.py --sets thpack7 --time 10 --config results/best_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasp import GASPParams, knapsack_upper_bound
from gasp.adaptive import AdaptiveGASP
from gasp.best_known import OPTIMA
from gasp.io_utils import write_json_atomic
from gasp.readers import load_set

ALL_SETS = ["ngcut", "cgcut", "gcut", "okp", "hccut",
            "ep_other", "ep2", "ep3",
            "thpack1", "thpack2", "thpack3", "thpack4",
            "thpack5", "thpack6", "thpack7", "thpack8",
            "thpack9", "thpack10", "thpack11", "thpack12",
            "thpack13", "thpack14", "thpack15", "br",
            "ngcutfs1", "ngcutfs2", "ngcutfs3"]

ROTATION_DEFAULT = {s: True for s in
                    ("ep2", "ep3", "br",
                     "thpack1", "thpack2", "thpack3", "thpack4",
                     "thpack5", "thpack6", "thpack7", "thpack8",
                     "thpack9", "thpack10", "thpack11", "thpack12",
                     "thpack13", "thpack14", "thpack15")}


def is_clp(set_name: str) -> bool:
    return set_name.startswith("thpack")


def opt_key(name: str) -> str:
    """Map EP-version instance names onto the optima table."""
    if name.startswith("beasley") and name[7:].isdigit():
        return f"ngcut{int(name[7:]):02d}"
    if name.startswith("ngcut") and name[5:].isdigit():
        return f"ngcut{int(name[5:]):02d}"
    return name


# ----------------------------------------------------------------------
def run_campaign(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}

    deltas = tuple(int(d) for d in args.pch_deltas.split(",")) \
        if args.pch_deltas else (10,)

    # ---- piano della campagna (per la stima dei tempi)
    plan = []
    for set_name in args.sets:
        instances = load_set(set_name)
        if args.n_instances:
            instances = instances[:args.n_instances]
        rot = args.rotation if args.rotation is not None \
            else ROTATION_DEFAULT.get(set_name, False)
        for name, items, ks in instances:
            rec = res.setdefault(set_name, {}).setdefault(
                name, {"n": len(items), "runs": {}})
            if "metric" not in rec:
                if is_clp(set_name):
                    rec["metric"] = "fill"
                else:
                    opt = OPTIMA.get(opt_key(name))
                    if opt is not None:
                        rec["metric"] = "opt"
                        rec["ref"] = opt.value
                    else:
                        rec["metric"] = "ub"
                        rec["ref"] = knapsack_upper_bound(items, ks)
            for seed in range(1, args.seeds + 1):
                plan.append((set_name, name, rot, seed,
                             str(seed) in rec["runs"]))
    todo = [p for p in plan if not p[4]]
    print(f"Campagna: {len(plan)} run totali, {len(plan) - len(todo)} "
          f"gia' in checkpoint, {len(todo)} da eseguire "
          f"(~{len(todo) * args.time / 3600:.1f} h al limite di "
          f"{args.time:.0f} s/run)\nCheckpoint: {out}\n")

    overrides_base = {}
    if args.config:
        overrides_base.update(json.loads(Path(args.config).read_text()))
    if args.pch_deltas:
        overrides_base["pch_deltas"] = deltas
    if args.exact_repair:
        overrides_base["exact_repair"] = True
    if args.basin_probe:
        overrides_base["basin_probe"] = True
    if getattr(args, "respect_orientation", False):
        overrides_base["respect_orientation"] = True
    if getattr(args, "layout_search", False):
        overrides_base["layout_search"] = True
    if getattr(args, "parreno_seed", False):
        overrides_base["parreno_seed"] = True
    if getattr(args, "solver", "gasp") == "alns":
        # ALNS is a standalone solver: it ignores GASP-only flags
        # (parreno_seed, layout_search are intrinsic to it) and carries
        # its own. Keep only respect_orientation, solver and objective.
        ro = overrides_base.get("respect_orientation", False)
        overrides_base = {"solver": "alns"}
        if ro:
            overrides_base["respect_orientation"] = True
        obj = getattr(args, "objective", "volume")
        if obj == "profit":
            overrides_base["objective_metric"] = "profit"

    # task = (set_name, name, inst_idx, rot, seed, overrides)
    tasks = []
    for set_name in args.sets:
        instances = load_set(set_name)
        if args.n_instances:
            instances = instances[:args.n_instances]
        rot = args.rotation if args.rotation is not None \
            else ROTATION_DEFAULT.get(set_name, False)
        for inst_idx, (name, items, ks) in enumerate(instances):
            rec = res[set_name][name]
            ov = dict(overrides_base)
            if rec["metric"] == "opt":
                ov["known_optimum"] = rec["ref"]
            for seed in range(1, args.seeds + 1):
                if str(seed) in rec["runs"]:
                    continue
                tasks.append((set_name, name, inst_idx, rot, seed, ov))

    def store(set_name, name, seed, profit, fill, k, t_start,
              seed_fill=None, pre_fill=None, elapsed=None):
        rec = res[set_name][name]
        val = fill if rec["metric"] == "fill" else profit
        rec["runs"][str(seed)] = val
        # stage decomposition (3D fill only): seed and pre-layout fills
        if rec["metric"] == "fill" and seed_fill is not None:
            rec.setdefault("seed_runs", {})[str(seed)] = seed_fill
            rec.setdefault("prelayout_runs", {})[str(seed)] = pre_fill
        if elapsed is not None:
            rec.setdefault("time_runs", {})[str(seed)] = elapsed
        write_json_atomic(out, res)
        el = time.time() - t_start
        eta = el / (k + 1) * (len(tasks) - k - 1)
        shown = f"fill={val:.2f}%" if rec["metric"] == "fill" else (
            f"gap={100*(rec['ref']-val)/rec['ref']:.2f}%")
        print(f"[{k+1}/{len(tasks)}] {set_name}/{name} s{seed}: {shown}"
              f"   (ETA {eta/3600:.1f} h)", flush=True)

    t_start = time.time()
    try:
        if args.workers <= 1:
            print(f"Esecuzione SERIALE di {len(tasks)} run "
                  f"(--workers 1).", flush=True)
            from gasp.runner import run_one
            for k, (sn, name, ii, rot, seed, ov) in enumerate(tasks):
                (_, profit, fill, sfill, pfill,
                 el) = run_one(sn, ii, seed, args.time, rot, ov)
                store(sn, name, seed, profit, fill, k, t_start,
                      sfill, pfill, el)
        else:
            from concurrent.futures import (ProcessPoolExecutor,
                                            as_completed)
            from gasp.runner import run_one
            nw = min(args.workers, len(tasks))
            print(f"Esecuzione PARALLELA: {len(tasks)} run su {nw} "
                  f"worker. Le prime {nw} partono insieme; i "
                  f"completamenti compaiono a ondate ogni ~{args.time}s "
                  f"(nessun output per i primi ~{args.time}s e' "
                  f"normale).", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_one, sn, ii, seed, args.time,
                                  rot, ov): (sn, name, seed)
                        for (sn, name, ii, rot, seed, ov) in tasks}
                for k, fut in enumerate(as_completed(futs)):
                    sn, name, seed = futs[fut]
                    (_, profit, fill, sfill, pfill,
                     el) = fut.result()
                    store(sn, name, seed, profit, fill, k, t_start,
                          sfill, pfill, el)
    except KeyboardInterrupt:
        print("\nInterrotto: il checkpoint contiene tutte le run "
              "completate. Rilanciare lo stesso comando per riprendere.")
        return

    print("\nCampagna completata.")
    summarize(res)


# ----------------------------------------------------------------------
def summarize(res: dict) -> None:
    import statistics as _st
    print("\n" + "=" * 96)
    print(f"{'set':<10} {'ist.':>5} {'seed':>4} {'single':>8} "
          f"{'mean':>8} {'std':>6} {'min':>8} {'max':>8} "
          f"{'best-of':>8} {'t/ist(s)':>9} {'opt':>4}")
    print("-" * 96)
    for set_name in sorted(res):
        # per-instance aggregates, then averaged over instances
        singles, means, stds, mins, maxs, bests, times = \
            [], [], [], [], [], [], []
        n_opt, n_seeds = 0, 0
        for name, rec in res[set_name].items():
            runs = list(rec["runs"].values())
            if not runs:
                continue
            n_seeds = max(n_seeds, len(runs))
            tr = list(rec.get("time_runs", {}).values())
            if tr:
                times.append(_st.mean(tr))
            if rec["metric"] == "fill":
                vals = runs                       # higher better
                s1 = rec["runs"].get("1", runs[0])
                singles.append(s1)
                means.append(_st.mean(vals))
                stds.append(_st.pstdev(vals) if len(vals) > 1 else 0.0)
                mins.append(min(vals)); maxs.append(max(vals))
                bests.append(max(vals))
            else:
                ref = rec["ref"]
                gaps = [100 * (ref - v) / ref for v in runs]  # lower better
                s1 = rec["runs"].get("1", runs[0])
                singles.append(100 * (ref - s1) / ref)
                means.append(_st.mean(gaps))
                stds.append(_st.pstdev(gaps) if len(gaps) > 1 else 0.0)
                mins.append(min(gaps)); maxs.append(max(gaps))
                bests.append(min(gaps))
                if rec["metric"] == "opt" and max(runs) >= ref:
                    n_opt += 1
        if not singles:
            continue
        n = len(singles)
        avg = lambda xs: sum(xs) / len(xs)
        t_str = f"{avg(times):>8.2f}" if times else f"{'-':>8}"
        print(f"{set_name:<10} {n:>5} {n_seeds:>4} "
              f"{avg(singles):>7.2f}% {avg(means):>7.2f}% "
              f"{avg(stds):>5.2f} {avg(mins):>7.2f}% {avg(maxs):>7.2f}% "
              f"{avg(bests):>7.2f}% {t_str} {n_opt:>4}")
    print("=" * 96)
    print("CLP (thpack*): valori = fill %. Altri set: valori = gap %.")
    print("PER IL CONFRONTO CON LA LETTERATURA usare 'mean' (media sui "
          "seed) con 'std' e min/max; NON usare 'best-of' (cherry-pick).")
    print("'t/ist(s)' = tempo medio per istanza in secondi (media sui "
          "seed). Riferimento 3D BR: fascia metaeuristica ~secondi; "
          "metodi pesanti ~decine di minuti.")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", choices=ALL_SETS,
                    help="set di istanze da eseguire")
    ap.add_argument("--n-instances", type=int, default=None,
                    help="limita alle prime N istanze di ogni set")
    ap.add_argument("--seeds", type=int, default=1,
                    help="numero di seed per istanza (default 1)")
    ap.add_argument("--time", type=float, default=10.0,
                    help="secondi per run (default 10)")
    ap.add_argument("--rotation", dest="rotation", action="store_true",
                    default=None, help="forza le rotazioni")
    ap.add_argument("--no-rotation", dest="rotation", action="store_false",
                    help="forza orientazione fissa")
    ap.add_argument("--basin-probe", action="store_true",
                    help="attiva la sonda colonnare CP")
    ap.add_argument("--respect-orientation", action="store_true",
                    help="rispetta i flag di orientamento BR (protocollo "
                         "della letteratura, solo thpack/BR)")
    ap.add_argument("--layout-search", action="store_true",
                    help="ricerca sul layout fisico come post-process (3D)")
    ap.add_argument("--parreno-seed", action="store_true",
                    help="semina la soluzione iniziale col costruttivo a "
                         "blocchi di Parreno (3D, forte su BR)")
    ap.add_argument("--solver", choices=["gasp", "alns", "java", "java_alns"], default="java",
                    help="solver: 'java' (default) backend compilato superveloce "
                         "o 'gasp' o 'alns' (python pipeline) o 'java_alns'")
    ap.add_argument("--objective", choices=["volume", "profit"],
                    default="volume",
                    help="solo per --solver alns: obiettivo da "
                         "massimizzare (volume per CLP, profit per "
                         "knapsack)")
    ap.add_argument("--exact-repair", action="store_true",
                    help="attiva la riparazione esatta (master + slave CP)")
    ap.add_argument("--pch-deltas", default=None,
                    help='valori di delta per la PCH, es. "5,10,20,33,50"')
    ap.add_argument("--workers", type=int, default=1,
                    help="processi paralleli (default 1; consigliato: "
                         "n. core fisici)")
    ap.add_argument("--config", metavar="FILE", default=None,
                    help="JSON con override di GASPParams (es. l'output "
                         "del tuner)")
    ap.add_argument("--out", default="results/campaign.json",
                    help="file di checkpoint/risultati")
    ap.add_argument("--summary", metavar="FILE",
                    help="stampa solo il riepilogo di un file risultati")
    ap.add_argument("--list-sets", action="store_true",
                    help="elenca i set disponibili e il numero di istanze")
    args = ap.parse_args()

    if args.list_sets:
        for s in ALL_SETS:
            try:
                print(f"  {s:<10} {len(load_set(s)):>4} istanze")
            except Exception as e:
                print(f"  {s:<10} non disponibile ({e})")
        return
    if args.summary:
        summarize(json.loads(Path(args.summary).read_text()))
        return
    if not args.sets:
        ap.error("specificare --sets (oppure --list-sets / --summary)")
    run_campaign(args)


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
