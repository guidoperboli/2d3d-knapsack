#!/usr/bin/env python3
"""Orchestratore della "notte di calcolo": quattro fasi in sequenza.

  1. tuning famiglia BR        -> results/best_br.json
  2. tuning famiglia classic   -> results/best_classic.json
  3. best-of-10 su BR7 con la configurazione vincente del tuning BR
  4. ngcutfs1-3 single-run, 30 s, parametri di default (fedeli al paper,
     per confrontabilita' con la Tabella 1)
  5. workbook Excel finale     -> results/Risultati_notte.xlsx

Ogni fase usa il proprio checkpoint: se la macchina si ferma, rilanciare
lo stesso comando riprende dal punto esatto (le fasi gia' complete
vengono attraversate in pochi secondi). Log completo con orari in
results/overnight.log.

Uso:
    python examples/run_overnight.py --workers 8
    python examples/run_overnight.py --workers 8 --skip tune_br tune_classic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def phases(workers: int, smoke: bool, time_limit: int = 10):
    w = ["--workers", str(workers)]
    t_str = str(time_limit)
    if smoke:   # collaudo rapido della catena (~1 minuto)
        return [
            ("tune_br", [PY, HERE / "tune_params.py", "--sets", "gcut",
                         "--n-instances", "1", "--configs", "2",
                         "--seeds", "1", "--time", "2",
                         "--out", "results/tuning_br.json",
                         "--best-out", "results/best_br.json", *w]),
            ("tune_classic", [PY, HERE / "tune_params.py", "--sets", "okp",
                              "--n-instances", "1", "--configs", "2",
                              "--seeds", "1", "--time", "2",
                              "--out", "results/tuning_classic.json",
                              "--best-out", "results/best_classic.json", *w]),
            ("br7_bestof10", [PY, HERE / "run_campaign.py", "--sets",
                              "thpack7", "--n-instances", "2", "--seeds",
                              "2", "--time", "2",
                              "--config", "results/best_br.json",
                              "--respect-orientation", "--parreno-seed",
                              "--layout-search",
                              "--out", "results/br7_bestof10.json", *w]),
            ("br_all", [PY, HERE / "run_campaign.py", "--sets", "thpack1",
                        "--n-instances", "2", "--time", "2",
                        "--config", "results/best_br.json",
                        "--respect-orientation", "--parreno-seed",
                        "--layout-search",
                        "--out", "results/br_all.json", *w]),
            ("br_alns", [PY, HERE / "run_campaign.py", "--sets",
                         "thpack1", "--n-instances", "2", "--time", "2",
                         "--respect-orientation", "--solver", "alns",
                         "--out", "results/br_alns.json", *w]),
            ("kp_gasp", [PY, HERE / "run_campaign.py", "--sets", "okp",
                         "--n-instances", "2", "--seeds", "1", "--time", "2",
                         "--out", "results/kp_gasp.json", *w]),
            ("kp_alns", [PY, HERE / "run_campaign.py", "--sets", "okp",
                         "--n-instances", "2", "--seeds", "1", "--time", "2",
                         "--solver", "alns", "--objective", "profit",
                         "--out", "results/kp_alns.json", *w]),
            ("ngcutfs", [PY, HERE / "run_campaign.py", "--sets", "ngcut",
                         "--n-instances", "2", "--time", "2",
                         "--out", "results/ngcutfs.json", *w]),
            ("xlsx", [PY, HERE / "make_results_xlsx.py", "--input",
                      "results/br7_bestof10.json", "results/br_all.json",
                      "results/br_alns.json", "results/kp_gasp.json",
                      "results/kp_alns.json", "results/ngcutfs.json",
                      "--out", "results/Risultati_notte.xlsx"]),
        ]
    return [
        ("tune_br", [PY, HERE / "tune_params.py", "--family", "br",
                     "--configs", "20", "--seeds", "2", "--time", t_str,
                     "--out", "results/tuning_br.json",
                     "--best-out", "results/best_br.json", *w]),
        ("tune_classic", [PY, HERE / "tune_params.py", "--family",
                          "classic", "--configs", "16", "--seeds", "2",
                          "--time", t_str,
                          "--out", "results/tuning_classic.json",
                          "--best-out", "results/best_classic.json", *w]),
        ("br7_bestof10", [PY, HERE / "run_campaign.py", "--sets",
                          "thpack7", "--seeds", "10", "--time", t_str,
                          "--config", "results/best_br.json",
                          "--respect-orientation", "--parreno-seed",
                          "--layout-search",
                          "--out", "results/br7_bestof10.json", *w]),
        ("br_all", [PY, HERE / "run_campaign.py", "--sets",
                    "thpack1", "thpack2", "thpack3", "thpack4",
                    "thpack5", "thpack6", "thpack7", "thpack8",
                    "thpack9", "thpack10", "thpack11", "thpack12",
                    "thpack13", "thpack14", "thpack15", "--time", t_str,
                    "--config", "results/best_br.json",
                    "--respect-orientation", "--parreno-seed",
                    "--layout-search",
                    "--out", "results/br_all.json", *w]),
        ("br_alns", [PY, HERE / "run_campaign.py", "--sets",
                     "thpack1", "thpack2", "thpack3", "thpack4",
                     "thpack5", "thpack6", "thpack7", "thpack8",
                     "thpack9", "thpack10", "thpack11", "thpack12",
                     "thpack13", "thpack14", "thpack15", "--time", t_str,
                     "--respect-orientation", "--solver", "alns",
                     "--out", "results/br_alns.json", *w]),
        ("kp_gasp", [PY, HERE / "run_campaign.py", "--sets",
                     "ngcut", "okp", "gcut", "ep2", "ep3", "--seeds", "5", "--time", t_str,
                     "--out", "results/kp_gasp.json", *w]),
        ("kp_alns", [PY, HERE / "run_campaign.py", "--sets",
                     "ngcut", "okp", "gcut", "ep2", "ep3", "--seeds", "5", "--time", t_str,
                     "--solver", "alns", "--objective", "profit",
                     "--out", "results/kp_alns.json", *w]),
        ("ngcutfs", [PY, HERE / "run_campaign.py", "--sets", "ngcutfs1",
                     "ngcutfs2", "ngcutfs3", "--time", "30",
                     "--out", "results/ngcutfs.json", *w]),
        ("xlsx", [PY, HERE / "make_results_xlsx.py", "--input",
                  "results/br7_bestof10.json", "results/br_all.json",
                  "results/br_alns.json", "results/kp_gasp.json",
                  "results/kp_alns.json", "results/ngcutfs.json",
                  "--out", "results/Risultati_notte.xlsx"]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4,
                    help="processi paralleli per fase (default 4; "
                         "consigliato: n. core fisici)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["tune_br", "tune_classic", "br7_bestof10",
                             "br_all", "ngcutfs", "xlsx"],
                    help="fasi da saltare")
    ap.add_argument("--smoke", action="store_true",
                    help="collaudo rapido della catena (~1 minuto)")
    ap.add_argument("--time", type=int, default=10,
                    help="tempo limite per singola istanza in secondi (default 10)")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    log = Path("results/overnight.log").open("a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"=== NOTTE DI CALCOLO: avvio (workers={args.workers}"
        f"{', SMOKE' if args.smoke else ''}) ===")
    t0 = time.time()
    for name, cmd in phases(args.workers, args.smoke, args.time):
        if name in args.skip:
            say(f"--- fase {name}: SALTATA")
            continue
        say(f"--- fase {name}: avvio")
        t1 = time.time()
        proc = subprocess.run([str(c) for c in cmd],
                              stdout=log, stderr=subprocess.STDOUT)
        dt = (time.time() - t1) / 60
        if proc.returncode != 0:
            say(f"--- fase {name}: ERRORE (codice {proc.returncode}) "
                f"dopo {dt:.1f} min — vedi results/overnight.log; "
                f"rilanciare lo stesso comando per riprendere")
            sys.exit(proc.returncode)
        say(f"--- fase {name}: completata in {dt:.1f} min")
    say(f"=== COMPLETATO in {(time.time()-t0)/3600:.2f} h — workbook: "
        f"results/Risultati_notte.xlsx ===")


if __name__ == "__main__":
    main()
