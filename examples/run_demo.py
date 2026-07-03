"""Demo: run GASP on randomly generated 2D-KP, 3D-KP and 3D-CLP
instances and report the gap against the 1D knapsack upper bound,
mirroring the experimental setup of Section 4."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasp import GASP, GASPParams, generate_2d, generate_3d, knapsack_upper_bound


def report(name, items, knapsack, result, ub):
    gap = 100.0 * (ub - result.best_profit) / ub if ub > 0 else 0.0
    fill = 100.0 * result.best_packing.used_volume / knapsack.volume
    print(f"{name:28s} items={len(items):4d} loaded={len(result.best_packing.placements):4d} "
          f"profit={result.best_profit:12.0f} UB-gap={gap:6.2f}% "
          f"fill={fill:5.1f}% iters={result.iterations:5d} time={result.elapsed:5.1f}s")


def main(time_limit=5.0):
    print("GASP demo (time limit per instance: %.0f s)\n" % time_limit)

    # ---- 2D-KP, no rotation -----------------------------------------
    items, ks = generate_2d(n=50, geom_class="D", p=90, seed=42)
    res = GASP(items, ks, GASPParams(time_limit=time_limit, seed=1)).run()
    report("2D-KP  (D, n=50, no rot)", items, ks, res, knapsack_upper_bound(items, ks))

    # ---- 2D-KP, with rotation ---------------------------------------
    items, ks = generate_2d(n=50, geom_class="L", p=90, seed=7)
    res = GASP(items, ks, GASPParams(time_limit=time_limit,
                                     allow_rotation=True, seed=1)).run()
    report("2D-KP  (L, n=50, rot)", items, ks, res, knapsack_upper_bound(items, ks))

    # ---- 3D-KP, with rotation ---------------------------------------
    items, ks = generate_3d(n=40, geom_class="D", p=90, seed=11)
    res = GASP(items, ks, GASPParams(time_limit=time_limit,
                                     allow_rotation=True, seed=1)).run()
    report("3D-KP  (D, n=40, rot)", items, ks, res, knapsack_upper_bound(items, ks))

    # ---- 3D-CLP (profits = volumes) ---------------------------------
    items, ks = generate_3d(n=60, geom_class="U", p=99, seed=3, clp=True)
    res = GASP(items, ks, GASPParams(time_limit=time_limit,
                                     allow_rotation=True, seed=1)).run()
    report("3D-CLP (U, n=60, rot)", items, ks, res, knapsack_upper_bound(items, ks))


if __name__ == "__main__":
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    main(tl)
