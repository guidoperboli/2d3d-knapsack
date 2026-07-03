"""Run GASP on the bundled benchmark instances and report gaps against
the proven optima / best-known values (or the 1D knapsack upper bound
when no optimum is available), mirroring Section 4 of the paper.

Usage:
    python examples/run_benchmark.py <set> [time_limit] [max_instances]

    <set> in: ngcut, cgcut, gcut, okp, hccut, ngcutfs1..3,
              thpack1..thpack7, br

Examples:
    python examples/run_benchmark.py ngcut 10
    python examples/run_benchmark.py gcut 10
    python examples/run_benchmark.py thpack1 10 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasp import GASP, GASPParams, knapsack_upper_bound
from gasp.best_known import BR_MEAN_VOLUME, optimum
from gasp.readers import load_set


def main():
    set_name = sys.argv[1] if len(sys.argv) > 1 else "ngcut"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    max_inst = int(sys.argv[3]) if len(sys.argv) > 3 else None

    instances = load_set(set_name)
    if max_inst:
        instances = instances[:max_inst]

    is_clp = set_name.startswith("thpack") or set_name == "br"
    rotation = is_clp  # CLP allows rotations; classic 2D sets do not

    print(f"Set: {set_name}  |  instances: {len(instances)}  |  "
          f"time limit: {time_limit:.0f}s  |  rotation: {rotation}\n")

    gaps, fills, solved = [], [], 0
    for name, items, ks in instances:
        bk = optimum(name)
        target = bk.value if bk else None
        params = GASPParams(time_limit=time_limit, allow_rotation=rotation,
                            seed=1, known_optimum=target)
        res = GASP(items, ks, params).run()

        fill = 100.0 * res.best_packing.used_volume / ks.volume
        fills.append(fill)

        if target is not None:
            gap = 100.0 * (target - res.best_profit) / target
            tag = "OPT" if (bk.proven and res.best_profit >= target) else \
                  ("=BK" if res.best_profit >= target else "")
            if res.best_profit >= target:
                solved += 1
        else:
            ub = knapsack_upper_bound(items, ks)
            gap = 100.0 * (ub - res.best_profit) / ub if ub > 0 else 0.0
            tag = "vs UB"
        gaps.append(gap)

        print(f"{name:14s} n={len(items):5d}  profit={res.best_profit:12.0f}"
              f"  gap={gap:7.2f}%  fill={fill:5.1f}%"
              f"  t={res.elapsed:5.1f}s  {tag}")

    print(f"\nMean gap: {sum(gaps)/len(gaps):.2f}%   "
          f"Mean fill: {sum(fills)/len(fills):.2f}%   "
          f"Solved to optimum/BK: {solved}/{len(instances)}")
    if set_name in BR_MEAN_VOLUME:
        print(f"Reference mean volume (lit., Table 4): "
              f"{BR_MEAN_VOLUME[set_name]:.2f}%")


if __name__ == "__main__":
    main()
