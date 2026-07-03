"""Random instance generators in the style of Section 4.1 of the paper
(Egeblad and Pisinger, 2007/2009 generation scheme), plus the 1D
knapsack upper bound used in Tables 2 and 3.

2D instances:
    classes S (square), D (diverse), L (long), U (uniform)
    profit p_i = k * (w_i * h_i), k random in {1, 2, 3}
3D instances:
    classes C (cubes), D, L, U
    profit p_i = 200 + w_i * d_i * h_i
3D-CLP:
    profits equal to the volumes.

The knapsack size is set so that its volume equals p% of the total
volume of the items (p in {50, 90}), with proportions kept cubic.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from .geometry import Item, Knapsack


def _size_2d(cls: str, rng: random.Random) -> Tuple[int, int]:
    if cls == "S":
        w = rng.randint(1, 100)
        return w, w
    if cls == "D":
        return rng.randint(1, 50), rng.randint(1, 50)
    if cls == "L":
        m = 200 // 3
        return rng.randint(1, m), rng.randint(1, m)
    if cls == "U":
        return rng.randint(50, 100), rng.randint(50, 100)
    raise ValueError(f"Unknown 2D class {cls}")


def _size_3d(cls: str, rng: random.Random) -> Tuple[int, int, int]:
    if cls == "C":
        w = rng.randint(1, 100)
        return w, w, w
    if cls == "D":
        return (rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 50))
    if cls == "L":
        m = 200 // 3
        return (rng.randint(1, m), rng.randint(50, 100), rng.randint(1, m))
    if cls == "U":
        return (rng.randint(50, 100), rng.randint(50, 100), rng.randint(50, 100))
    raise ValueError(f"Unknown 3D class {cls}")


def generate_2d(n: int, geom_class: str = "D", clustered: bool = False,
                p: int = 90, seed: int = 0) -> Tuple[List[Item], Knapsack]:
    rng = random.Random(seed)
    base = 20 if clustered else n
    sizes = [_size_2d(geom_class, rng) for _ in range(base)]
    if clustered:
        sizes = [sizes[i % base] for i in range(n)]
    items = []
    for idx, (w, d) in enumerate(sizes):
        k = rng.randint(1, 3)
        items.append(Item(idx, w, d, 1, profit=k * w * d))
    total_area = sum(i.w * i.d for i in items)
    side = max(1, int(math.sqrt(total_area * p / 100.0)))
    return items, Knapsack(side, side, 1)


def generate_3d(n: int, geom_class: str = "D", clustered: bool = False,
                p: int = 90, seed: int = 0,
                clp: bool = False) -> Tuple[List[Item], Knapsack]:
    rng = random.Random(seed)
    base = 20 if clustered else n
    sizes = [_size_3d(geom_class, rng) for _ in range(base)]
    if clustered:
        sizes = [sizes[i % base] for i in range(n)]
    items = []
    for idx, (w, d, h) in enumerate(sizes):
        vol = w * d * h
        profit = vol if clp else 200 + vol
        items.append(Item(idx, w, d, h, profit=profit))
    total_vol = sum(i.volume for i in items)
    side = max(1, int(round((total_vol * p / 100.0) ** (1.0 / 3.0))))
    return items, Knapsack(side, side, side)


# ----------------------------------------------------------------------
def knapsack_upper_bound(items: List[Item], knapsack: Knapsack) -> float:
    """1D knapsack upper bound (Section 4.2.2 / 4.2.3): items weighted by
    their area (2D) or volume (3D), capacity = knapsack area/volume.

    Solved exactly by DP when the capacity is tractable, otherwise by
    the Dantzig LP relaxation bound (still a valid upper bound).
    """
    if knapsack.is_3d:
        weights = [i.volume for i in items]
        cap = knapsack.volume
    else:
        weights = [i.base_area for i in items]
        cap = knapsack.W * knapsack.D

    if cap <= 2_000_000 and len(items) * cap <= 200_000_000:
        dp = [0.0] * (cap + 1)
        for it, wgt in zip(items, weights):
            for c in range(cap, wgt - 1, -1):
                cand = dp[c - wgt] + it.profit
                if cand > dp[c]:
                    dp[c] = cand
        return dp[cap]

    if cap <= 50_000_000:
        # exact 0-1 DP, vectorised with numpy (needed e.g. for the EP
        # 3D instances, whose knapsack volume is ~3.5M: the Dantzig LP
        # bound is far too loose there, especially on cube classes)
        import numpy as np
        dp = np.zeros(cap + 1, dtype=np.float64)
        for it, wgt in zip(items, weights):
            if wgt > cap:
                continue
            cand = dp[:-wgt] + it.profit   # snapshot read (0-1 semantics)
            np.maximum(dp[wgt:], cand, out=dp[wgt:])
        return float(dp[cap])

    # LP relaxation (Dantzig bound)
    order = sorted(zip(items, weights), key=lambda t: -(t[0].profit / t[1]))
    ub, remaining = 0.0, cap
    for it, wgt in order:
        if wgt <= remaining:
            ub += it.profit
            remaining -= wgt
        else:
            ub += it.profit * remaining / wgt
            break
    return ub
