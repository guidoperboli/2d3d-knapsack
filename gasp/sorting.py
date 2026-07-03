"""Sorting criteria used by the Packing Constructive Heuristic (PCH),
Section 3.2 of the paper.

Eight rules: Profit-Height, Height-Profit, Area-Height, Height-Area
and their clustered variants. Clustered rules group values into bands
of width delta% of the relevant range, sort by cluster, then break
ties with the secondary key.
"""

from __future__ import annotations

import math
from typing import Callable, List

from .geometry import Item, Knapsack

DELTA_DEFAULT = 10  # delta in [1, 100]


def _cluster(value: float, lo: float, hi: float, delta: int) -> int:
    """Cluster index of `value` in [lo, hi] using bands of delta%."""
    span = hi - lo
    if span <= 0:
        return 0
    width = span * delta / 100.0
    return int(math.floor((value - lo) / width)) if width > 0 else 0


def profit_height(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-i.profit, -i.h))


def clustered_profit_height(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    if not items:
        return []
    pmin = min(i.profit for i in items)
    pmax = max(i.profit for i in items)
    return sorted(items, key=lambda i: (-_cluster(i.profit, pmin, pmax, delta), -i.h))


def height_profit(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-i.h, -i.profit))


def clustered_height_profit(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-_cluster(i.h, 0, k.H, delta), -i.profit))


def area_height(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-i.base_area, -i.h))


def clustered_area_height(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-_cluster(i.base_area, 0, k.W * k.D, delta), -i.h))


def height_area(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-i.h, -i.base_area))


def clustered_height_area(items: List[Item], k: Knapsack, delta: int = DELTA_DEFAULT):
    return sorted(items, key=lambda i: (-_cluster(i.h, 0, k.H, delta), -i.base_area))


SORTING_RULES: List[Callable] = [
    profit_height,
    clustered_profit_height,
    height_profit,
    clustered_height_profit,
    area_height,
    clustered_area_height,
    height_area,
    clustered_height_area,
]
