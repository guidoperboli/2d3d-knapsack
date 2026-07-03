"""Basin-escape probe ("sonda colonnare").

Motivation: the okp5 optimum is an 8-item, 2-type columnar structure,
while the score-ordering search lives in a 16-item basin and marginal
exchanges across the divide are provably infeasible (all 300 master
proposals refuted by the exact slave). Escaping requires *seeding*
whole alternative structures, not perturbing the current one.

Mechanism: enumerate small combinations (pairs/triples) of the most
promising item types, cap each type's copies by its axial-fit count
(the number of copies that could geometrically tile the knapsack on
their own), and solve the exact KP restricted to those copies with
CP-SAT. Models stay tiny (typically under 25 items), so each solve
costs milliseconds to a second; the best packing found across the
combinations is returned as a candidate incumbent for the
metaheuristic to adopt and refine.

On okp5 the optimal pair (42x32, 58x20) yields a 10-item model that
CP-SAT closes instantly at the literature optimum 27923.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Item, Knapsack, Packing


def _axial_fit(dims: Tuple[int, int, int], ks: Knapsack,
               allow_rotation: bool, item: Item) -> int:
    """Max copies of this type that could tile the knapsack alone,
    maximised over allowed rotations (an optimistic per-type cap)."""
    best = 0
    for (w, d, h) in item.rotations(allow_rotation, ks.is_3d):
        if w > ks.W or d > ks.D or h > ks.H:
            continue
        f = (ks.W // w) * (ks.D // d)
        if ks.is_3d:
            f *= (ks.H // h)
        best = max(best, f)
    return best


def columnar_probe(items: Sequence[Item], ks: Knapsack,
                   allow_rotation: bool,
                   current_best: float = 0.0,
                   top_types: int = 8,
                   max_combo: int = 3,
                   max_model_items: int = 28,
                   time_per_model: float = 1.0,
                   total_budget: float = 15.0) -> Optional[Packing]:
    """Return the best packing found across restricted exact solves,
    or None if nothing beats current_best within the budget."""
    from .cp_slave import cp_solve_kp

    # group by type
    groups: Dict[tuple, List[Item]] = {}
    for it in items:
        groups.setdefault((it.w, it.d, it.h, it.profit), []).append(it)

    meas = (lambda i: i.volume) if ks.is_3d else (lambda i: i.base_area)
    cap = ks.volume if ks.is_3d else ks.W * ks.D

    # rank types by density, keep the top ones
    typed = []
    for key, members in groups.items():
        rep = members[0]
        fit = _axial_fit(key[:3], ks, allow_rotation, rep)
        if fit == 0:
            continue
        n_use = min(len(members), fit)
        typed.append((rep.profit / meas(rep), key, members[:n_use]))
    typed.sort(key=lambda t: -t[0])
    typed = typed[:top_types]
    if not typed:
        return None

    # enumerate combinations, most promising first
    combos = []
    for r in range(1, max_combo + 1):
        for combo in combinations(range(len(typed)), r):
            members: List[Item] = []
            for ci in combo:
                members.extend(typed[ci][2])
            if len(members) > max_model_items:
                continue
            pot = sum(i.profit for i in members)
            vol = sum(meas(i) for i in members)
            # optimistic potential: profit of what fits by volume alone
            if vol > cap:
                dens_sorted = sorted(members, key=lambda i: -i.profit / meas(i))
                pot, used = 0.0, 0
                for i in dens_sorted:
                    if used + meas(i) <= cap:
                        pot += i.profit
                        used += meas(i)
            if pot <= current_best:
                continue
            combos.append((pot, members))
    combos.sort(key=lambda t: -t[0])

    best_profit = current_best
    best_packing: Optional[Packing] = None
    t0 = time.time()
    for pot, members in combos:
        if time.time() - t0 > total_budget:
            break
        if pot <= best_profit:
            continue
        profit, _bound, pl, _opt = cp_solve_kp(
            members, ks, allow_rotation, time_limit=time_per_model)
        if pl is not None and profit > best_profit:
            best_profit = profit
            best_packing = Packing(ks, pl)
    return best_packing
