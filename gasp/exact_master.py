"""Exact subset-selection master (matheuristic layer).

Motivated by the diagnostic finding that GASP's score-ordering paradigm
stalls on subset-selection-dominated instances (okp5, cube classes,
p=25 fills): the packing is near-perfect but the item mix is wrong, and
local score swaps cannot express group exchanges because the greedy
refills freed space with the same items.

Components
----------
1. Pairwise conflict test (exact): two boxes can coexist in the
   knapsack iff, for some pair of allowed rotations, they can be
   separated along at least one axis. Necessary condition used as a
   valid cut; sufficiency is delegated to the greedy slave.
2. Conflict-aware upper bound: branch-and-bound maximising total
   profit subject to the volume capacity and pairwise compatibility.
   Strictly dominates the 1D knapsack bound; exact when it completes
   within the node budget, otherwise falls back to the 1D bound.
3. Marginal exchange enumeration: given the incumbent, enumerate
   (S out, T in) exchanges with profit gain, volume budget
   vol(S) + waste, pairwise-compatible T (also against kept items),
   ranked by gain. Failed exchanges are recorded as no-good cuts
   (logic-based Benders style).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .geometry import Item, Knapsack, Packing


# ----------------------------------------------------------------------
def _fits_alone(dims, ks: Knapsack) -> bool:
    w, d, h = dims
    return w <= ks.W and d <= ks.D and h <= ks.H


def compatible(i: Item, j: Item, ks: Knapsack, allow_rotation: bool) -> bool:
    """Exact coexistence test for two boxes: separable on some axis
    for some pair of feasible rotations."""
    ri = [r for r in i.rotations(allow_rotation, ks.is_3d) if _fits_alone(r, ks)]
    rj = [r for r in j.rotations(allow_rotation, ks.is_3d) if _fits_alone(r, ks)]
    for (wi, di, hi) in ri:
        for (wj, dj, hj) in rj:
            if (wi + wj <= ks.W or di + dj <= ks.D or hi + hj <= ks.H):
                return True
    return False


def conflict_matrix(items: Sequence[Item], ks: Knapsack,
                    allow_rotation: bool) -> Dict[int, Set[int]]:
    """conflicts[idx] = set of item idx that CANNOT coexist with it.
    Computed per type and broadcast to copies."""
    types: Dict[tuple, List[Item]] = {}
    for it in items:
        types.setdefault((it.w, it.d, it.h), []).append(it)
    keys = list(types)
    conf_t: Dict[tuple, Set[tuple]] = {k: set() for k in keys}
    for a in range(len(keys)):
        for b in range(a, len(keys)):
            ia, ib = types[keys[a]][0], types[keys[b]][0]
            if not compatible(ia, ib, ks, allow_rotation):
                conf_t[keys[a]].add(keys[b])
                conf_t[keys[b]].add(keys[a])
    conflicts: Dict[int, Set[int]] = {}
    for k, members in types.items():
        bad = set()
        for k2 in conf_t[k]:
            bad.update(m.idx for m in types[k2])
        for m in members:
            conflicts[m.idx] = bad - {m.idx} if k in conf_t[k] else set(bad)
            # same-type self conflict (two copies incompatible with
            # each other) is captured by k in conf_t[k]
            if k in conf_t[k]:
                conflicts[m.idx] |= {x.idx for x in members if x is not m}
    return conflicts


# ----------------------------------------------------------------------
def _chain_dim(item: Item, ks: Knapsack, allow_rotation: bool) -> int:
    """Big-item chain cut. If, in every feasible rotation, the item's
    two base dimensions exceed W/2 and D/2, then any two such items
    cannot share the base plane and the selected ones must stack along
    H: sum of their H-dims <= H. Returns the item's (minimal) H-dim in
    the chain, or 0 if the item is not forced into the chain."""
    best_h = None
    for (w, d, h) in item.rotations(allow_rotation, ks.is_3d):
        if not _fits_alone((w, d, h), ks):
            continue
        if 2 * w > ks.W and 2 * d > ks.D:
            best_h = h if best_h is None else min(best_h, h)
        else:
            return 0  # some rotation escapes the chain
    return best_h or 0


def conflict_aware_bound(items: Sequence[Item], ks: Knapsack,
                         allow_rotation: bool,
                         node_budget: int = 2_000_000) -> Optional[float]:
    """B&B upper bound: max profit s.t. volume capacity, pairwise
    compatibility, and the big-item chain cut (items oversized on both
    base axes must stack along H: sum of H-dims <= H). Returns None if
    the node budget is exhausted (caller falls back to the 1D bound)."""
    conf = conflict_matrix(items, ks, allow_rotation)
    meas = (lambda i: i.volume) if ks.is_3d else (lambda i: i.base_area)
    cap = ks.volume if ks.is_3d else ks.W * ks.D
    chain = {i.idx: (_chain_dim(i, ks, allow_rotation) if ks.is_3d else 0)
             for i in items}
    order = sorted(items, key=lambda i: -i.profit / meas(i))
    n = len(order)

    best = 0.0
    nodes = 0

    def frac_bound(k: int, rem: int) -> float:
        ub, r = 0.0, rem
        for t in range(k, n):
            m = meas(order[t])
            if m <= r:
                ub += order[t].profit
                r -= m
            else:
                ub += order[t].profit * r / m
                break
        return ub

    def dfs(k: int, rem: int, rem_h: int, profit: float,
            chosen: List[int]) -> bool:
        nonlocal best, nodes
        nodes += 1
        if nodes > node_budget:
            return False
        if profit > best:
            best = profit
        if k == n or rem <= 0:
            return True
        if profit + frac_bound(k, rem) <= best:
            return True
        it = order[k]
        ok = True
        m = meas(it)
        ch = chain[it.idx]
        if (m <= rem and ch <= rem_h
                and not (conf[it.idx] & set(chosen))):
            chosen.append(it.idx)
            ok = dfs(k + 1, rem - m, rem_h - ch, profit + it.profit, chosen)
            chosen.pop()
        if ok:
            ok = dfs(k + 1, rem, rem_h, profit, chosen)
        return ok

    completed = dfs(0, cap, ks.H, 0.0, [])
    return best if completed else None


# ----------------------------------------------------------------------
class ExchangeMaster:
    """Enumerates profit-improving marginal exchanges on the incumbent."""

    def __init__(self, items: Sequence[Item], ks: Knapsack,
                 allow_rotation: bool, max_out: int = 3, max_in: int = 4,
                 out_pool: int = 6, in_pool: int = 18,
                 slack: float = 0.02):
        self.items = list(items)
        self.ks = ks
        self.conf = conflict_matrix(items, ks, allow_rotation)
        self.meas = (lambda i: i.volume) if ks.is_3d else (lambda i: i.base_area)
        self.cap = ks.volume if ks.is_3d else ks.W * ks.D
        self.max_out, self.max_in = max_out, max_in
        self.out_pool, self.in_pool = out_pool, in_pool
        self.slack = slack
        self.nogood: Set[FrozenSet[int]] = set()

    def add_nogood(self, loaded_ids: FrozenSet[int]) -> None:
        self.nogood.add(loaded_ids)

    # ------------------------------------------------------------------
    def exchanges(self, incumbent: Packing,
                  top_k: int = 25) -> List[Tuple[float, List[Item], List[Item]]]:
        """Return up to top_k (gain, S_out, T_in) exchanges, best first.
        Filters: volume budget, pairwise compatibility inside T and
        between T and the kept loaded items, type-level dedup, no-goods
        on the resulting loaded set."""
        loaded_ids = incumbent.loaded_ids
        loaded = [i for i in self.items if i.idx in loaded_ids]
        unloaded = [i for i in self.items if i.idx not in loaded_ids]
        if not loaded or not unloaded:
            return []
        waste = self.cap - sum(self.meas(i) for i in loaded)

        dens = lambda i: i.profit / self.meas(i)
        out_cand = sorted(loaded, key=dens)[:self.out_pool]
        # one representative per type among the unloaded (copies are
        # interchangeable), best density first
        seen, in_cand = set(), []
        for i in sorted(unloaded, key=lambda x: -dens(x)):
            k = (i.w, i.d, i.h, i.profit)
            if k in seen:
                continue
            seen.add(k)
            in_cand.append(i)
            if len(in_cand) >= self.in_pool:
                break

        results = []
        out_subsets = [[]]
        for r in range(1, self.max_out + 1):
            out_subsets.extend(list(c) for c in combinations(out_cand, r))

        for S in out_subsets:
            s_ids = {i.idx for i in S}
            kept = [i for i in loaded if i.idx not in s_ids]
            kept_ids = {i.idx for i in kept}
            budget = (sum(self.meas(i) for i in S) + waste) * (1 + self.slack)
            p_out = sum(i.profit for i in S)
            # in-candidates compatible with every kept item
            ok_in = [i for i in in_cand
                     if not (self.conf[i.idx] & kept_ids)
                     and self.meas(i) <= budget]
            for r in range(1, self.max_in + 1):
                for T in combinations(ok_in, r):
                    vol_t = sum(self.meas(i) for i in T)
                    if vol_t > budget:
                        continue
                    gain = sum(i.profit for i in T) - p_out
                    if gain <= 0:
                        continue
                    # pairwise compatibility inside T
                    bad = False
                    for a in range(len(T)):
                        for b in range(a + 1, len(T)):
                            if T[b].idx in self.conf[T[a].idx]:
                                bad = True
                                break
                        if bad:
                            break
                    if bad:
                        continue
                    new_loaded = frozenset(kept_ids | {i.idx for i in T})
                    if new_loaded in self.nogood:
                        continue
                    results.append((gain, S, list(T)))

        results.sort(key=lambda t: -t[0])
        return results[:top_k]
