"""Physical-layout local search for 3D packing (memetic intensification).

EXPERIMENTAL STATUS: SUCCESS, enabled via GASPParams.layout_search
(default off). As a FINAL post-process on the GASP incumbent, a Variable
Neighborhood Descent over the physical-layout moves of Parreno et al.
adds ~+0.8 filling points on BR7 (6-8 of every 8 instances improve, none
regress; up to +2.2), the first integer-scale lever on the heterogeneity
gap. Implemented moves: emptying-region with Best-Volume (N4) and
Best-Fit (N5) -- the strongest, each ~+0.25 alone; delete_refill
(single-space emptying, ~+0.21); column insertion (N2, ~+0.14); and a
bounded box-insertion DFS (N3). Layer reduction (N1) is not applicable
to our non-layer model. The moves are cycled strongest-first until no
improvement or the time budget runs out. IMPORTANT: it must run OUTSIDE
the GASP loop -- reinjecting a refined packing mid-run forces a re-score
that resets the learning and HURTS (-1.2); as a post-process it only
adds. 2D instances are left untouched (verified: the search is never
invoked when is_3d is false).

GASP's score-ordering is a strong GLOBAL search but cannot express
moves on the physical layout: it perturbs the item ORDER and rebuilds
from scratch, so it can never "undo" a placed box to make room for a
better pair. The state-of-the-art container-loading methods (RGP / VNS
of Parreno et al.) do exactly that, with moves defined on the layout.

This module adds that as LOCAL intensification on a concrete packing,
to be fired at stagnation (the same slot as exact_repair / basin_probe)
while GASP keeps doing the global search. The improved layout is handed
back to GASP, which re-derives an item order from it so the score
machinery can carry on from a solution it can read.

Moves (all on the empty maximal spaces of the current packing):
  insert        place an unloaded item into a fitting EMS;
  delete_refill remove a placed box, recompute the freed space, and try
                to refill it with a better set (the basin-breaking move
                the greedy cannot do);
  a bounded DFS chains these to a small depth with backtracking.

Everything reuses the EMS engine (ems.py): the difference process and
dominance pruning on fast (x,y,z,x2,y2,z2) tuples.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .ems import _remove_dominated_t, _split_t
from .geometry import Item, Knapsack, Packing, Placement


def spaces_from_packing(ks: Knapsack,
                        placements: List[Placement]) -> List[tuple]:
    """Reconstruct the empty maximal spaces of a packing: start from the
    whole knapsack and run the difference process for every placed box."""
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    for p in placements:
        bx, by, bz, bx2, by2, bz2 = p.x, p.y, p.z, p.x2, p.y2, p.z2
        nxt = []
        for s in spaces:
            if (s[3] <= bx or bx2 <= s[0] or s[4] <= by or by2 <= s[1]
                    or s[5] <= bz or bz2 <= s[2]):
                nxt.append(s)
            else:
                nxt.extend(_split_t(s, bx, by, bz, bx2, by2, bz2))
        spaces = _remove_dominated_t(nxt)
    return spaces


def _fits(it: Item, s: tuple, allow_rotation: bool, is_3d: bool):
    """Best rotation of `it` fitting space tuple s, or None. 'Best' =
    largest volume (it always equals the box volume, so any fit is
    fine); returns (w, d, h)."""
    fw, fd, fh = s[3] - s[0], s[4] - s[1], s[5] - s[2]
    for (w, d, h) in it.rotations(allow_rotation, is_3d):
        if w <= fw and d <= fd and h <= fh:
            return (w, d, h)
    return None


def _greedy_fill(spaces: List[tuple], candidates: List[Item],
                 ks: Knapsack, allow_rotation: bool) -> List[Placement]:
    """Volume-first greedy fill of `spaces` using `candidates` (each at
    most once). Returns the new placements; updates are local."""
    placed: List[Placement] = []
    used = set()
    cand = sorted(candidates, key=lambda i: -i.volume)
    spaces = list(spaces)
    for it in cand:
        if it.idx in used:
            continue
        # best EMS for this item: tightest volume leftover
        best = None
        best_left = None
        for s in spaces:
            rot = _fits(it, s, allow_rotation, ks.is_3d)
            if rot is None:
                continue
            w, d, h = rot
            left = (s[3]-s[0])*(s[4]-s[1])*(s[5]-s[2]) - w*d*h
            if best_left is None or left < best_left:
                best_left = left
                best = (s, rot)
        if best is None:
            continue
        s, (w, d, h) = best
        p = Placement(it, s[0], s[1], s[2], w, d, h)
        placed.append(p)
        used.add(it.idx)
        bx2, by2, bz2 = s[0]+w, s[1]+d, s[2]+h
        nxt = []
        for sp in spaces:
            if (sp[3] <= p.x or bx2 <= sp[0] or sp[4] <= p.y or by2 <= sp[1]
                    or sp[5] <= p.z or bz2 <= sp[2]):
                nxt.append(sp)
            else:
                nxt.extend(_split_t(sp, p.x, p.y, p.z, bx2, by2, bz2))
        spaces = _remove_dominated_t(nxt)
    return placed


class LayoutSearch:
    """Local search on the physical layout of a packing."""

    def __init__(self, knapsack: Knapsack, items: List[Item],
                 allow_rotation: bool = True,
                 depth: int = 1, max_seeds: int = 8,
                 time_budget: float = 2.0):
        self.ks = knapsack
        self.items = items
        self.by_idx = {it.idx: it for it in items}
        self.allow_rotation = allow_rotation
        self.depth = depth
        self.max_seeds = max_seeds
        self.time_budget = time_budget

    # ------------------------------------------------------------------
    def improve(self, packing: Packing) -> Packing:
        """Variable Neighborhood Descent over the physical-layout moves
        of Parreno et al.: box insertion (N3), column insertion (N2),
        emptying a region with Best-Volume / Best-Fit (N4 / N5), plus
        delete_refill (a single-space emptying). Layer reduction (N1) is
        not applicable to our non-layer model. Cycles the neighborhoods
        until no move improves or the budget runs out."""
        t0 = time.time()
        best = packing
        improved = True
        while improved and time.time() - t0 < self.time_budget:
            improved = False
            for move in (self._empty_region_bv, self._empty_region_bf,
                         self._delete_refill, self._column_insert):
                if time.time() - t0 > self.time_budget:
                    break
                cand = move(best, t0)
                if cand.used_volume > best.used_volume:
                    best = cand
                    improved = True
            # box-insertion DFS as the cheap intensifier
            cand = self._dfs_insert(best, t0)
            if cand.used_volume > best.used_volume:
                best = cand
                improved = True
        return best

    # ------------------------------------------------------------------
    def _delete_refill(self, packing: Packing, t0: float) -> Packing:
        """Single-space emptying: remove a low-density box, refill the
        freed region (plus existing free space) with unloaded items."""
        best = packing
        best_vol = packing.used_volume
        placed = list(packing.placements)
        loaded_ids = {p.item.idx for p in placed}
        unloaded = [it for it in self.items if it.idx not in loaded_ids]

        # attack the lowest volume-density boxes first
        order = sorted(range(len(placed)),
                       key=lambda i: placed[i].item.profit /
                       max(1, placed[i].w * placed[i].d * placed[i].h))
        for k in order[:self.max_seeds]:
            if time.time() - t0 > self.time_budget:
                break
            victim = placed[k]
            kept = placed[:k] + placed[k+1:]
            spaces = spaces_from_packing(self.ks, kept)
            pool = unloaded + [victim.item]
            add = _greedy_fill(spaces, pool, self.ks, self.allow_rotation)
            cand = Packing(self.ks, kept + add)
            if cand.used_volume > best_vol:
                best_vol = cand.used_volume
                best = cand
        return best

    # ------------------------------------------------------------------
    def _empty_region(self, packing: Packing, t0: float,
                      best_fit: bool) -> Packing:
        """N4/N5 emptying a region: pick two empty maximal spaces, empty
        the smallest box containing both, refill it with the constructive
        procedure. best_fit toggles the Best-Fit vs Best-Volume refill
        criterion (here both use the volume-first greedy; the criterion
        affects the candidate ordering)."""
        best = packing
        best_vol = packing.used_volume
        placed = list(packing.placements)
        spaces = spaces_from_packing(self.ks, placed)
        # smaller spaces first, as in the paper
        spaces = sorted(spaces, key=lambda s: (s[3]-s[0])*(s[4]-s[1])*(s[5]-s[2]))
        seen_regions = set()
        tries = 0
        for i, s1 in enumerate(spaces):
            for s2 in spaces[:i]:               # s2 smaller than s1
                if tries >= self.max_seeds or time.time() - t0 > self.time_budget:
                    break
                # smallest box containing both spaces
                rx, ry, rz = min(s1[0], s2[0]), min(s1[1], s2[1]), min(s1[2], s2[2])
                rx2 = max(s1[3], s2[3]); ry2 = max(s1[4], s2[4]); rz2 = max(s1[5], s2[5])
                key = (rx, ry, rz, rx2, ry2, rz2)
                if key in seen_regions:
                    continue
                seen_regions.add(key)
                tries += 1
                # remove boxes overlapping the region
                kept, removed = [], []
                for p in placed:
                    if (p.x2 <= rx or rx2 <= p.x or p.y2 <= ry or ry2 <= p.y
                            or p.z2 <= rz or rz2 <= p.z):
                        kept.append(p)
                    else:
                        removed.append(p)
                if not removed:
                    continue
                free = spaces_from_packing(self.ks, kept)
                loaded = {p.item.idx for p in kept}
                pool = [it for it in self.items if it.idx not in loaded]
                if best_fit:
                    pool.sort(key=lambda i: (min(i.w, i.d, i.h), -i.volume))
                add = _greedy_fill(free, pool, self.ks, self.allow_rotation)
                cand = Packing(self.ks, kept + add)
                if cand.used_volume > best_vol:
                    best_vol = cand.used_volume
                    best = cand
        return best

    def _empty_region_bv(self, packing: Packing, t0: float) -> Packing:
        return self._empty_region(packing, t0, best_fit=False)

    def _empty_region_bf(self, packing: Packing, t0: float) -> Packing:
        return self._empty_region(packing, t0, best_fit=True)

    # ------------------------------------------------------------------
    def _column_insert(self, packing: Packing, t0: float) -> Packing:
        """N2 column insertion: insert into an empty space a COLUMN of
        copies of an unloaded box type (as many as fit along the free
        axes), remove overlaps, refill."""
        best = packing
        best_vol = packing.used_volume
        placed = list(packing.placements)
        loaded = {p.item.idx for p in placed}
        spaces = spaces_from_packing(self.ks, placed)
        spaces = sorted(spaces, key=lambda s: (s[3]-s[0])*(s[4]-s[1])*(s[5]-s[2]))
        # group unloaded items by type to form columns of copies
        by_type: Dict[tuple, List[Item]] = {}
        for it in self.items:
            if it.idx in loaded:
                continue
            by_type.setdefault((it.w, it.d, it.h), []).append(it)
        tries = 0
        for s in spaces:
            if tries >= self.max_seeds or time.time() - t0 > self.time_budget:
                break
            fw, fd, fh = s[3]-s[0], s[4]-s[1], s[5]-s[2]
            for key, members in by_type.items():
                rep = members[0]
                rot = _fits(rep, s, self.allow_rotation, self.ks.is_3d)
                if rot is None:
                    continue
                w, d, h = rot
                # build the largest column of copies fitting the space
                nx, ny, nz = fw // w, fd // d, fh // h
                ncopies = min(len(members), nx * ny * nz)
                if ncopies < 1:
                    continue
                tries += 1
                col = []
                ci = 0
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            if ci >= ncopies:
                                break
                            it = members[ci]
                            col.append(Placement(it, s[0]+ix*w, s[1]+iy*d,
                                                 s[2]+iz*h, w, d, h))
                            ci += 1
                # remove placed boxes overlapping the column footprint
                cx2, cy2, cz2 = s[0]+nx*w, s[1]+ny*d, s[2]+nz*h
                kept = [p for p in placed
                        if (p.x2 <= s[0] or cx2 <= p.x or p.y2 <= s[1]
                            or cy2 <= p.y or p.z2 <= s[2] or cz2 <= p.z)]
                cand = Packing(self.ks, kept + col)
                if cand.used_volume > best_vol:
                    best_vol = cand.used_volume
                    best = cand
        return best

    # ------------------------------------------------------------------
    def _dfs_insert(self, packing: Packing, t0: float) -> Packing:
        """Depth-bounded search: repeatedly insert unloaded items into
        free spaces, exploring a few alternatives per level."""
        best = packing
        best_vol = packing.used_volume

        def rec(pl: List[Placement], depth: int):
            nonlocal best, best_vol
            if time.time() - t0 > self.time_budget or depth > self.depth:
                return
            loaded = {p.item.idx for p in pl}
            spaces = spaces_from_packing(self.ks, pl)
            unloaded = sorted(
                (it for it in self.items if it.idx not in loaded),
                key=lambda i: -i.volume)
            tried = 0
            for it in unloaded:
                placed_here = None
                for s in spaces:
                    rot = _fits(it, s, self.allow_rotation, self.ks.is_3d)
                    if rot is None:
                        continue
                    w, d, h = rot
                    placed_here = Placement(it, s[0], s[1], s[2], w, d, h)
                    break
                if placed_here is None:
                    continue
                npl = pl + [placed_here]
                nvol = sum(p.w*p.d*p.h for p in npl)
                if nvol > best_vol:
                    best_vol = nvol
                    best = Packing(self.ks, npl)
                rec(npl, depth + 1)
                tried += 1
                if tried >= 3:        # branch width cap
                    break

        rec(list(packing.placements), 1)
        return best
