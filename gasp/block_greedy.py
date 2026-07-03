"""EP greedy with block (group) loading.

EXPERIMENTAL RESULT (both strategies compared): block loading helps the
PURE CONSTRUCTIVE -- at parity of ordering, best-fit gains up to +7
points on BR4 and modestly elsewhere, max gains a bit on BR7 -- but
DOES NOT help inside GASP. End-to-end on constrained BR7, neither
block_mode improves over the plain EP greedy (88.4% base vs 88.3 max vs
88.2 bestfit), and used alone (without parreno_seed) block-fit is worse
(83% vs 88.4%). The reason is structural and recurs throughout this
project: the block constructive is rigid, reduces the diversity GASP's
learning exploits, and the parreno_seed already supplies the block
structure once, as a seed, which is where it belongs. Conclusion: load
blocks as the INITIAL SOLUTION (parreno_seed), not as the per-iteration
greedy backend. Of the two strategies, best-fit is the stronger
constructive; both are kept behind GASPParams.block_mode (off|max|
bestfit, default off) for the pure-constructive use case.

The classical EP greedy places one box at a time. When a box TYPE has
several copies available, the state-of-the-art container-loading
constructives place a GROUP -- a column or layer of copies -- in one
shot, which tiles dense regions far better on multiplicity-rich
instances (BR). This module incorporates that decision INTO the EP
greedy itself: at each placement, depending on the type's available
multiplicity, it may place a block instead of a single box.

Two strategies, compared experimentally:

  block_mode="max"      strategy 1: whenever the chosen type has >1 copy
                        available, place the largest block (column/layer)
                        that fits the extreme point. Pure block greedy.

  block_mode="bestfit"  strategy 2: at each placement evaluate BOTH the
                        single box and the candidate blocks, and keep
                        whichever scores best by the existing merit
                        (tight fit wins). The block competes with the
                        single and is chosen only when it fills better;
                        single types naturally never form blocks.

block_mode="off" reproduces the classical single-box greedy.

This is a Python implementation (no numba): used for the comparison and,
if a strategy wins, as an alternative backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .extreme_points import EPManager
from .geometry import Item, Knapsack, Packing, Placement
from .merit import merit_value


def _block_dims(w, d, h, ep, ks, navail):
    """Max copies of an oriented box along each axis from the EP, capped
    by available copies. Returns (nx, ny, nz) for the largest block that
    fits, and the per-axis maxima."""
    fw = ks.W - ep.x
    fd = ks.D - ep.y
    fh = ks.H - ep.z
    maxx = max(1, fw // w)
    maxy = max(1, fd // d)
    maxz = max(1, fh // h)
    return maxx, maxy, maxz


def _make_block(item_group, ep, w, d, h, nx, ny, nz, ncopies):
    """Build placements for a block of ncopies at the EP, growing in +x,
    +y, +z. item_group is the list of available Item copies."""
    out = []
    ci = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                if ci >= ncopies:
                    return out
                it = item_group[ci]
                out.append(Placement(it, ep.x + ix*w, ep.y + iy*d,
                                     ep.z + iz*h, w, d, h))
                ci += 1
    return out


class BlockGreedyState:
    """EP greedy that can place groups of same-type copies."""

    def __init__(self, knapsack: Knapsack, criterion: str = "RS",
                 allow_rotation: bool = False, block_mode: str = "bestfit",
                 min_multiplicity: int = 2):
        self.ks = knapsack
        self.criterion = criterion
        self.allow_rotation = allow_rotation
        self.block_mode = block_mode
        self.min_multiplicity = min_multiplicity
        self.packing = Packing(knapsack)
        self.epm = EPManager(knapsack)
        # remaining copies per type, populated in run()
        self._avail: Dict[tuple, List[Item]] = defaultdict(list)

    # ------------------------------------------------------------------
    def run(self, items_in_order: List[Item]) -> Packing:
        # group copies by type, preserving the requested order of types
        self._avail = defaultdict(list)
        for it in items_in_order:
            self._avail[(it.w, it.d, it.h)].append(it)
        placed_ids = set()

        for it in items_in_order:
            if it.idx in placed_ids:
                continue
            tkey = (it.w, it.d, it.h)
            group = [x for x in self._avail[tkey] if x.idx not in placed_ids]
            if not group:
                continue
            new = self._place(group, placed_ids)
            for p in new:
                placed_ids.add(p.item.idx)
        return self.packing

    # ------------------------------------------------------------------
    def _place(self, group: List[Item], placed_ids) -> List[Placement]:
        """Place a single box or a block from `group`, by the active
        strategy. Returns the placements made (possibly empty)."""
        rep = group[0]
        navail = len(group)
        is_3d = self.ks.is_3d
        block_on = (self.block_mode != "off"
                    and navail >= self.min_multiplicity and is_3d)

        best_single = None
        best_single_merit = None
        best_block = None
        best_block_score = None

        for (w, d, h) in rep.rotations(self.allow_rotation, is_3d):
            for order, ep in enumerate(self.epm.eps):
                cand = Placement(rep, ep.x, ep.y, ep.z, w, d, h)
                if not self.packing.feasible(cand):
                    continue
                m = merit_value(self.criterion, ep, w, d, h,
                                self.packing, order)
                if best_single_merit is None or m < best_single_merit:
                    best_single_merit = m
                    best_single = cand

                if block_on:
                    maxx, maxy, maxz = _block_dims(w, d, h, ep, self.ks,
                                                   navail)
                    if self.block_mode == "max":
                        # largest block fitting, capped by copies
                        for (nx, ny, nz) in self._max_block_config(
                                maxx, maxy, maxz, navail):
                            blk = self._try_block(group, ep, w, d, h,
                                                  nx, ny, nz)
                            if blk:
                                score = self._block_merit(ep, w, d, h,
                                                          nx, ny, nz, order)
                                if (best_block_score is None
                                        or score < best_block_score):
                                    best_block_score = score
                                    best_block = blk
                                break
                    else:  # bestfit: evaluate a few block shapes
                        for (nx, ny, nz) in self._candidate_configs(
                                maxx, maxy, maxz, navail):
                            blk = self._try_block(group, ep, w, d, h,
                                                  nx, ny, nz)
                            if not blk:
                                continue
                            score = self._block_merit(ep, w, d, h,
                                                      nx, ny, nz, order)
                            if (best_block_score is None
                                    or score < best_block_score):
                                best_block_score = score
                                best_block = blk

        # decide single vs block
        chosen = None
        if self.block_mode == "max" and best_block is not None:
            chosen = best_block
        elif self.block_mode == "bestfit":
            # block competes with single on the merit scale; a block of
            # k>1 boxes is preferred when it fills at least as tightly
            if best_block is not None and best_block_score is not None and \
               best_single_merit is not None and \
               best_block_score <= best_single_merit:
                chosen = best_block
            else:
                chosen = [best_single] if best_single else None
        else:
            chosen = [best_single] if best_single else None

        if not chosen:
            return []
        for p in chosen:
            self.packing.placements.append(p)
            self.epm.add_item(p, self.packing)
        return chosen

    # ------------------------------------------------------------------
    def _try_block(self, group, ep, w, d, h, nx, ny, nz):
        """Build a block and check it is feasible against the packing."""
        ncopies = min(len(group), nx * ny * nz)
        if ncopies < 2:
            return None
        blk = _make_block(group, ep, w, d, h, nx, ny, nz, ncopies)
        # block bounding box must be in-bounds and not overlap existing
        bx2 = ep.x + nx * w
        by2 = ep.y + ny * d
        bz2 = ep.z + nz * h
        if bx2 > self.ks.W or by2 > self.ks.D or bz2 > self.ks.H:
            return None
        # feasibility: the block footprint must not overlap placed boxes
        for p in self.packing.placements:
            if not (p.x2 <= ep.x or bx2 <= p.x or p.y2 <= ep.y
                    or by2 <= p.y or p.z2 <= ep.z or bz2 <= p.z):
                return None
        return blk

    def _block_merit(self, ep, w, d, h, nx, ny, nz, order):
        """Merit of a block: use the existing merit on the block's outer
        envelope, so a tight-filling block scores like a tight box."""
        return merit_value(self.criterion, ep, nx*w, ny*d, nz*h,
                           self.packing, order)

    @staticmethod
    def _max_block_config(maxx, maxy, maxz, navail):
        """Yield the single largest block configuration (strategy 1)."""
        # cap the product by navail, preferring to grow along all axes
        nx, ny, nz = maxx, maxy, maxz
        while nx * ny * nz > navail:
            # shrink the largest axis first
            if nz >= ny and nz >= nx and nz > 1:
                nz -= 1
            elif ny >= nx and ny > 1:
                ny -= 1
            elif nx > 1:
                nx -= 1
            else:
                break
        yield (nx, ny, nz)

    @staticmethod
    def _candidate_configs(maxx, maxy, maxz, navail):
        """Yield a small set of block shapes for best-fit (strategy 2):
        columns along each axis and the full layer, capped by navail."""
        seen = set()
        cands = [
            (maxx, 1, 1), (1, maxy, 1), (1, 1, maxz),
            (maxx, maxy, 1), (maxx, 1, maxz), (1, maxy, maxz),
            (maxx, maxy, maxz),
        ]
        for (nx, ny, nz) in cands:
            nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)
            while nx * ny * nz > navail:
                if nz >= ny and nz >= nx and nz > 1:
                    nz -= 1
                elif ny >= nx and ny > 1:
                    ny -= 1
                elif nx > 1:
                    nx -= 1
                else:
                    break
            if (nx, ny, nz) not in seen and nx*ny*nz >= 2:
                seen.add((nx, ny, nz))
                yield (nx, ny, nz)


def ep_kph_blocks(items_in_order: List[Item], knapsack: Knapsack,
                  criterion: str = "RS", allow_rotation: bool = False,
                  block_mode: str = "bestfit") -> Packing:
    """EP greedy with block loading. block_mode in {off, max, bestfit}."""
    st = BlockGreedyState(knapsack, criterion, allow_rotation, block_mode)
    return st.run(items_in_order)
