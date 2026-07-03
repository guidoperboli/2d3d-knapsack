"""Empty Maximal Spaces (EMS) placement for 3D packing.

EXPERIMENTAL STATUS: implemented, verified, and a measured partial
success, available behind GASPParams.use_ems (default off).

Findings (BR7, strongly heterogeneous):
  * EMS as a placement backend BEATS the EP greedy at parity of item
    ordering, by 4-8 filling points -- the first vocabulary extension
    to do so since composed projections. An EMS explicitly represents
    the free box to fill, which the point-based EP vocabulary cannot.
  * Inside the GASP loop, however, EMS only matches EP end-to-end: the
    score-ordering machinery was tuned around EP placement and does not
    steer EMS well (the scores order the items, but with EMS the
    decisive choice is which SPACE an item fills, not its position in
    the order). EMS also runs slower than the numba EP kernel, so it
    completes fewer iterations.
  * Pure EMS multi-restart (RGP-style, no learning) does worse than
    GASP-EP: the learning is what carries the method.

Conclusion: the EMS vocabulary is genuinely stronger, but realising its
advantage requires redesigning the score-ordering around the
space-item choice (essentially the RGP/VNS of Parreno et al.), not just
swapping the placement backend. This module provides a correct, fast
EMS engine and the backend hook; the score redesign is the open step.

The state-of-the-art constructive paradigm for Container Loading
(Lai & Chan 1997; RGP of Parreno et al. 2008; VNS of Parreno et al.
2010) maintains, instead of placement POINTS (the Extreme Points), the
list of empty maximal spaces: the largest empty boxes not contained in
any other. Placing an item splits every EMS it intersects into up to
six residual maximal sub-spaces (the difference process); dominated
spaces are then removed.

This module provides:
  EMS                 a free box (x, y, z, w, d, h)
  difference_process  split the EMS list around a placed box
  EMSGreedy           a placement backend interchangeable with EP-KPH
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .geometry import Item, Knapsack, Packing, Placement


# Below this many candidate spaces, the pure-Python dominance scan is
# faster than converting to a numpy array and calling the numba kernel;
# above it, the kernel wins (heterogeneous instances reach hundreds of
# spaces per step). Tuned empirically.
_DOM_NUMBA_THRESHOLD = 60


@dataclass(frozen=True)
class EMS:
    x: int
    y: int
    z: int
    w: int
    d: int
    h: int

    @property
    def x2(self) -> int: return self.x + self.w
    @property
    def y2(self) -> int: return self.y + self.d
    @property
    def z2(self) -> int: return self.z + self.h
    @property
    def volume(self) -> int: return self.w * self.d * self.h

    def contains(self, o: "EMS") -> bool:
        return (self.x <= o.x and self.y <= o.y and self.z <= o.z
                and self.x2 >= o.x2 and self.y2 >= o.y2 and self.z2 >= o.z2)


# Internal fast representation: a space is the 6-tuple
# (x, y, z, x2, y2, z2). The greedy and difference process operate on
# these tuples to avoid per-access property overhead and dataclass cost
# on the hot path (the dominance check runs tens of millions of times).
def _contains_t(a, b) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]
            and a[3] >= b[3] and a[4] >= b[4] and a[5] >= b[5])


def _split_t(s, bx, by, bz, bx2, by2, bz2):
    """Difference of space tuple s minus box; up to six residual maximal
    sub-spaces as tuples (x,y,z,x2,y2,z2)."""
    sx, sy, sz, sx2, sy2, sz2 = s
    out = []
    if bx > sx:  out.append((sx, sy, sz, bx, sy2, sz2))
    if bx2 < sx2: out.append((bx2, sy, sz, sx2, sy2, sz2))
    if by > sy:  out.append((sx, sy, sz, sx2, by, sz2))
    if by2 < sy2: out.append((sx, by2, sz, sx2, sy2, sz2))
    if bz > sz:  out.append((sx, sy, sz, sx2, sy2, bz))
    if bz2 < sz2: out.append((sx, sy, bz2, sx2, sy2, sz2))
    return out


def _remove_dominated_t(spaces, min_dim=1):
    """Drop tuples contained in another and those too thin to hold an
    item. Sorted by volume desc so larger ones are kept first.

    For large candidate lists the O(n^2) containment scan dominates the
    GRASP runtime on heterogeneous instances, so it is offloaded to a
    numba kernel (gasp.ems_numba). Small lists stay in pure Python,
    where the array-conversion overhead would not pay off."""
    cand = [s for s in spaces
            if s[3] - s[0] >= min_dim and s[4] - s[1] >= min_dim
            and s[5] - s[2] >= min_dim]
    cand.sort(key=lambda s: -((s[3]-s[0]) * (s[4]-s[1]) * (s[5]-s[2])))

    if len(cand) >= _DOM_NUMBA_THRESHOLD:
        from .ems_numba import _keep_mask, _HAVE_NUMBA
        if _HAVE_NUMBA:
            import numpy as _np
            arr = _np.asarray(cand, dtype=_np.int64)
            mask = _keep_mask(arr)
            return [cand[i] for i in range(len(cand)) if mask[i]]

    kept = []
    for s in cand:
        dominated = False
        for t in kept:
            if (t[0] <= s[0] and t[1] <= s[1] and t[2] <= s[2]
                    and t[3] >= s[3] and t[4] >= s[4] and t[5] >= s[5]):
                dominated = True
                break
        if not dominated:
            kept.append(s)
    return kept


def _overlaps_box(s: EMS, bx, by, bz, bx2, by2, bz2) -> bool:
    return not (s.x2 <= bx or bx2 <= s.x
                or s.y2 <= by or by2 <= s.y
                or s.z2 <= bz or bz2 <= s.z)


def _split_one(s: EMS, bx, by, bz, bx2, by2, bz2) -> List[EMS]:
    """Difference of EMS s minus the axis-aligned box; up to six
    maximal residual sub-spaces. Each is s clipped on one side of the
    box, keeping s full extent on the other two axes."""
    out = []
    # along X
    if bx > s.x:
        out.append(EMS(s.x, s.y, s.z, bx - s.x, s.d, s.h))
    if bx2 < s.x2:
        out.append(EMS(bx2, s.y, s.z, s.x2 - bx2, s.d, s.h))
    # along Y
    if by > s.y:
        out.append(EMS(s.x, s.y, s.z, s.w, by - s.y, s.h))
    if by2 < s.y2:
        out.append(EMS(s.x, by2, s.z, s.w, s.y2 - by2, s.h))
    # along Z
    if bz > s.z:
        out.append(EMS(s.x, s.y, s.z, s.w, s.d, bz - s.z))
    if bz2 < s.z2:
        out.append(EMS(s.x, s.y, bz2, s.w, s.d, s.z2 - bz2))
    return out


def _remove_dominated(spaces: List[EMS], min_dim: int = 1) -> List[EMS]:
    """Drop spaces contained in another, and spaces too small to hold
    any item (below min_dim on some axis). O(n^2) but n stays small
    thanks to this very pruning."""
    kept: List[EMS] = []
    spaces = [s for s in spaces
              if s.w >= min_dim and s.d >= min_dim and s.h >= min_dim]
    # sort by volume desc so larger spaces are tested first
    spaces.sort(key=lambda s: -s.volume)
    for s in spaces:
        if any(t.contains(s) for t in kept):
            continue
        kept.append(s)
    return kept


def difference_process(spaces: List[EMS], p: Placement,
                       min_dim: int = 1) -> List[EMS]:
    """Update the EMS list after placing box p."""
    bx, by, bz = p.x, p.y, p.z
    bx2, by2, bz2 = p.x2, p.y2, p.z2
    new_spaces: List[EMS] = []
    for s in spaces:
        if _overlaps_box(s, bx, by, bz, bx2, by2, bz2):
            new_spaces.extend(_split_one(s, bx, by, bz, bx2, by2, bz2))
        else:
            new_spaces.append(s)
    return _remove_dominated(new_spaces, min_dim)


# ----------------------------------------------------------------------
class EMSGreedy:
    """Placement backend over Empty Maximal Spaces, interchangeable with
    the EP greedy: consumes an ordered item list, returns a Packing.

    For each item (in the given order) it scans the EMS list for the
    best (space, rotation) by a fill-quality merit, places it at the
    space's near corner, and runs the difference process."""

    def __init__(self, knapsack: Knapsack, criterion: str = "VOL",
                 allow_rotation: bool = True):
        self.ks = knapsack
        self.criterion = criterion
        self.allow_rotation = allow_rotation

    # ------------------------------------------------------------------
    def run(self, items_in_order: List[Item]) -> Packing:
        ks = self.ks
        # spaces as fast tuples (x, y, z, x2, y2, z2)
        spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
        placements: List[Placement] = []
        bss = self.criterion == "BSS"

        for it in items_in_order:
            rots = it.rotations(self.allow_rotation, ks.is_3d)
            best = None
            best_score = None
            for s in spaces:
                sx, sy, sz, sx2, sy2, sz2 = s
                fw, fd, fh = sx2 - sx, sy2 - sy, sz2 - sz
                for (w, d, h) in rots:
                    if w > fw or d > fd or h > fh:
                        continue
                    if bss:
                        m0, m1, m2 = fw - w, fd - d, fh - h
                        if m1 < m0: m0, m1 = m1, m0
                        if m2 < m1: m1, m2 = m2, m1
                        if m1 < m0: m0, m1 = m1, m0
                        score = (m0, m1, m2, sz, sy, sx)
                    else:
                        score = (fw * fd * fh - w * d * h, sz, sy, sx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (sx, sy, sz, w, d, h)
            if best is None:
                continue
            bx, by, bz, w, d, h = best
            placements.append(Placement(it, bx, by, bz, w, d, h))
            bx2, by2, bz2 = bx + w, by + d, bz + h
            # difference process on tuples
            new_spaces = []
            for s in spaces:
                if (s[3] <= bx or bx2 <= s[0] or s[4] <= by or by2 <= s[1]
                        or s[5] <= bz or bz2 <= s[2]):
                    new_spaces.append(s)
                else:
                    new_spaces.extend(_split_t(s, bx, by, bz, bx2, by2, bz2))
            spaces = _remove_dominated_t(new_spaces)
            if not spaces:
                break

        return Packing(ks, placements)

    # ------------------------------------------------------------------
    def _best_placement(self, it: Item,
                        spaces: List[EMS]
                        ) -> Optional[Tuple[EMS, Tuple[int, int, int]]]:
        rots = it.rotations(self.allow_rotation, self.ks.is_3d)
        best, best_score = None, None
        for s in spaces:
            for (w, d, h) in rots:
                if w > s.w or d > s.d or h > s.h:
                    continue
                score = self._merit(s, w, d, h)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (s, (w, d, h))
        return best

    def _merit(self, s: EMS, w: int, d: int, h: int):
        """Lower is better. Returns a tuple for deterministic ordering.
        Default VOL prefers the space whose residual volume after
        placement is smallest (tight fit); BSS minimises the smallest
        leftover dimension (best-short-side). Both tie-break towards the
        lower-back-left corner for compact, deterministic packings."""
        if self.criterion == "BSS":
            margins = sorted((s.w - w, s.d - d, s.h - h))
            return (margins[0], margins[1], margins[2], s.z, s.y, s.x)
        leftover = s.volume - w * d * h
        return (leftover, s.z, s.y, s.x)
