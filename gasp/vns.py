"""Pure VNS solver for Container Loading (Parreno et al. 2010 architecture).

EXPERIMENTAL STATUS: diagnostic, documented negative result. Measured
standalone on constrained BR7 it reaches only ~82.8%, BELOW GASP+layout
(~86.95%), because of a structural flaw this experiment exposed: the
constructive procedure and the five moves share the same deterministic
greedy fill, so shaking removes boxes and the rebuild puts them back
where they were -- the loop does not diversify and stalls at the
constructive's own fill (~79%). A genuine VNS needs a RANDOMIZED
(GRASP-style) maximal-space constructive to generate diversity at each
shake, which this module does not implement.

The diagnostic value is decisive: the residual ~5-point gap to the
top-tier (VNS/GRASP at ~91-92% on BR7) is NOT closed by the VNS
architecture per se, nor by tuning the five moves -- GASP+layout is
already the best of our architectural family. The top-tier advantage
must come from the randomized maximal-space construction (the diversity
engine our VNS lacks) plus moves tuned on the problem for years. This is
where future effort on BR should go, not into the score-ordering or the
post-process, both of which are near their ceiling.

NOTE ON THE CONSTRUCTIVE (work in progress): the block constructive now
follows the Parreno et al. (2008) criteria -- lexicographic corner
distance for space selection (Step 1) and best-fit block selection
(Step 2) -- and reaches ~78% on BR1 but still underperforms on BR7. Full
fidelity needs the complete enumeration of the six layer configurations
(their Fig. 4), corner-anchored growth (not always origin), and the
reactive-GRASP on delta. This is effectively a faithful re-implementation
of their GRASP, a multi-session task, not a drop-in.

Architecture (for reference): maximal-space construction, VND over the
five physical-layout moves, shaking (remove 10-30% and rebuild), looped
to a time budget. Reuses the EMS engine and the moves of layout_search.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .ems import _remove_dominated_t, _split_t
from .geometry import Item, Knapsack, Packing, Placement
from .layout_search import LayoutSearch, _greedy_fill, spaces_from_packing


def _apply_box(spaces, x, y, z, x2, y2, z2):
    """Difference process for a placed box footprint."""
    nxt = []
    for sp in spaces:
        if (sp[3] <= x or x2 <= sp[0] or sp[4] <= y or y2 <= sp[1]
                or sp[5] <= z or z2 <= sp[2]):
            nxt.append(sp)
        else:
            nxt.extend(_split_t(sp, x, y, z, x2, y2, z2))
    return _remove_dominated_t(nxt)


def _space_distance(s, ks):
    """Lexicographic distance of a maximal space to a container corner
    (Parreno et al. Step 1): for the space vertex nearest a container
    corner, the vector of the three coordinate differences sorted
    non-decreasing. The space with the minimum such vector is filled
    first, so corners are filled before sides before the inner volume."""
    sx, sy, sz, sx2, sy2, sz2 = s
    corners_s = [(sx, sy, sz), (sx2, sy, sz), (sx, sy2, sz), (sx, sy, sz2),
                 (sx2, sy2, sz), (sx2, sy, sz2), (sx, sy2, sz2),
                 (sx2, sy2, sz2)]
    cont = [(0, 0, 0), (ks.W, 0, 0), (0, ks.D, 0), (0, 0, ks.H),
            (ks.W, ks.D, 0), (ks.W, 0, ks.H), (0, ks.D, ks.H),
            (ks.W, ks.D, ks.H)]
    best = None
    best_corner = None
    for a in corners_s:
        for c in cont:
            v = tuple(sorted((abs(a[0]-c[0]), abs(a[1]-c[1]), abs(a[2]-c[2]))))
            if best is None or v < best:
                best = v
                best_corner = a
    return best, best_corner


def _block_configs(rep, members, s, allow_rotation, is_3d):
    """All column/layer configurations of a box type in a space, with
    their best-fit distance vector. A configuration is (nx,ny,nz) copies
    of an oriented box (w,d,h); its distance is the per-axis gap between
    the block extent and the space, sorted non-decreasing (Parreno Step
    2, criterion ii: best fit). Yields (distance, w, d, h, nx, ny, nz,
    ncopies)."""
    sx, sy, sz, sx2, sy2, sz2 = s
    fw, fd, fh = sx2 - sx, sy2 - sy, sz2 - sz
    out = []
    for (w, d, h) in rep.rotations(allow_rotation, is_3d):
        if w > fw or d > fd or h > fh:
            continue
        maxx, maxy, maxz = fw // w, fd // d, fh // h
        # consider columns (one axis) and layers (two axes); cap copies
        # by available members. Enumerate a modest set of configurations:
        # full extent on each subset of axes.
        for nx in (1, maxx):
            for ny in (1, maxy):
                for nz in (1, maxz):
                    ncopies = min(len(members), nx * ny * nz)
                    if ncopies < 1:
                        continue
                    # actual block extent given ncopies filled in order
                    bw, bd, bh = nx * w, ny * d, nz * h
                    dist = tuple(sorted((fw - bw, fd - bd, fh - bh)))
                    out.append((dist, w, d, h, nx, ny, nz, ncopies))
    return out


def grasp_construct(items: List[Item], ks: Knapsack, allow_rotation: bool,
                    rng: random.Random, delta: float = 0.3) -> Packing:
    """Type-based block GRASP constructive (Parreno et al. 2008, Sec. 2-3).

    Iterate: choose the maximal space nearest a container corner
    (lexicographic distance, volume tie-break); among all box types and
    their column/layer configurations fitting it, build a restricted
    candidate list of the best-fitting 100*delta%, pick one at random,
    pack the block at the space's near corner, run the difference
    process. Block selection uses best-fit (criterion ii); the
    randomization is over type+configuration."""
    by_type: Dict[tuple, List[Item]] = defaultdict(list)
    for it in items:
        by_type[(it.w, it.d, it.h)].append(it)
    avail = {k: list(v) for k, v in by_type.items()}
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    placements: List[Placement] = []

    while spaces:
        # Step 1: pick the space with min lexicographic corner distance,
        # volume as tie-break (larger first)
        best_s = None
        best_key = None
        for s in spaces:
            dist, corner = _space_distance(s, ks)
            vol = (s[3]-s[0]) * (s[4]-s[1]) * (s[5]-s[2])
            key = (dist, -vol)
            if best_key is None or key < best_key:
                best_key = key
                best_s = (s, corner)
        s, corner = best_s
        sx, sy, sz, sx2, sy2, sz2 = s

        # Step 2: gather all configurations of all fitting types
        cands = []  # (dist, key, w,d,h, nx,ny,nz, ncopies)
        for tkey, members in avail.items():
            if not members:
                continue
            rep = members[0]
            for cfg in _block_configs(rep, members, s, allow_rotation, ks.is_3d):
                dist = cfg[0]
                cands.append((dist, tkey) + cfg[1:])
        if not cands:
            # no type fits this space: drop it and continue
            spaces = [sp for sp in spaces if sp != s]
            continue

        # RCL: best-fit 100*delta% (smallest distance), random pick
        cands.sort(key=lambda c: c[0])
        k = max(1, int(delta * len(cands)))
        choice = rng.choice(cands[:k])
        _dist, tkey, w, d, h, nx, ny, nz, ncopies = choice
        members = avail[tkey]

        # pack the block at the chosen near corner. The near corner tells
        # us which way to grow; we grow from the space's min corner for
        # simplicity (origin-anchored), which matches filling the corner.
        ci = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if ci >= ncopies:
                        break
                    it = members[ci]
                    placements.append(Placement(
                        it, sx + ix*w, sy + iy*d, sz + iz*h, w, d, h))
                    ci += 1
        avail[tkey] = members[ncopies:]
        bx2, by2, bz2 = sx + nx*w, sy + ny*d, sz + nz*h
        spaces = _apply_box(spaces, sx, sy, sz, bx2, by2, bz2)

    return Packing(ks, placements)


def _maximal_space_construct(items: List[Item], ks: Knapsack,
                             allow_rotation: bool,
                             order: Optional[List[Item]] = None) -> Packing:
    """Deterministic single-item maximal-space fill (kept for reference;
    the type-based grasp_construct is the productive one)."""
    pool = order if order is not None else sorted(items,
                                                  key=lambda i: -i.volume)
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    placements = _greedy_fill(spaces, pool, ks, allow_rotation)
    return Packing(ks, placements)


def solve_vns(items: List[Item], ks: Knapsack,
              allow_rotation: bool = True,
              time_limit: float = 10.0,
              seed: int = 1,
              delta: float = 0.3,
              vnd_budget: float = 1.0) -> Packing:
    """Pure GRASP/VNS for one container loading problem.

    Each iteration: type-based GRASP construction (column loading,
    randomized type choice) -> VND over the five layout moves -> accept
    if better. The randomized constructive supplies the structural
    diversity that a deterministic fill cannot, so the descent has
    genuinely different starting points to improve."""
    if not ks.is_3d:
        return grasp_construct(items, ks, allow_rotation,
                               random.Random(seed), delta)

    rng = random.Random(seed)
    t0 = time.time()
    searcher = LayoutSearch(ks, items, allow_rotation, depth=1,
                            max_seeds=12, time_budget=vnd_budget)

    best = grasp_construct(items, ks, allow_rotation, rng, delta)
    best = searcher.improve(best)

    while time.time() - t0 < time_limit:
        remaining = time_limit - (time.time() - t0)
        if remaining <= 0:
            break
        cand = grasp_construct(items, ks, allow_rotation, rng, delta)
        searcher.time_budget = min(vnd_budget, remaining)
        cand = searcher.improve(cand)
        if cand.used_volume > best.used_volume:
            best = cand

    return best
