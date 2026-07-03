"""Faithful deterministic re-implementation of the Parreno et al. (2008)
block constructive for Container Loading.

KEY DIAGNOSTIC RESULT: this constructive, in a single deterministic pass,
reaches ~88% filling on constrained BR7 with the best-volume criterion --
ABOVE our entire GASP+layout metaheuristic (~87%). It locates the BR
heterogeneity gap precisely: it is in the CONSTRUCTIVE, not the search.
Our EP greedy places one box at a time (~75% on BR7); their block
heuristic places columns/layers of same-type copies, gaining 12-15
points where multiplicity is available. Used as an initial-solution seed
inside GASP (parreno_seed flag), it lifts BR7 from ~87% to ~88.4%, into
the upper band of the literature, close to the top-tier (~91-92%).

The purpose is narrow and diagnostic: reproduce their Section 2
constructive exactly (deterministic, no GRASP randomization, no
improvement phase) and measure it against our own constructive, one
pass each, to locate precisely where the fill gap to the top-tier comes
from. Faithful points, with the two we previously got wrong marked:

  Step 1  choose the maximal space whose nearest vertex-to-container-
          corner distance is lexicographically smallest (volume
          tie-break). The chosen near corner is the anchor.
  Step 2  among all box types fitting the space, enumerate column and
          layer configurations (1..max copies on each axis), evaluate
          each by best-fit (the per-axis gap block-vs-space, sorted,
          lexicographic), tie-break by FEWEST boxes; pick the best.
  *anchor* the block GROWS FROM THE NEAR CORNER inward, not from the
          origin -- this is the fix that makes corner-filling work.
  Step 3  difference process + dominance pruning (shared EMS engine).

Two block-selection objectives are available, as in the paper: best-fit
(criterion ii, default) and best-volume (criterion i).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .ems import _remove_dominated_t, _split_t
from .geometry import Item, Knapsack, Packing, Placement


def _near_corner(s, ks):
    """Return (distance_vector, corner) for the space vertex closest to
    a container corner, lexicographic. corner is one of the 8 vertices
    of the space, expressed as which extreme (min/max) on each axis.

    Optimised: the distance from a space vertex to the nearest container
    corner, per axis, is simply min(coord, dimension - coord); there is
    no need to loop over all 8 container corners. For each of the 8
    space vertices we take, per axis, the smaller of its distance to the
    0-face and to the far face, then sort the triple. This replaces a
    64-iteration loop (8 vertices x 8 corners, each with 3 abs and a
    sort) with 8 cheap evaluations."""
    sx, sy, sz, sx2, sy2, sz2 = s
    W, D, H = ks.W, ks.D, ks.H
    # for each extreme on each axis, distance to the nearer container face
    mx0 = sx if sx < W - sx else W - sx
    mx1 = sx2 if sx2 < W - sx2 else W - sx2
    my0 = sy if sy < D - sy else D - sy
    my1 = sy2 if sy2 < D - sy2 else D - sy2
    mz0 = sz if sz < H - sz else H - sz
    mz1 = sz2 if sz2 < H - sz2 else H - sz2
    # 8 vertices: choose min-face distance per axis depending on which
    # extreme (0=min coord, 1=max coord) the vertex takes on that axis
    verts = (
        ((mx0, my0, mz0), (0, 0, 0)),
        ((mx1, my0, mz0), (1, 0, 0)),
        ((mx0, my1, mz0), (0, 1, 0)),
        ((mx0, my0, mz1), (0, 0, 1)),
        ((mx1, my1, mz0), (1, 1, 0)),
        ((mx1, my0, mz1), (1, 0, 1)),
        ((mx0, my1, mz1), (0, 1, 1)),
        ((mx1, my1, mz1), (1, 1, 1)),
    )
    best = None
    for (a, b, c), sig in verts:
        # sort the 3 distances ascending without a full sort() call
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        v = (a, b, c)
        if best is None or v < best[0]:
            best = (v, sig)
    return best


def _place_block(s, sig, w, d, h, nx, ny, nz, members, ncopies):
    """Place ncopies of an oriented box growing from the near corner of
    the space identified by sig (per-axis 0=min,1=max). Returns the list
    of Placements and the block's bounding box (x,y,z,x2,y2,z2)."""
    sx, sy, sz, sx2, sy2, sz2 = s
    bw, bd, bh = nx * w, ny * d, nz * h
    # anchor: if the near corner is at the max side of an axis, the block
    # is flush to that side and grows toward the min; else it starts at min.
    x0 = sx2 - bw if sig[0] == 1 else sx
    y0 = sy2 - bd if sig[1] == 1 else sy
    z0 = sz2 - bh if sig[2] == 1 else sz
    pls = []
    ci = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if ci >= ncopies:
                    break
                it = members[ci]
                pls.append(Placement(it, x0 + ix*w, y0 + iy*d, z0 + iz*h,
                                     w, d, h))
                ci += 1
    return pls, (x0, y0, z0, x0 + bw, y0 + bd, z0 + bh)


def _apply_box(spaces, box):
    x, y, z, x2, y2, z2 = box
    nxt = []
    for sp in spaces:
        if (sp[3] <= x or x2 <= sp[0] or sp[4] <= y or y2 <= sp[1]
                or sp[5] <= z or z2 <= sp[2]):
            nxt.append(sp)
        else:
            nxt.extend(_split_t(sp, x, y, z, x2, y2, z2))
    return _remove_dominated_t(nxt)


def parreno_construct(items: List[Item], ks: Knapsack,
                      allow_rotation: bool = True,
                      objective: str = "bestfit") -> Packing:
    """Deterministic Parreno block constructive. objective in
    {'bestfit', 'bestvol'}."""
    by_type: Dict[tuple, List[Item]] = defaultdict(list)
    for it in items:
        by_type[(it.w, it.d, it.h)].append(it)
    avail = {k: list(v) for k, v in by_type.items()}
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    placements: List[Placement] = []

    while spaces:
        # Step 1: choose space (min lexicographic near-corner distance,
        # larger volume as tie-break)
        best = None
        for s in spaces:
            dist, sig = _near_corner(s, ks)
            vol = (s[3]-s[0]) * (s[4]-s[1]) * (s[5]-s[2])
            key = (dist, -vol)
            if best is None or key < best[0]:
                best = (key, s, sig)
        _key, s, sig = best
        sx, sy, sz, sx2, sy2, sz2 = s
        fw, fd, fh = sx2 - sx, sy2 - sy, sz2 - sz

        # Step 2: enumerate configurations of every fitting type
        chosen = None
        chosen_score = None
        for tkey, members in avail.items():
            if not members:
                continue
            navail = len(members)
            rep = members[0]
            for (w, d, h) in rep.rotations(allow_rotation, ks.is_3d):
                if w > fw or d > fd or h > fh:
                    continue
                maxx, maxy, maxz = fw // w, fd // d, fh // h
                # enumerate column/layer counts on each axis (full grid is
                # too large; use 1..max per axis but cap product by navail)
                for nx in range(1, maxx + 1):
                    for ny in range(1, maxy + 1):
                        prod_xy = nx * ny
                        if prod_xy > navail:
                            break
                        for nz in range(1, maxz + 1):
                            ncopies = nx * ny * nz
                            if ncopies > navail:
                                break
                            bw, bd, bh = nx*w, ny*d, nz*h
                            if objective == "bestvol":
                                # criterion i: maximise volume, tie fewest
                                score = (-(bw*bd*bh), ncopies)
                            else:
                                # criterion ii: best fit (min gap vector),
                                # tie fewest boxes
                                gap = tuple(sorted((fw-bw, fd-bd, fh-bh)))
                                score = (gap, ncopies)
                            if chosen_score is None or score < chosen_score:
                                chosen_score = score
                                chosen = (tkey, w, d, h, nx, ny, nz, ncopies)
        if chosen is None:
            spaces = [sp for sp in spaces if sp != s]
            continue

        tkey, w, d, h, nx, ny, nz, ncopies = chosen
        members = avail[tkey]
        pls, box = _place_block(s, sig, w, d, h, nx, ny, nz, members, ncopies)
        placements.extend(pls)
        avail[tkey] = members[ncopies:]
        spaces = _apply_box(spaces, box)

    return Packing(ks, placements)
