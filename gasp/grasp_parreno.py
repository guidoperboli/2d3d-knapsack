"""Faithful GRASP for the container loading problem (Parreno et al. 2008).

This is a deliberately faithful re-implementation of the GRASP described
in the paper, intended for an apples-to-apples comparison AT PARITY OF
ITERATIONS with the published results. It is NOT the GASP pipeline: here
one ITERATION = one randomized construction + one improvement phase,
exactly as in the paper (Figure 6), so a stop criterion of "5000
iterations" means the same thing it means in the paper.

The constructive building blocks (lexicographic near-corner space
selection, block placement growing from the corner, maximal-space
update) are shared with parreno_construct.py. What this module adds:

  * a RANDOMIZED construction: all feasible block configurations for the
    chosen space are scored, the best 100*delta% form the RCL, and one
    is chosen at random (Section 3.1);
  * the REACTIVE-GRASP adaptation of delta over D={0.1..0.9}, updated
    every `react_period` iterations following Figure 6 (alpha=10);
  * the IMPROVEMENT phase: remove the final 50% of blocks and refill with
    the deterministic constructive, once per objective (volume, then
    best-fit), keeping the better; called only when the constructed
    value clears the threshold V >= Vworst + 0.5(Vbest - Vworst)
    (Section 3.2);
  * the outer loop with a MAX-ITERATIONS stop, as an alternative to a
    wall-clock limit. A time limit can also be given as a safety cap.

Use this for the container-packing comparison only. For everyday solving
the GASP pipeline (parreno_seed + layout_search) remains the default.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import Item, Knapsack, Packing, Placement
from .parreno_construct import (_near_corner, _place_block, _apply_box)


# Module-level cache of near-corner distance keyed by the space tuple.
# A space tuple (x,y,z,x2,y2,z2) always has the same near-corner
# distance for a given container, and _near_corner dominates the GRASP
# profile, so caching it across constructions is a large speed-up.
_dist_cache: dict = {}


@dataclass
class GRASPParrenoResult:
    best_packing: Packing
    best_fill: float
    iterations: int
    elapsed: float
    history: List[float] = field(default_factory=list)
    delta_final: Optional[float] = None


def _enumerate_configs(s, sig, fw, fd, fh, avail, allow_rotation, is_3d,
                       objective):
    """Return a list of candidate block configurations for space `s`,
    each as (score, tkey, w, d, h, nx, ny, nz, ncopies). Lower score is
    better. Shared scoring with the deterministic constructive."""
    cands = []
    for tkey, members in avail.items():
        navail = len(members)
        if navail == 0:
            continue
        rep = members[0]
        for (w, d, h) in rep.rotations(allow_rotation, is_3d):
            if w > fw or d > fd or h > fh:
                continue
            maxx, maxy, maxz = fw // w, fd // d, fh // h
            for nx in range(1, maxx + 1):
                for ny in range(1, maxy + 1):
                    if nx * ny > navail:
                        break
                    for nz in range(1, maxz + 1):
                        ncopies = nx * ny * nz
                        if ncopies > navail:
                            break
                        bw, bd, bh = nx * w, ny * d, nz * h
                        if objective == "bestvol":
                            score = (-(bw * bd * bh), ncopies)
                        elif objective == "bestprofit":
                            # maximise block profit = copies * unit profit
                            score = (-(ncopies * rep.profit),
                                     -(bw * bd * bh))
                        else:
                            gap = tuple(sorted((fw - bw, fd - bd, fh - bh)))
                            score = (gap, ncopies)
                        cands.append((score, tkey, w, d, h,
                                      nx, ny, nz, ncopies))
    return cands


def _construct(items, ks, allow_rotation, objective, delta, rng):
    """One randomized construction. delta in [0,1]: delta=0 is fully
    greedy (RCL of size 1), delta=1 is fully random. Returns
    (placements, block_order) where block_order lists the bounding boxes
    in placement order (needed by the improvement phase)."""
    by_type: Dict[tuple, List[Item]] = defaultdict(list)
    for it in items:
        by_type[(it.w, it.d, it.h)].append(it)
    avail = {k: list(v) for k, v in by_type.items()}
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    placements: List[Placement] = []
    blocks: List[tuple] = []

    while spaces:
        # Step 1: choose space by lexicographic near-corner distance.
        # The distance only depends on the space tuple, so memoise it
        # across steps (the same space is re-evaluated every step until
        # it is consumed). _near_corner dominates the profile otherwise.
        best = None
        for s in spaces:
            ck = (ks.W, ks.D, ks.H, s)
            cached = _dist_cache.get(ck)
            if cached is None:
                cached = _near_corner(s, ks)
                _dist_cache[ck] = cached
            dist, sig = cached
            vol = (s[3]-s[0]) * (s[4]-s[1]) * (s[5]-s[2])
            key = (dist, -vol)
            if best is None or key < best[0]:
                best = (key, s, sig)
        _key, s, sig = best
        sx, sy, sz, sx2, sy2, sz2 = s
        fw, fd, fh = sx2 - sx, sy2 - sy, sz2 - sz

        cands = _enumerate_configs(s, sig, fw, fd, fh, avail,
                                   allow_rotation, ks.is_3d, objective)
        if not cands:
            spaces = [sp for sp in spaces if sp != s]
            continue

        # Step 2 randomized: RCL = best 100*delta% of candidates by score
        cands.sort(key=lambda c: c[0])
        if delta <= 0.0:
            pick = cands[0]
        else:
            rcl_size = max(1, int(len(cands) * delta + 0.9999))
            pick = cands[rng.randint(0, rcl_size - 1)]

        _score, tkey, w, d, h, nx, ny, nz, ncopies = pick
        members = avail[tkey]
        pls, box = _place_block(s, sig, w, d, h, nx, ny, nz,
                                members, ncopies)
        placements.extend(pls)
        blocks.append((box, len(pls)))
        avail[tkey] = members[ncopies:]
        spaces = _apply_box(spaces, box)

    return placements, blocks


def _rebuild_from_partial(placements_keep, items, ks, allow_rotation,
                          objective):
    """Improvement helper: given a set of placements to keep, rebuild the
    free space and fill it deterministically with the given objective.
    Returns the full placement list."""
    # reconstruct available counts: total minus kept
    used = defaultdict(int)
    for p in placements_keep:
        used[(p.item.w, p.item.d, p.item.h)] += 1  # note: oriented dims
    # rebuild free spaces by carving every kept box out of the container
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    for p in placements_keep:
        spaces = _apply_box(spaces, (p.x, p.y, p.z, p.x2, p.y2, p.z2))

    # remaining items: those whose ids are not in the kept set
    kept_ids = {id(p.item) for p in placements_keep}
    # group remaining items by original type
    by_type: Dict[tuple, List[Item]] = defaultdict(list)
    for it in items:
        by_type[(it.w, it.d, it.h)].append(it)
    kept_item_ids = {p.item.idx for p in placements_keep}
    avail = {k: [it for it in v if it.idx not in kept_item_ids]
             for k, v in by_type.items()}

    placements = list(placements_keep)
    while spaces:
        best = None
        for s in spaces:
            ck = (ks.W, ks.D, ks.H, s)
            cached = _dist_cache.get(ck)
            if cached is None:
                cached = _near_corner(s, ks)
                _dist_cache[ck] = cached
            dist, sig = cached
            vol = (s[3]-s[0]) * (s[4]-s[1]) * (s[5]-s[2])
            key = (dist, -vol)
            if best is None or key < best[0]:
                best = (key, s, sig)
        _key, s, sig = best
        sx, sy, sz, sx2, sy2, sz2 = s
        fw, fd, fh = sx2 - sx, sy2 - sy, sz2 - sz
        cands = _enumerate_configs(s, sig, fw, fd, fh, avail,
                                   allow_rotation, ks.is_3d, objective)
        if not cands:
            spaces = [sp for sp in spaces if sp != s]
            continue
        cands.sort(key=lambda c: c[0])
        _score, tkey, w, d, h, nx, ny, nz, ncopies = cands[0]
        members = avail[tkey]
        pls, box = _place_block(s, sig, w, d, h, nx, ny, nz,
                                members, ncopies)
        placements.extend(pls)
        avail[tkey] = members[ncopies:]
        spaces = _apply_box(spaces, box)
    return placements


def _improve(placements, blocks, items, ks, allow_rotation, remove_frac):
    """Improvement phase: remove the final `remove_frac` of the blocks
    and refill deterministically, once per objective; return the best of
    the two refills (by used volume)."""
    n_keep = int(len(blocks) * (1.0 - remove_frac))
    keep_boxes = set(id(blocks[i][0]) for i in range(n_keep))
    # which placements belong to kept blocks: rebuild by walking blocks
    keep = []
    idx = 0
    for bi, (box, count) in enumerate(blocks):
        chunk = placements[idx:idx + count]
        if bi < n_keep:
            keep.extend(chunk)
        idx += count

    best_pk = None
    for objective in ("bestvol", "bestfit"):
        full = _rebuild_from_partial(keep, items, ks, allow_rotation,
                                     objective)
        pk = Packing(ks, full)
        if best_pk is None or pk.used_volume > best_pk.used_volume:
            best_pk = pk
    # the improvement must never worsen the incoming solution: if the
    # destroy-rebuild produced less, keep the original.
    orig = Packing(ks, list(placements))
    if orig.used_volume > best_pk.used_volume:
        best_pk = orig
    return best_pk


def solve_grasp_parreno(items: List[Item], ks: Knapsack,
                        allow_rotation: bool = True,
                        max_iter: int = 5000,
                        time_limit: Optional[float] = None,
                        seed: int = 1,
                        delta_set: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4,
                                                        0.5, 0.6, 0.7, 0.8,
                                                        0.9),
                        react_period: int = 500,
                        alpha: float = 10.0,
                        remove_frac: float = 0.5,
                        improve_threshold: float = 0.5):
    """Faithful GRASP-Parreno. Stops after `max_iter` iterations (one
    construction + one improvement each), or when `time_limit` seconds
    elapse if given. Returns GRASPParrenoResult."""
    rng = random.Random(seed)
    D = list(delta_set)
    n_d = len(D)
    prob = [1.0 / n_d] * n_d
    sum_d = [0.0] * n_d
    cnt_d = [0] * n_d

    start = time.time()
    v_best = -1.0
    v_worst = float("inf")
    best_pk = None
    history = []

    # alternate the construction objective per iteration as the paper
    # uses volume as the main objective; keep it volume-based for
    # construction (best-fit is significantly worse per Table 1/Fig 9).
    construct_obj = "bestvol"

    it = 0
    while it < max_iter:
        if time_limit is not None and time.time() - start >= time_limit:
            break
        # choose delta by current probabilities
        di = rng.choices(range(n_d), weights=prob, k=1)[0]
        delta = D[di]

        placements, blocks = _construct(items, ks, allow_rotation,
                                        construct_obj, delta, rng)
        pk = Packing(ks, placements)
        v = pk.used_volume

        # improvement only if the solution clears the threshold
        if blocks and v_best > 0 and v_worst < float("inf"):
            thresh = v_worst + improve_threshold * (v_best - v_worst)
        else:
            thresh = -1.0
        if v >= thresh and blocks:
            imp = _improve(placements, blocks, items, ks, allow_rotation,
                           remove_frac)
            if imp.used_volume > v:
                pk = imp
                v = imp.used_volume

        # bookkeeping for reactive GRASP
        if v > v_best:
            v_best = v
            best_pk = pk
        if v < v_worst:
            v_worst = v
        sum_d[di] += v
        cnt_d[di] += 1
        history.append(100.0 * v / ks.volume)

        it += 1
        # reactive update every react_period iterations
        if it % react_period == 0 and v_best > v_worst:
            evals = []
            for j in range(n_d):
                if cnt_d[j] == 0:
                    evals.append(1.0)
                else:
                    mean_j = sum_d[j] / cnt_d[j]
                    base = (mean_j - v_worst) / (v_best - v_worst)
                    evals.append(max(base, 1e-6) ** alpha)
            tot = sum(evals)
            prob = [e / tot for e in evals]

    elapsed = time.time() - start
    fill = 100.0 * best_pk.used_volume / ks.volume if best_pk else 0.0
    return GRASPParrenoResult(
        best_packing=best_pk, best_fill=fill, iterations=it,
        elapsed=elapsed, history=history,
        delta_final=D[max(range(n_d), key=lambda j: prob[j])])
