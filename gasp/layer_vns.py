"""Exact layer solver for Container Loading (documented dead end).

EXPERIMENTAL STATUS: this module implements pure exact layers (W x H
slabs stacked along D, each face filled by an exact 2D knapsack with
depth backtracking). On strongly heterogeneous instances (BR7) it
reaches only ~52% filling even with a generous budget, against ~86% for
GASP -- because dense layers need high box MULTIPLICITY, while the BR7
class has high VARIETY and low multiplicity.

Crucially, a literature check (Parreno, Alvarez-Valdes, Oliveira &
Tamarit, J. Heuristics 2010) shows that the state-of-the-art VNS does
NOT use layers: it combines a constructive procedure based on EMPTY
MAXIMAL SPACES with five physical-layout movements. Pure horizontal
layers are the older, weaker Bischoff-Ratcliff (1995) greedy paradigm.
So the productive direction for closing the heterogeneity gap is
maximal-space placement, not layers. This module is retained as a
documented negative result and a precise measurement of the layer
ceiling; it is not wired into GASP by default.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .geometry import Item, Knapsack, Packing, Placement


def _thickness_options(items: List[Item], ks: Knapsack,
                       allow_rotation: bool) -> List[int]:
    """Distinct feasible layer thicknesses (a box dimension that can lie
    along the depth axis D), most common first."""
    counts: Dict[int, int] = defaultdict(int)
    for it in items:
        for (w, d, h) in it.rotations(allow_rotation, ks.is_3d):
            # box presents (w,h) on the W x H face, thickness d along D
            if w <= ks.W and h <= ks.H and d <= ks.D:
                counts[d] += 1
    return [d for d, _ in sorted(counts.items(),
                                 key=lambda kv: (-kv[1], kv[0]))]


def _face_items(items: List[Item], ks: Knapsack, thickness: int,
                allow_rotation: bool) -> List[Item]:
    """Items that can present the given thickness along D (i.e. have a
    rotation with d == thickness fitting the W x H face)."""
    out = []
    for it in items:
        for (w, d, h) in it.rotations(allow_rotation, ks.is_3d):
            if d == thickness and w <= ks.W and h <= ks.H:
                out.append(it)
                break
    return out


def _solve_face(face_items: List[Item], ks: Knapsack, thickness: int,
                allow_rotation: bool, time_limit: float,
                max_model_items: int = 30):
    """Exact 2D knapsack on the W x H face for a slab of given
    thickness: maximise the placed face area (=> volume, since depth is
    fixed). To keep the CP model small and fast we feed it a capped pool
    of copies of the densest thickness-compatible types (a layer is
    dominated by few types anyway). Returns a list of
    (item, x_W, z_H, w, h) placements, or None."""
    from .cp_slave import cp_solve_kp

    face_ks = Knapsack(ks.W, ks.H, 1)
    proj: List[Item] = []
    back: Dict[int, Item] = {}
    # group thickness-compatible items by their face (w, h); keep the
    # densest types first, capping the total model size
    by_face: Dict[Tuple[int, int], List[Item]] = defaultdict(list)
    for it in face_items:
        faces = [(w, h) for (w, d, h) in it.rotations(allow_rotation, ks.is_3d)
                 if d == thickness and w <= ks.W and h <= ks.H]
        if faces:
            by_face[faces[0]].append(it)
    # order faces by area-density of profit (here profit == volume proxy)
    face_keys = sorted(by_face, key=lambda f: -(f[0] * f[1]))
    for (w, h) in face_keys:
        for it in by_face[(w, h)]:
            fi = Item(it.idx, w, h, 1, profit=w * h)
            proj.append(fi)
            back[it.idx] = it
            if len(proj) >= max_model_items:
                break
        if len(proj) >= max_model_items:
            break

    if not proj:
        return None
    # face items are already projected to a single orientation: solve
    # without rotation to keep the model lean
    profit, _bound, placements, _opt = cp_solve_kp(
        proj, face_ks, allow_rotation=False, time_limit=time_limit)
    if not placements:
        return None
    out = []
    for p in placements:
        out.append((back[p.item.idx], p.x, p.y, p.w, p.d))  # x on W, y on H
    return out


def solve_layers(items: List[Item], ks: Knapsack,
                 allow_rotation: bool = True,
                 max_thickness_tries: int = 4,
                 time_per_face: float = 1.0,
                 total_budget: float = 30.0,
                 max_depth: int = 40) -> Packing:
    """Exact-layer solver. Stacks W x H layers along D with bounded
    recursion over thicknesses."""
    if not ks.is_3d:
        # 2D: a single layer is the whole problem
        face = _solve_face(items, ks, 1, allow_rotation, total_budget)
        placements = []
        if face:
            for it, xw, zh, w, h in face:
                placements.append(Placement(it, xw, 0, zh, w, 1, h))
        return Packing(ks, placements)

    t0 = time.time()
    best_placements: List[Placement] = []
    best_vol = 0

    def recurse(d_base: int, avail_ids: set, acc: List[Placement],
                acc_vol: int, depth: int) -> None:
        nonlocal best_placements, best_vol
        if acc_vol > best_vol:
            best_vol = acc_vol
            best_placements = list(acc)
        if (time.time() - t0 > total_budget or depth >= max_depth
                or d_base >= ks.D):
            return

        rem_items = [it for it in items if it.idx in avail_ids]
        if not rem_items:
            return
        rem_depth = ks.D - d_base
        thicks = [t for t in _thickness_options(rem_items, ks, allow_rotation)
                  if t <= rem_depth][:max_thickness_tries]

        for thk in thicks:
            if time.time() - t0 > total_budget:
                break
            fitems = _face_items(rem_items, ks, thk, allow_rotation)
            if not fitems:
                continue
            face = _solve_face(fitems, ks, thk, allow_rotation,
                               time_per_face)
            if not face:
                continue
            # place the layer at depth [d_base, d_base+thk)
            layer_pl = []
            used = set()
            lvol = 0
            for it, xw, zh, w, h in face:
                if it.idx in used:
                    continue  # one copy of an item per layer
                layer_pl.append(Placement(it, xw, d_base, zh, w, thk, h))
                used.add(it.idx)
                lvol += w * thk * h
            if not layer_pl:
                continue
            recurse(d_base + thk, avail_ids - used,
                    acc + layer_pl, acc_vol + lvol, depth + 1)

    recurse(0, {it.idx for it in items}, [], 0, 0)
    return Packing(ks, best_placements)
