"""EP evaluation criteria (merit functions) of Section 3.1.

Four criteria are implemented; lower merit values are better.

FF  - First Fit: the first compatible EP is taken.
MP  - Minimize the maximum packing size on the X and Y axes.
LEV - Level the packing on the X and Y axes (penalised increase,
      otherwise minimise the distance from the envelope side).
RS  - Maximize the utilization of the EPs' Residual Space:
      f = (RSx - w) + (RSy - d) + (RSz - h), minimised.
"""

from __future__ import annotations

from typing import Tuple

from .extreme_points import ExtremePoint
from .geometry import Packing

MERIT_SEQUENCE = ("RS", "TP", "MP", "LEV", "FF")

# LEX (lexicographic corner distance) is implemented in merit_value but
# kept OUT of the default sequence. Experimental finding: at parity of
# ordering LEX beats RS by a few points on BR (the corner-distance rule
# of the Parreno constructive transfers to EP selection), but TP
# (touching perimeter) is stronger than both there. Inside GASP with the
# parreno_seed + layout-search pipeline the EP merit criterion has become
# irrelevant -- removing TP or adding LEX leaves the result unchanged --
# because the seed supplies the structure and the local search refines
# it, leaving the mid-loop greedy non-decisive. LEX is therefore offered
# as an opt-in criterion (useful for the pure single-pass constructive,
# or for GASP runs WITHOUT the Parreno seed), not added to the portfolio.


def merit_value(criterion: str, ep: ExtremePoint, w: int, d: int, h: int,
                packing: Packing, order: int) -> Tuple:
    """Return a comparable merit tuple (lower is better)."""
    if criterion == "FF":
        # The first compatible EP wins: merit is the discovery order
        return (order,)

    if criterion == "MP":
        W_mp, D_mp, _ = packing.envelope
        fx = (ep.x + w - W_mp) if ep.x + w > W_mp else 0
        fy = (ep.y + d - D_mp) if ep.y + d > D_mp else 0
        return (fx + fy, ep.z, ep.y, ep.x)

    if criterion == "LEV":
        W_mp, D_mp, _ = packing.envelope
        C = max(packing.knapsack.W, packing.knapsack.D) + 1
        fx = (ep.x + w - W_mp) * C if ep.x + w > W_mp else (W_mp - (ep.x + w))
        fy = (ep.y + d - D_mp) * C if ep.y + d > D_mp else (D_mp - (ep.y + d))
        return (fx + fy, ep.z, ep.y, ep.x)

    if criterion == "TP":
        # Touching Perimeter (Hadjiconstantinou & Iori 2007), generalised
        # to contact area in 3D: maximise the surface shared with the
        # knapsack walls and the items already placed.
        contact = _contact_area(ep.x, ep.y, ep.z, w, d, h, packing)
        return (-contact, ep.z, ep.y, ep.x)

    if criterion == "RS":
        f = (ep.rs_x - w) + (ep.rs_y - d) + (ep.rs_z - h)
        return (f, ep.z, ep.y, ep.x)

    if criterion == "LEX":
        # Lexicographic corner distance (Parreno et al. space-selection
        # rule, applied to the extreme point): prefer the EP closest to a
        # container corner, where "closeness" is the per-axis distance to
        # the nearest container face, sorted non-decreasing and compared
        # lexicographically. This fills corners before edges before
        # faces before the interior. The sorted triple makes the rule
        # isotropic (which axis touches a wall does not matter, only that
        # one does). Tie-break by tightest residual space, then origin.
        ks = packing.knapsack
        dx = ep.x if ep.x < (ks.W - (ep.x + w)) else (ks.W - (ep.x + w))
        dy = ep.y if ep.y < (ks.D - (ep.y + d)) else (ks.D - (ep.y + d))
        dz = ep.z if ep.z < (ks.H - (ep.z + h)) else (ks.H - (ep.z + h))
        a, b, c = dx, dy, dz
        if b < a: a, b = b, a
        if c < b: b, c = c, b
        if b < a: a, b = b, a
        rs = (ep.rs_x - w) + (ep.rs_y - d) + (ep.rs_z - h)
        return (a, b, c, rs, ep.z, ep.y, ep.x)

    raise ValueError(f"Unknown merit criterion: {criterion}")


def _overlap_1d(a1, a2, b1, b2) -> int:
    lo = a1 if a1 > b1 else b1
    hi = a2 if a2 < b2 else b2
    return hi - lo if hi > lo else 0


def _contact_area(x, y, z, w, d, h, packing) -> int:
    ks = packing.knapsack
    c = 0
    if x == 0:
        c += d * h
    if x + w == ks.W:
        c += d * h
    if y == 0:
        c += w * h
    if y + d == ks.D:
        c += w * h
    if z == 0:
        c += w * d
    if z + h == ks.H:
        c += w * d
    for p in packing.placements:
        if x + w == p.x or p.x2 == x:
            c += _overlap_1d(y, y + d, p.y, p.y2) * _overlap_1d(z, z + h, p.z, p.z2)
        if y + d == p.y or p.y2 == y:
            c += _overlap_1d(x, x + w, p.x, p.x2) * _overlap_1d(z, z + h, p.z, p.z2)
        if z + h == p.z or p.z2 == z:
            c += _overlap_1d(x, x + w, p.x, p.x2) * _overlap_1d(y, y + d, p.y, p.y2)
    return c
