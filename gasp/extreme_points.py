"""Extreme Points (EPs) and Residual Space (RS) management.

Implements the Extreme Point concept introduced by Crainic, Perboli and
Tadei (2008): when an item k with sizes (w, d, h) is placed at
(x, y, z), the candidate EPs for further items are the orthogonal
projections of the three points

    (x + w, y, z),  (x, y + d, z),  (x, y, z + h)

towards the axes, taking into account the items already accommodated.

Each EP carries a Residual Space (RS): along each axis, the distance
from the EP to the knapsack wall or to the nearest item blocking that
direction. The RS is initialised against the knapsack walls and updated
every time a new item is added to the packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .geometry import Knapsack, Packing, Placement

# Extended EP vocabulary (fixed-point settle continuations + diagonal
# edges). Tested on BR1/BR7 and cubes: no measurable gain over the
# composed projections (means within noise), at higher per-item cost.
# Kept behind a flag for reproducibility; default off.
EXTENDED_PROJECTIONS = False


@dataclass
class ExtremePoint:
    x: int
    y: int
    z: int
    rs_x: int
    rs_y: int
    rs_z: int

    @property
    def pos(self):
        return (self.x, self.y, self.z)


def _project_x(x: int, y: int, z: int, placements: List[Placement]) -> int:
    """Project point towards x = 0, stopping on the first item face."""
    best = 0
    for p in placements:
        if p.x2 <= x and p.y <= y < p.y2 and p.z <= z < p.z2:
            best = max(best, p.x2)
    return best


def _project_y(x: int, y: int, z: int, placements: List[Placement]) -> int:
    best = 0
    for p in placements:
        if p.y2 <= y and p.x <= x < p.x2 and p.z <= z < p.z2:
            best = max(best, p.y2)
    return best


def _project_z(x: int, y: int, z: int, placements: List[Placement]) -> int:
    best = 0
    for p in placements:
        if p.z2 <= z and p.x <= x < p.x2 and p.y <= y < p.y2:
            best = max(best, p.z2)
    return best


class EPManager:
    """Maintains the EP list and the residual spaces of a packing."""

    def __init__(self, knapsack: Knapsack):
        self.knapsack = knapsack
        # The origin is the first EP, with RS equal to the knapsack itself
        self.eps: List[ExtremePoint] = [
            ExtremePoint(0, 0, 0, knapsack.W, knapsack.D, knapsack.H)
        ]

    # ------------------------------------------------------------------
    def _residual_space(self, x: int, y: int, z: int,
                        placements: List[Placement]) -> ExtremePoint:
        """Compute the RS of a new EP at (x, y, z)."""
        rs_x = self.knapsack.W - x
        rs_y = self.knapsack.D - y
        rs_z = self.knapsack.H - z
        for p in placements:
            # item ahead on X, overlapping on Y and Z
            if p.x >= x and p.y < y + 1 <= p.y2 and p.z < z + 1 <= p.z2:
                rs_x = min(rs_x, p.x - x)
            if p.y >= y and p.x < x + 1 <= p.x2 and p.z < z + 1 <= p.z2:
                rs_y = min(rs_y, p.y - y)
            if p.z >= z and p.x < x + 1 <= p.x2 and p.y < y + 1 <= p.y2:
                rs_z = min(rs_z, p.z - z)
        return ExtremePoint(x, y, z, rs_x, rs_y, rs_z)

    # ------------------------------------------------------------------
    def add_item(self, placed: Placement, packing: Packing) -> None:
        """Update the EP list after `placed` has been added to `packing`.

        The placement is assumed to be already appended to
        ``packing.placements``.
        """
        others = [p for p in packing.placements if p is not placed]
        x, y, z = placed.x, placed.y, placed.z
        w, d, h = placed.w, placed.d, placed.h
        # Ordered candidate list (deterministic: same sequence as the
        # numba kernel in fast_greedy, so FF behaves identically)
        new_pts = [
            # Projections of (x + w, y, z) on Y and Z directions
            (x + w, _project_y(x + w, y, z, others), z),
            (x + w, y, _project_z(x + w, y, z, others)),
            # Projections of (x, y + d, z) on X and Z directions
            (_project_x(x, y + d, z, others), y + d, z),
            (x, y + d, _project_z(x, y + d, z, others)),
        ]
        if self.knapsack.is_3d:
            # Projections of (x, y, z + h) on X and Y directions
            new_pts.append((_project_x(x, y, z + h, others), y, z + h))
            new_pts.append((x, _project_y(x, y, z + h, others), z + h))
            # Composed projections: project the simple EP again along
            # the remaining orthogonal axis, reaching "inner corner"
            # positions that single projections cannot generate.
            py1 = new_pts[0][1]
            new_pts.append((x + w, py1, _project_z(x + w, py1, z, others)))
            pz1 = new_pts[1][2]
            new_pts.append((x + w, _project_y(x + w, y, pz1, others), pz1))
            px1 = new_pts[2][0]
            new_pts.append((px1, y + d, _project_z(px1, y + d, z, others)))
            pz2 = new_pts[3][2]
            new_pts.append((_project_x(x, y + d, pz2, others), y + d, pz2))
            px2 = new_pts[4][0]
            new_pts.append((px2, _project_y(px2, y, z + h, others), z + h))
            py2 = new_pts[5][1]
            new_pts.append((_project_x(x, py2, z + h, others), py2, z + h))

            if EXTENDED_PROJECTIONS:
                # Fixed-point continuations: keep alternating the two
                # orthogonal projections of each composed candidate until
                # convergence, emitting every intermediate point (the point
                # "settles" into inner notches; intermediates are kept
                # because some merit functions may prefer them). Capped at
                # 2 extra emissions per candidate.
                specs = ((6, "y", "z"), (7, "z", "y"), (8, "x", "z"),
                         (9, "z", "x"), (10, "x", "y"), (11, "y", "x"))
                for idx, a1, a2 in specs:
                    cx, cy, cz = new_pts[idx]
                    emitted = stall = t = 0
                    while emitted < 2 and stall < 2:
                        a = a1 if t % 2 == 0 else a2
                        t += 1
                        moved = False
                        if a == "x":
                            nv = _project_x(cx, cy, cz, others)
                            if nv < cx:
                                cx, moved = nv, True
                        elif a == "y":
                            nv = _project_y(cx, cy, cz, others)
                            if nv < cy:
                                cy, moved = nv, True
                        else:
                            nv = _project_z(cx, cy, cz, others)
                            if nv < cz:
                                cz, moved = nv, True
                        if moved:
                            new_pts.append((cx, cy, cz))
                            emitted += 1
                            stall = 0
                        else:
                            stall += 1

                # Diagonal-edge candidates: corners with two offsets,
                # projected along their single remaining orthogonal axis.
                new_pts.append((x + w, y + d, _project_z(x + w, y + d, z, others)))
                new_pts.append((x + w, _project_y(x + w, y, z + h, others), z + h))
                new_pts.append((_project_x(x, y + d, z + h, others), y + d, z + h))

        # Remove EPs covered by the new item, keep the others
        survivors = []
        for ep in self.eps:
            inside = (placed.x <= ep.x < placed.x2
                      and placed.y <= ep.y < placed.y2
                      and placed.z <= ep.z < placed.z2)
            if not inside:
                survivors.append(ep)
        self.eps = survivors

        existing = {ep.pos for ep in self.eps}
        for (px, py, pz) in new_pts:
            if (px, py, pz) in existing:
                continue
            if px >= self.knapsack.W or py >= self.knapsack.D or pz >= self.knapsack.H:
                continue
            self.eps.append(self._residual_space(px, py, pz, packing.placements))
            existing.add((px, py, pz))

        # Update the RS of all the EPs against the new item
        for ep in self.eps:
            if (placed.x >= ep.x and placed.y <= ep.y < placed.y2
                    and placed.z <= ep.z < placed.z2):
                ep.rs_x = min(ep.rs_x, placed.x - ep.x)
            if (placed.y >= ep.y and placed.x <= ep.x < placed.x2
                    and placed.z <= ep.z < placed.z2):
                ep.rs_y = min(ep.rs_y, placed.y - ep.y)
            if (placed.z >= ep.z and placed.x <= ep.x < placed.x2
                    and placed.y <= ep.y < placed.y2):
                ep.rs_z = min(ep.rs_z, placed.z - ep.z)
