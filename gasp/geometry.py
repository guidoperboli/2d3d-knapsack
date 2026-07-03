"""Core geometric entities for the Multi-Dimensional Knapsack problems.

The 2D case is handled as a degenerate 3D case with height = 1 for all
items and for the knapsack, so the whole code base works uniformly in
2D and 3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Tuple


# Precomputed once: the 6 axis permutations used to enumerate 3D
# orientations, and a process-wide cache of rotation results keyed by
# (w, d, h, vflags, allow_rotation, is_3d). rotations() is called
# hundreds of thousands of times per run on a small set of distinct
# items, so memoising removes a large amount of redundant work.
_AXIS_PERMS = tuple(permutations((0, 1, 2)))
_ROT_CACHE: dict = {}


@dataclass(frozen=True)
class Item:
    """A rectangular (boxed) item.

    Attributes
    ----------
    idx     : unique index of the item in the instance
    w, d, h : width (x), depth (y), height (z)
    profit  : profit gained when the item is loaded
    vflags  : optional (vw, vd, vh) booleans, one per dimension, telling
              whether that dimension may be placed vertically (along the
              container height). Used by the BR container-loading sets,
              where orientation is constrained. None means free rotation.
    """

    idx: int
    w: int
    d: int
    h: int
    profit: float
    vflags: Tuple[bool, bool, bool] = None

    @property
    def volume(self) -> int:
        return self.w * self.d * self.h

    @property
    def base_area(self) -> int:
        return self.w * self.d

    def rotations(self, allow_rotation: bool, is_3d: bool) -> List[Tuple[int, int, int]]:
        """Return the list of distinct (w, d, h) orientations.

        When vflags is set (BR orientation constraints), only rotations
        whose vertical dimension comes from an original dimension allowed
        to be vertical are kept. The filter is axis-based: it tracks
        which original dimension (w, d or h) ends up vertical, so it is
        correct even when two dimensions share a value.

        Memoised: the result depends only on the (immutable) dimensions,
        vflags and the two flags, so it is computed once per distinct
        key and cached. This is called hundreds of thousands of times in
        a run, so the cache is a large speed-up."""
        key = (self.w, self.d, self.h, self.vflags, allow_rotation, is_3d)
        cached = _ROT_CACHE.get(key)
        if cached is not None:
            return cached
        dims = (self.w, self.d, self.h)
        if not allow_rotation:
            rots = [dims]
        elif is_3d:
            seen = set()
            rots = []
            for perm in _AXIS_PERMS:
                cand = (dims[perm[0]], dims[perm[1]], dims[perm[2]])
                if self.vflags is not None and not self.vflags[perm[2]]:
                    continue
                if cand not in seen:
                    seen.add(cand)
                    rots.append(cand)
            rots = sorted(rots) if rots else [dims]
        else:
            rots = sorted({dims, (self.d, self.w, self.h)})
        _ROT_CACHE[key] = rots
        return rots


@dataclass(frozen=True)
class Placement:
    """An item accommodated in the knapsack at position (x, y, z) with a
    specific orientation (w, d, h)."""

    item: Item
    x: int
    y: int
    z: int
    w: int
    d: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.d

    @property
    def z2(self) -> int:
        return self.z + self.h

    def overlaps(self, other: "Placement") -> bool:
        return not (
            self.x2 <= other.x or other.x2 <= self.x
            or self.y2 <= other.y or other.y2 <= self.y
            or self.z2 <= other.z or other.z2 <= self.z
        )


@dataclass
class Knapsack:
    """The single large object (knapsack / container)."""

    W: int
    D: int
    H: int = 1  # H = 1 -> 2D problem

    @property
    def is_3d(self) -> bool:
        return self.H > 1

    @property
    def volume(self) -> int:
        return self.W * self.D * self.H

    def fits(self, x: int, y: int, z: int, w: int, d: int, h: int) -> bool:
        return x + w <= self.W and y + d <= self.D and z + h <= self.H


@dataclass
class Packing:
    """A (partial) solution: the set of placements inside the knapsack."""

    knapsack: Knapsack
    placements: List[Placement] = field(default_factory=list)

    @property
    def profit(self) -> float:
        return sum(p.item.profit for p in self.placements)

    @property
    def used_volume(self) -> int:
        return sum(p.w * p.d * p.h for p in self.placements)

    @property
    def loaded_ids(self):
        return {p.item.idx for p in self.placements}

    # Minimum box envelope of the current packing (used by MP / LEV)
    @property
    def envelope(self) -> Tuple[int, int, int]:
        if not self.placements:
            return 0, 0, 0
        return (
            max(p.x2 for p in self.placements),
            max(p.y2 for p in self.placements),
            max(p.z2 for p in self.placements),
        )

    def feasible(self, cand: Placement) -> bool:
        if not self.knapsack.fits(cand.x, cand.y, cand.z, cand.w, cand.d, cand.h):
            return False
        return all(not cand.overlaps(p) for p in self.placements)
