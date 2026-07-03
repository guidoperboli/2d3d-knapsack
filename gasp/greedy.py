"""EP-KPH: Extreme Point-based Knapsack Problem Heuristic (Section 3.1).

Given an ordered list of items, the heuristic tries to accommodate them
one by one. For each item, every feasible rotation is evaluated on
every compatible EP through the chosen merit function; the best
(rotation, EP) couple is selected, otherwise the item is discarded.

The greedy is implemented as a resumable ``GreedyState`` so that the
adaptive variant of GASP can snapshot the packing after a prefix of the
ordering and warm-start from it when a perturbation only touches the
tail of the sequence (the state after k items depends only on the first
k items of the order, the merit criterion, and the rotation flag).
"""

from __future__ import annotations

from copy import copy
from typing import List, Optional

from .extreme_points import EPManager, ExtremePoint
from .geometry import Item, Knapsack, Packing, Placement
from .merit import merit_value


class GreedyState:
    """Incremental EP-KPH state: feed items one at a time."""

    def __init__(self, knapsack: Knapsack, criterion: str = "RS",
                 allow_rotation: bool = False):
        self.knapsack = knapsack
        self.criterion = criterion
        self.allow_rotation = allow_rotation
        self.packing = Packing(knapsack)
        self.epm = EPManager(knapsack)
        self.n_processed = 0

    # ------------------------------------------------------------------
    def clone(self) -> "GreedyState":
        """Cheap snapshot: Placements are immutable (shared), EPs copied."""
        st = GreedyState.__new__(GreedyState)
        st.knapsack = self.knapsack
        st.criterion = self.criterion
        st.allow_rotation = self.allow_rotation
        st.packing = Packing(self.knapsack, list(self.packing.placements))
        st.epm = EPManager.__new__(EPManager)
        st.epm.knapsack = self.knapsack
        st.epm.eps = [copy(ep) for ep in self.epm.eps]
        st.n_processed = self.n_processed
        return st

    # ------------------------------------------------------------------
    def place(self, item: Item) -> bool:
        """Try to accommodate one item; returns True if loaded."""
        best: Optional[Placement] = None
        best_merit = None
        is_3d = self.knapsack.is_3d

        for (w, d, h) in item.rotations(self.allow_rotation, is_3d):
            for order, ep in enumerate(self.epm.eps):
                cand = Placement(item, ep.x, ep.y, ep.z, w, d, h)
                if not self.packing.feasible(cand):
                    continue
                m = merit_value(self.criterion, ep, w, d, h,
                                self.packing, order)
                if best_merit is None or m < best_merit:
                    best_merit = m
                    best = cand
                if self.criterion == "FF":
                    break  # first compatible EP for this rotation
            if self.criterion == "FF" and best is not None:
                break

        self.n_processed += 1
        if best is not None:
            self.packing.placements.append(best)
            self.epm.add_item(best, self.packing)
            return True
        return False

    # ------------------------------------------------------------------
    def run(self, items: List[Item]) -> Packing:
        for item in items:
            self.place(item)
        return self.packing

    @property
    def residual_eps(self) -> List[ExtremePoint]:
        """EPs of the current packing (used by spatial score policies)."""
        return self.epm.eps


def ep_kph(items_in_order: List[Item], knapsack: Knapsack,
           criterion: str = "RS", allow_rotation: bool = False,
           return_eps: bool = False):
    """Run the EP-KPH greedy on `items_in_order` and return the packing.

    With ``return_eps=True`` the (packing, EPManager) pair is returned,
    so callers can inspect the residual free spaces of the final
    packing (used by the waste-driven score policy)."""
    state = make_greedy_state(knapsack, criterion, allow_rotation)
    packing = state.run(items_in_order)
    if return_eps:
        class _EPView:
            pass
        view = _EPView()
        view.eps = state.residual_eps
        return packing, view
    return packing


# ----------------------------------------------------------------------
# Backend selection: numba-accelerated state when available
# ----------------------------------------------------------------------
import os

_FAST = None
if os.environ.get("GASP_NO_NUMBA") != "1":
    try:
        from .fast_greedy import FastGreedyState as _FAST  # noqa: F401
        from .fast_greedy import warmup as _warmup
        _warmup()
    except Exception:
        _FAST = None


def make_greedy_state(knapsack: Knapsack, criterion: str = "RS",
                      allow_rotation: bool = False):
    """Factory: numba backend if importable, pure Python otherwise.

    LEX is only implemented in the Python merit, so it always uses the
    Python backend regardless of numba availability."""
    if criterion == "LEX":
        return GreedyState(knapsack, criterion, allow_rotation)
    cls = _FAST if _FAST is not None else GreedyState
    return cls(knapsack, criterion, allow_rotation)
