"""Layer-building constructive front-stage (VNS-inspired).

EXPERIMENTAL STATUS (weak / reuse version): falsified as an improvement,
kept behind a flag (GASPParams.layer_greedy, default off) at zero cost.
On heterogeneous BR instances the layer construction, even with mixed
secondary types filling the border gaps, produces a worse initial
solution than the best of the eight existing sorting rules already tried
by GASP's initial_solution (e.g. on thpack7-001: layer 69% vs best
classical sort 79% of base filling). An earlier positive reading (+0.65
points) came from comparing the layer against a SINGLE density sort
rather than against GASP's full sorting portfolio. The genuine VNS gain
requires the STRONG version -- exact 2D-knapsack per layer, optimal
slab thickness, stacking with backtracking and vertical stability --
which is a different algorithm, not a front-stage reusing the EP greedy.
The weak version is retained as a documented negative result and a
scaffold for the strong one.

Original design rationale follows.

Container Loading on heterogeneous instances (many box types) is where
the score-ordering paradigm loses ground to layer-based methods such as
the VNS of Parreno et al.: those split a 20-type problem into stacked
*layers*, each a near-homogeneous 2D sub-problem that tiles almost
perfectly, and stacking full layers is trivial. This module adds that
idea as a FRONT-STAGE before the existing EP greedy, without replacing
it.

Design (weak / reuse version)
-----------------------------
1. Type guard. Count distinct box types; below a threshold the layer
   idea brings nothing (few-type instances are already handled well and
   the bottleneck there is selection, not packing), so we delegate
   entirely to the classical EP greedy. This is the "se non ci sono i
   tipi usa la vecchia" rule.
2. Layer type selection. Among the available types, pick the one whose
   layer (a slab of height equal to one of its dimensions) tiles the
   base best, scored by axial fit times profit density.
3. Layer fill. Fill a height-h slab by calling the EP greedy RESTRICTED
   to that type (optionally a second type that fits in the same height),
   then translate the placements up by the current stack height.
4. Stack and repeat over the residual height; a final pass of the
   classical EP greedy fills the gaps with the remaining items.

Everything reuses the existing geometry, EP machinery and merit
functions; the layer stage only decides WHICH items, in WHICH thin
sub-knapsack, in WHICH order.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .geometry import Item, Knapsack, Packing, Placement
from .greedy import ep_kph


def _type_key(it: Item) -> Tuple[int, int, int]:
    return (it.w, it.d, it.h)


def _heights(it: Item, ks: Knapsack, allow_rotation: bool) -> List[int]:
    """Distinct feasible heights this item can present (its stacking
    dimension), one per usable rotation."""
    hs = {r[2] for r in it.rotations(allow_rotation, ks.is_3d)
          if r[0] <= ks.W and r[1] <= ks.D and r[2] <= ks.H}
    return sorted(hs)


def _axial_fit_layer(it: Item, h: int, ks: Knapsack,
                     allow_rotation: bool) -> int:
    """How many copies of `it` tile the W x D base in a slab of height
    h, over rotations that present exactly height h."""
    best = 0
    for (w, d, rh) in it.rotations(allow_rotation, ks.is_3d):
        if rh != h or w > ks.W or d > ks.D:
            continue
        best = max(best, (ks.W // w) * (ks.D // d))
    return best


class LayerGreedy:
    """Layer front-stage with a type guard, delegating to ep_kph."""

    def __init__(self, knapsack: Knapsack, criterion: str = "RS",
                 allow_rotation: bool = True,
                 type_threshold: int = 6,
                 min_layer_fill: float = 0.55,
                 max_second_types: int = 1):
        self.ks = knapsack
        self.criterion = criterion
        self.allow_rotation = allow_rotation
        self.type_threshold = type_threshold
        self.min_layer_fill = min_layer_fill
        self.max_second_types = max_second_types

    # ------------------------------------------------------------------
    def run(self, items: List[Item]) -> Packing:
        types = defaultdict(list)
        for it in items:
            types[_type_key(it)].append(it)

        # ---- type guard: few types -> classical EP greedy
        if not self.ks.is_3d or len(types) <= self.type_threshold:
            return ep_kph(self._sorted(items), self.ks, self.criterion,
                          self.allow_rotation)

        # ---- layer construction along the H axis
        placements: List[Placement] = []
        avail = {k: list(v) for k, v in types.items()}
        z_base = 0
        H = self.ks.H

        while z_base < H:
            rem_h = H - z_base
            choice = self._pick_layer_type(avail, rem_h)
            if choice is None:
                break
            key, h = choice
            slab = Knapsack(self.ks.W, self.ks.D, h)

            # items for this layer: the chosen type, plus optionally a
            # second compatible type sharing the same height
            layer_items = list(avail[key])
            second = self._second_type(avail, key, h)
            for k2 in second:
                layer_items += avail[k2]

            packing = ep_kph(self._sorted(layer_items), slab,
                             self.criterion, self.allow_rotation)
            fill = packing.used_volume / slab.volume if slab.volume else 0.0
            if not packing.placements or fill < self.min_layer_fill:
                # layer too sparse: stop the layer stage, let the final
                # EP pass handle the rest in the residual box
                break

            used_ids = packing.loaded_ids
            for p in packing.placements:
                placements.append(Placement(p.item, p.x, p.y, p.z + z_base,
                                            p.w, p.d, p.h))
            # consume used items
            for k in [key, *second]:
                avail[k] = [it for it in avail[k] if it.idx not in used_ids]
            avail = {k: v for k, v in avail.items() if v}
            z_base += h
            if not avail:
                break

        # ---- final EP pass: fill the residual top box with whatever
        # items remain, on top of the stacked layers
        remaining = [it for v in avail.values() for it in v]
        if remaining and z_base < H:
            placements = self._fill_residual(placements, remaining, z_base)

        return Packing(self.ks, placements)

    # ------------------------------------------------------------------
    def _sorted(self, items: List[Item]) -> List[Item]:
        # density-first, a sane default order for the EP greedy
        meas = (lambda i: i.volume) if self.ks.is_3d else (lambda i: i.base_area)
        return sorted(items, key=lambda i: -i.profit / meas(i))

    def _pick_layer_type(self, avail: Dict, rem_h: int):
        """Choose (type_key, height) maximising how well the layer tiles
        the base, among types that fit in the remaining height. We
        reward base COVERAGE (fraction of W x D the placed copies cover)
        times profit density, and prefer thinner layers at equal
        coverage: a thin, well-tiled slab is what makes stacking pay."""
        base = self.ks.W * self.ks.D
        best, best_score = None, 0.0
        for key, members in avail.items():
            if not members:
                continue
            rep = members[0]
            for (w, d, h) in rep.rotations(self.allow_rotation, self.ks.is_3d):
                if h > rem_h or w > self.ks.W or d > self.ks.D:
                    continue
                fit = (self.ks.W // w) * (self.ks.D // d)
                if fit == 0:
                    continue
                copies = min(len(members), fit)
                coverage = copies * w * d / base          # in [0, 1]
                if coverage < 0.35:
                    continue                              # too sparse a layer
                dens = rep.profit / rep.volume
                # thinner layers preferred at equal coverage/density
                score = coverage * dens / h
                if score > best_score:
                    best_score, best = score, (key, h)
        return best

    def _second_type(self, avail: Dict, key, h: int) -> List:
        """Optionally pick compatible secondary type(s) presenting the
        same layer height, to fill the base gaps the primary leaves."""
        if self.max_second_types <= 0:
            return []
        out = []
        for k2, members in avail.items():
            if k2 == key or not members:
                continue
            rep = members[0]
            if h in _heights(rep, self.ks, self.allow_rotation):
                out.append((rep.profit / rep.volume, k2))
        out.sort(reverse=True)
        return [k2 for _, k2 in out[:self.max_second_types]]

    def _fill_residual(self, placements: List[Placement],
                       remaining: List[Item], z_base: int) -> List[Placement]:
        """Fill the residual top box [z_base, H) with the leftover items
        using the classical EP greedy, then lift the result."""
        top = Knapsack(self.ks.W, self.ks.D, self.ks.H - z_base)
        packing = ep_kph(self._sorted(remaining), top, self.criterion,
                         self.allow_rotation)
        for p in packing.placements:
            placements.append(Placement(p.item, p.x, p.y, p.z + z_base,
                                        p.w, p.d, p.h))
        return placements
