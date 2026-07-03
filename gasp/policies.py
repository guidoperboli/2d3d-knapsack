"""Targeted score-update policies for GASP.

The original Score Update (Section 3.4 of the paper) perturbs exactly
one (loaded, unloaded) pair per iteration. The classes below generalise
it into a portfolio of policies, each identifying a different subset of
items and re-ranking only those:

PairSwapPolicy     the original rule (kept as baseline / escape policy)
BlockSwapPolicy    k worst-loaded vs k best-unloaded, k adaptive
FrontierPolicy     items around the loaded/unloaded cutoff of the
                   current ordering, where decision ambiguity lives
DensityBandPolicy  items within a density band (p/area or p/volume) of
                   the marginal loaded item - formalises the paper's
                   remark that mistakes happen among similar profits
WasteMatchPolicy   spatial rule: promotes unloaded items whose sizes
                   best match the Residual Spaces left unused by the
                   current packing

An ALNS-style roulette (PolicySelector) learns online which policy is
paying off on the instance at hand; weights are reset at every
Long-term Score Reinitialization, in the same spot where the original
GASP cycles the merit functions.

Every policy receives a PolicyContext and mutates ``ctx.scores`` only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .extreme_points import ExtremePoint
from .geometry import Item, Packing


@dataclass
class PolicyContext:
    items: Sequence[Item]
    scores: Dict[int, float]
    f_loaded: Dict[int, int]
    f_unloaded: Dict[int, int]
    current: Packing                      # solution of the last iteration
    order: List[Item]                     # ordering used by that iteration
    residual_eps: List[ExtremePoint]      # EPs left by that packing
    rng: random.Random
    alpha: float = 0.1
    beta: float = 0.1

    def split(self):
        loaded_ids = self.current.loaded_ids
        loaded = [i for i in self.items if i.idx in loaded_ids]
        unloaded = [i for i in self.items if i.idx not in loaded_ids]
        return loaded, unloaded


def _density(item: Item, is_3d: bool) -> float:
    meas = item.volume if is_3d else item.base_area
    return item.profit / meas


def _type_key(item: Item):
    # Items expanded from the same piece type are interchangeable:
    # swapping scores between two copies is a packing no-op.
    return (item.w, item.d, item.h, item.profit)


class Policy:
    name = "base"

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        """Mutate ctx.scores; return the position (in ctx.order) of the
        first item whose score changed, or None if no-op. The caller
        uses it as the warm-start boundary."""
        raise NotImplementedError

    # helper: earliest order position among a set of item ids
    @staticmethod
    def _first_pos(ctx: PolicyContext, ids) -> Optional[int]:
        pos = [k for k, it in enumerate(ctx.order) if it.idx in ids]
        return min(pos) if pos else None


# ----------------------------------------------------------------------
class PairSwapPolicy(Policy):
    """Original Section 3.4 rule: one loaded / one unloaded, score swap."""
    name = "pair"

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded, unloaded = ctx.split()
        if not loaded or not unloaded:
            return None
        is_3d = ctx.current.knapsack.is_3d
        j = min(loaded, key=lambda i: _density(i, is_3d)
                * (1 + ctx.f_loaded[i.idx]))
        diff = [i for i in unloaded if _type_key(i) != _type_key(j)]
        pool = diff if diff else unloaded
        l = max(pool, key=lambda i: _density(i, is_3d)
                / (1 + ctx.f_unloaded[i.idx]))
        ctx.scores[j.idx] *= (1 - ctx.alpha)
        ctx.scores[l.idx] *= (1 + ctx.beta)
        ctx.scores[j.idx], ctx.scores[l.idx] = \
            ctx.scores[l.idx], ctx.scores[j.idx]
        return self._first_pos(ctx, {j.idx, l.idx})


# ----------------------------------------------------------------------
class BlockSwapPolicy(Policy):
    """k worst-loaded vs k best-unloaded; k grows under stagnation."""
    name = "block"

    def __init__(self, k_max: int = 5):
        self.k = 2
        self.k_max = k_max

    def on_improvement(self):
        self.k = 2

    def on_stagnation(self):
        self.k = min(self.k + 1, self.k_max)

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded, unloaded = ctx.split()
        if not loaded or not unloaded:
            return None
        is_3d = ctx.current.knapsack.is_3d
        k = min(self.k, len(loaded), len(unloaded))
        worst = sorted(loaded, key=lambda i: _density(i, is_3d)
                       * (1 + ctx.f_loaded[i.idx]))[:k]
        best = sorted(unloaded, key=lambda i: -_density(i, is_3d)
                      / (1 + ctx.f_unloaded[i.idx]))[:k]
        touched = set()
        for j, l in zip(worst, best):
            if _type_key(j) == _type_key(l):
                continue  # copies of the same type: packing no-op
            ctx.scores[j.idx] *= (1 - ctx.alpha)
            ctx.scores[l.idx] *= (1 + ctx.beta)
            ctx.scores[j.idx], ctx.scores[l.idx] = \
                ctx.scores[l.idx], ctx.scores[j.idx]
            touched.update((j.idx, l.idx))
        return self._first_pos(ctx, touched)


# ----------------------------------------------------------------------
class FrontierPolicy(Policy):
    """Perturb only the decision frontier: the last items that entered
    the knapsack and the first that were rejected, in order position.
    Top-of-list and bottom-of-list decisions are left untouched."""
    name = "frontier"

    def __init__(self, width: int = 3):
        self.width = width

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded_ids = ctx.current.loaded_ids
        pos_loaded = [k for k, it in enumerate(ctx.order)
                      if it.idx in loaded_ids]
        pos_unloaded = [k for k, it in enumerate(ctx.order)
                        if it.idx not in loaded_ids]
        if not pos_loaded or not pos_unloaded:
            return None
        last_in = [ctx.order[k] for k in pos_loaded[-self.width:]]
        first_out = [ctx.order[k] for k in pos_unloaded[:self.width]]
        touched = set()
        for j, l in zip(reversed(last_in), first_out):
            if _type_key(j) == _type_key(l):
                continue  # copies of the same type: packing no-op
            ctx.scores[j.idx] *= (1 - ctx.alpha)
            ctx.scores[l.idx] *= (1 + ctx.beta)
            ctx.scores[j.idx], ctx.scores[l.idx] = \
                ctx.scores[l.idx], ctx.scores[j.idx]
            touched.update((j.idx, l.idx))
        return self._first_pos(ctx, touched)


# ----------------------------------------------------------------------
class DensityBandPolicy(Policy):
    """Re-rank the item *types* whose profit density lies within
    delta% of the marginal (least dense) loaded item.

    Type-aware: copies of the same piece are interchangeable, so the
    shuffle happens between type representatives (copies keep a common
    score and stay adjacent in the ordering) and the band is capped to
    the ``max_types`` types closest to the pivot density, preventing
    the policy from degenerating into a global shuffle on instances
    with many duplicated pieces (e.g. the okp set)."""
    name = "band"

    def __init__(self, delta: float = 0.15, max_types: int = 8):
        self.delta = delta
        self.max_types = max_types

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded, unloaded = ctx.split()
        if not loaded or not unloaded:
            return None
        is_3d = ctx.current.knapsack.is_3d
        pivot = min(_density(i, is_3d) for i in loaded)
        lo, hi = pivot * (1 - self.delta), pivot * (1 + self.delta)

        groups = {}
        for i in ctx.items:
            dns = _density(i, is_3d)
            if lo <= dns <= hi:
                groups.setdefault(_type_key(i), []).append(i)
        if len(groups) < 2:
            return None
        # keep only the types closest to the pivot density
        keys = sorted(groups,
                      key=lambda k: abs(_density(groups[k][0], is_3d)
                                        - pivot))[:self.max_types]
        if len(keys) < 2:
            return None
        # shuffle one representative score per type; copies share it
        type_scores = [max(ctx.scores[i.idx] for i in groups[k])
                       for k in keys]
        ctx.rng.shuffle(type_scores)
        touched = set()
        for k, s in zip(keys, type_scores):
            for i in groups[k]:
                ctx.scores[i.idx] = s
                touched.add(i.idx)
        return self._first_pos(ctx, touched)


# ----------------------------------------------------------------------
class WideBandPolicy(Policy):
    """Original item-level density band (no type collapsing, no cap):
    a stronger, more disruptive shuffle. Kept in the portfolio next to
    the type-aware DensityBandPolicy so the ALNS roulette can pick the
    right granularity per instance."""
    name = "band_wide"

    def __init__(self, delta: float = 0.15):
        self.delta = delta

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded, unloaded = ctx.split()
        if not loaded or not unloaded:
            return None
        is_3d = ctx.current.knapsack.is_3d
        pivot = min(_density(i, is_3d) for i in loaded)
        lo, hi = pivot * (1 - self.delta), pivot * (1 + self.delta)
        band = [i for i in ctx.items if lo <= _density(i, is_3d) <= hi]
        if len(band) < 2:
            return None
        band_scores = [ctx.scores[i.idx] for i in band]
        ctx.rng.shuffle(band_scores)
        for i, s in zip(band, band_scores):
            ctx.scores[i.idx] = s
        return self._first_pos(ctx, {i.idx for i in band})


# ----------------------------------------------------------------------
class GroupExchangePolicy(Policy):
    """Subset-exchange move for profit-mix plateaus: pick a small group
    S of low-density loaded items and a group T of high-density
    unloaded items whose total area roughly matches area(S) plus the
    current waste, then force the exchange by hard score bias (T above
    every score, S below every score). Pair/block swaps cannot express
    this move: freeing one slot at a time lets the greedy refill it
    with the same items."""
    name = "group"

    def __init__(self, max_out: int = 3, slack: float = 0.05):
        self.max_out = max_out
        self.slack = slack

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded, unloaded = ctx.split()
        if not loaded or not unloaded:
            return None
        is_3d = ctx.current.knapsack.is_3d
        meas = (lambda i: i.volume) if is_3d else (lambda i: i.base_area)

        n_out = ctx.rng.randint(1, self.max_out)
        out = sorted(loaded, key=lambda i: _density(i, is_3d)
                     * (1 + ctx.f_loaded[i.idx]))[:n_out]
        ks = ctx.current.knapsack
        waste = ks.volume - ctx.current.used_volume
        budget = sum(meas(i) for i in out) + waste
        budget *= (1 + self.slack)

        cand = sorted(unloaded, key=lambda i: -_density(i, is_3d)
                      / (1 + ctx.f_unloaded[i.idx]))
        out_types = {_type_key(i) for i in out}
        group, used = [], 0
        for i in cand:
            if _type_key(i) in out_types:
                continue
            if used + meas(i) <= budget:
                group.append(i)
                used += meas(i)
            if len(group) >= 2 * self.max_out:
                break
        if not group:
            return None

        hi = max(ctx.scores.values())
        lo = min(ctx.scores.values())
        for r, i in enumerate(group):
            ctx.scores[i.idx] = hi * (1.5 - 0.01 * r)   # force in, early
        for i in out:
            ctx.scores[i.idx] = lo * 0.5                # force out
        touched = {i.idx for i in group} | {i.idx for i in out}
        return self._first_pos(ctx, touched)


# ----------------------------------------------------------------------
class WasteMatchPolicy(Policy):
    """Spatial policy: rank the unloaded items by how well their sizes
    fit the Residual Spaces left empty by the current packing, and
    promote the best matches over the least dense loaded items."""
    name = "waste"

    def __init__(self, k: int = 3):
        self.k = k

    @staticmethod
    def _fit(item: Item, eps: List[ExtremePoint], is_3d: bool,
             allow_rotation: bool) -> float:
        best = float("inf")
        rots = item.rotations(allow_rotation, is_3d)
        for ep in eps:
            for (w, d, h) in rots:
                if w <= ep.rs_x and d <= ep.rs_y and h <= ep.rs_z:
                    slack = (ep.rs_x - w) + (ep.rs_y - d) + (ep.rs_z - h)
                    if slack < best:
                        best = slack
        return best  # inf -> fits nowhere

    def apply(self, ctx: PolicyContext) -> Optional[int]:
        loaded, unloaded = ctx.split()
        if not loaded or not unloaded or not ctx.residual_eps:
            return None
        is_3d = ctx.current.knapsack.is_3d
        fits = [(self._fit(i, ctx.residual_eps, is_3d, True), i)
                for i in unloaded]
        fits = [(f, i) for f, i in fits if f != float("inf")]
        if not fits:
            return None
        fits.sort(key=lambda t: t[0])
        promote = [i for _, i in fits[:self.k]]
        demote = sorted(loaded, key=lambda i: _density(i, is_3d))[:len(promote)]
        touched = set()
        for j, l in zip(demote, promote):
            ctx.scores[j.idx], ctx.scores[l.idx] = \
                ctx.scores[l.idx] * (1 + ctx.beta), \
                ctx.scores[j.idx] * (1 - ctx.alpha)
            touched.update((j.idx, l.idx))
        return self._first_pos(ctx, touched)


# ----------------------------------------------------------------------
@dataclass
class PolicySelector:
    """ALNS-style roulette wheel over the policy portfolio.

    Rewards: 3 for a new global best, 1 for improving on the previous
    iteration, 0 otherwise; weights follow an exponential smoothing
    w <- (1-rho) w + rho reward and are reset at every long-term
    reinitialization."""
    policies: List[Policy]
    rho: float = 0.2
    w_init: float = 1.0
    weights: Dict[str, float] = field(default_factory=dict)
    last_used: Optional[Policy] = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.weights = {p.name: self.w_init for p in self.policies}
        if not hasattr(self, "stats"):
            self.stats = {p.name: {"used": 0, "reward": 0.0, "best": 0}
                          for p in self.policies}

    def pick(self, rng: random.Random) -> Policy:
        total = sum(self.weights.values())
        r = rng.uniform(0, total)
        acc = 0.0
        for p in self.policies:
            acc += self.weights[p.name]
            if r <= acc:
                self.last_used = p
                self.stats[p.name]["used"] += 1
                return p
        self.last_used = self.policies[-1]
        return self.last_used

    def reward(self, value: float):
        if self.last_used is None:
            return
        name = self.last_used.name
        self.stats[name]["reward"] += value
        if value >= 3.0:
            self.stats[name]["best"] += 1
        self.weights[name] = max(
            0.05, (1 - self.rho) * self.weights[name] + self.rho * value)


def default_portfolio() -> List[Policy]:
    return [PairSwapPolicy(), BlockSwapPolicy(), FrontierPolicy(),
            DensityBandPolicy(), WideBandPolicy(), WasteMatchPolicy(),
            GroupExchangePolicy()]
