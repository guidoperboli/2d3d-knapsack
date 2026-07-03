"""Adaptive variant of GASP: policy portfolio + warm-started greedy.

Differences w.r.t. the baseline GASP (gasp.gasp.GASP):

1. The fixed Score Update of Section 3.4 is replaced by a portfolio of
   targeted policies (gasp.policies) chosen at each iteration by an
   ALNS-style roulette that learns which policy pays off.
2. Each policy reports the earliest ordering position it touched: when
   the new ordering shares a prefix with the previous one, the greedy
   is warm-started from a cached snapshot instead of re-running from
   scratch (the EP-KPH state after k items depends only on the first k
   items of the order). Snapshots are taken at fixed fractions of the
   sequence during each greedy run.
3. The Long-term Score Reinitialization additionally resets the
   roulette weights (same place where the merit function is cycled),
   and BlockSwap's k grows under stagnation.

Everything else (score initialization, long-term reinit, parameter k,
stopping conditions) follows the original scheme.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .gasp import GASP, GASPParams, GASPResult
from .geometry import Item, Knapsack, Packing
from .greedy import GreedyState, make_greedy_state
from .merit import MERIT_SEQUENCE
from .policies import (BlockSwapPolicy, PolicyContext, PolicySelector,
                       default_portfolio)

SNAPSHOT_FRACTIONS = (0.25, 0.5, 0.75)


class AdaptiveGASP(GASP):
    def __init__(self, items: List[Item], knapsack: Knapsack,
                 params: Optional[GASPParams] = None,
                 policies=None):
        super().__init__(items, knapsack, params)
        self.selector = PolicySelector(policies or default_portfolio())
        # cache of the last greedy run
        self._last_order: List[Item] = []
        self._snapshots: List[GreedyState] = []   # sorted by n_processed
        self._last_criterion: Optional[str] = None
        self._master = None
        self.repair_stats = {"calls": 0, "accepted": 0, "tried": 0}
        self._repair_fruitless = 0   # consecutive fully-refuted sweeps
        self._reinit_streak = 0      # reinits since the last global best
        self._probes_done = 0        # columnar probes spent this run
        self._layout_fruitless = 0   # consecutive fruitless layout searches
        self._t_start = 0.0

    # ------------------------------------------------------------------
    def _exact_repair(self, best: Packing) -> Optional[Packing]:
        """Exchange-master repair: enumerate proven profit-improving,
        conflict-feasible marginal exchanges and let the greedy slave
        verify geometric feasibility. Failed exchanges become no-good
        cuts (logic-based Benders style)."""
        if self._master is None:
            from .exact_master import ExchangeMaster
            self._master = ExchangeMaster(self.items, self.knapsack,
                                          self.p.allow_rotation)
        self.repair_stats["calls"] += 1
        exchanges = self._master.exchanges(best, top_k=15)
        if not exchanges:
            return None

        loaded_ids = best.loaded_ids
        by_score = sorted(self.items, key=lambda i: -self.scores[i.idx])
        hints = {p.item.idx: (p.x, p.y, p.z)
                 for p in best.placements}
        for gain, S, T in exchanges:
            self.repair_stats["tried"] += 1
            s_ids = {i.idx for i in S}
            t_ids = {i.idx for i in T}
            kept = [i for i in by_score
                    if i.idx in loaded_ids and i.idx not in s_ids]
            rest = [i for i in by_score
                    if i.idx not in loaded_ids and i.idx not in t_ids]

            # --- exact CP slave (three-outcome protocol) on small subsets
            subset = kept + list(T)
            if len(subset) <= 25:
                from .cp_slave import FEASIBLE, INFEASIBLE, cp_pack
                st, pl = cp_pack(subset, self.knapsack,
                                 self.p.allow_rotation, time_limit=0.15,
                                 hints=hints)
                if st == FEASIBLE:
                    self.repair_stats["accepted"] += 1
                    return Packing(self.knapsack, pl)
                if st == INFEASIBLE:
                    # exact permanent no-good
                    self._master.add_nogood(
                        frozenset((loaded_ids - s_ids) | t_ids))
                    continue
                # UNKNOWN: fall through to the greedy attempt

            order = sorted(T, key=lambda i: -i.profit) + kept + rest + list(S)
            target = best.profit + gain
            improved = None
            for crit in MERIT_SEQUENCE:
                st = make_greedy_state(self.knapsack, crit,
                                       self.p.allow_rotation)
                packing = st.run(order)
                if packing.profit > best.profit:
                    if improved is None or packing.profit > improved.profit:
                        improved = packing
                    if packing.profit >= target:
                        break
            if improved is not None:
                self.repair_stats["accepted"] += 1
                return improved
            self._master.add_nogood(
                frozenset((loaded_ids - s_ids) | t_ids))
        return None

    # ------------------------------------------------------------------
    def _run_greedy(self, order: List[Item], criterion: str,
                    boundary: Optional[int]) -> Packing:
        """Run EP-KPH, warm-starting from a cached snapshot when the
        ordering prefix up to `boundary` is unchanged."""
        if getattr(self.p, "use_ems", False) and self.knapsack.is_3d:
            from .ems import EMSGreedy
            crit = "BSS" if criterion in ("RS", "TP") else "VOL"
            return EMSGreedy(self.knapsack, crit,
                             self.p.allow_rotation).run(order)
        bmode = getattr(self.p, "block_mode", "off")
        if bmode != "off" and self.knapsack.is_3d:
            from .block_greedy import ep_kph_blocks
            return ep_kph_blocks(order, self.knapsack, criterion,
                                 self.p.allow_rotation, bmode)
        start_state: Optional[GreedyState] = None

        if (boundary is not None and criterion == self._last_criterion
                and self._snapshots):
            # longest common prefix between old and new ordering
            lcp = 0
            for a, b in zip(self._last_order, order):
                if a.idx != b.idx:
                    break
                lcp += 1
            usable = min(lcp, boundary)
            for snap in reversed(self._snapshots):
                if snap.n_processed <= usable:
                    start_state = snap.clone()
                    break

        if start_state is None:
            state = make_greedy_state(self.knapsack, criterion,
                                      self.p.allow_rotation)
            start_from = 0
        else:
            state = start_state
            start_from = state.n_processed

        n = len(order)
        marks = sorted({max(1, int(f * n)) for f in SNAPSHOT_FRACTIONS})
        new_snaps = [s for s in self._snapshots
                     if s.n_processed <= start_from] if start_state else []

        for k in range(start_from, n):
            state.place(order[k])
            if (k + 1) in marks:
                new_snaps.append(state.clone())

        self._last_order = list(order)
        self._snapshots = new_snaps
        self._last_criterion = criterion
        self._last_state = state
        return state.packing

    # ------------------------------------------------------------------
    def run(self) -> GASPResult:
        start = time.time()
        self._t_start = start

        # if the layout-search post-process is active, reserve a slice of
        # the total budget for it, so that loop + post-process together
        # stay within time_limit (the reported time is then the true
        # total solve time, comparable to the literature).
        layout_on = (getattr(self.p, "layout_search", False)
                     and self.knapsack.is_3d)
        layout_reserve = min(3.0, 0.35 * self.p.time_limit) if layout_on else 0.0
        loop_budget = self.p.time_limit - layout_reserve

        best = self.initial_solution()
        seed_volume = best.used_volume       # stage decomposition: seed
        self.init_scores(best)
        history = [best.profit]

        iterations = 0
        non_improving = 0
        prev_profit = best.profit
        boundary: Optional[int] = None
        last_packing: Optional[Packing] = None

        block = next((p for p in self.selector.policies
                      if isinstance(p, BlockSwapPolicy)), None)

        while True:
            if time.time() - start >= loop_budget:
                break
            if (self.p.known_optimum is not None
                    and best.profit >= self.p.known_optimum):
                break

            iterations += 1
            order = sorted(self.items, key=lambda i: -self.scores[i.idx])
            criterion = MERIT_SEQUENCE[self.merit_idx]
            current = self._run_greedy(order, criterion, boundary)

            # ---- rewards & bookkeeping
            if current.profit > best.profit:
                best = current
                non_improving = 0
                self._reinit_streak = 0
                self.k += 1
                self.selector.reward(3.0)
                if block:
                    block.on_improvement()
            else:
                non_improving += 1
                self.selector.reward(
                    1.0 if current.profit > prev_profit else 0.0)
                if block:
                    block.on_stagnation()
            prev_profit = current.profit

            # ---- update memories (as in Section 3.4)
            loaded = current.loaded_ids
            for item in self.items:
                if item.idx in loaded:
                    self.f_loaded[item.idx] += 1
                else:
                    self.f_unloaded[item.idx] += 1

            # ---- next perturbation
            if non_improving >= self.p.non_improving_limit:
                # columnar CP probe at deep stagnation (basin escape)
                if (self.p.basin_probe and self._reinit_streak >= 2
                        and self._probes_done < 2):
                    import time as _time
                    remaining = self.p.time_limit - (_time.time()
                                                     - self._t_start)
                    if remaining > 4.0:
                        from .basin_probe import columnar_probe
                        self._probes_done += 1
                        budget = min(6.0, max(1.5,
                                              0.15 * self.p.time_limit),
                                     remaining / 3)
                        seeded = columnar_probe(
                            self.items, self.knapsack,
                            self.p.allow_rotation,
                            current_best=best.profit,
                            time_per_model=0.5,
                            total_budget=budget)
                        if seeded is not None:
                            best = seeded
                            self.init_scores(best)
                            non_improving = 0
                            self._reinit_streak = 0
                            boundary = None
                            continue
                do_repair = (self.p.exact_repair
                             and self._repair_fruitless < 2)
                repaired = self._exact_repair(best) if do_repair else None
                if do_repair:
                    self._repair_fruitless = (
                        0 if repaired is not None
                        else self._repair_fruitless + 1)
                if repaired is not None:
                    best = repaired
                    self.init_scores(best)
                    non_improving = 0
                    boundary = None
                else:
                    self.long_term_reinit(best)
                    self.selector.reset()
                    self._reinit_streak += 1
                    non_improving = 0
                    boundary = None       # criterion changed: cache invalid
            else:
                ctx = PolicyContext(
                    items=self.items, scores=self.scores,
                    f_loaded=self.f_loaded, f_unloaded=self.f_unloaded,
                    current=current, order=order,
                    residual_eps=getattr(self, "_last_state", None)
                    and getattr(self._last_state, "residual_eps", []) or [],
                    rng=self.rng, alpha=self.p.alpha, beta=self.p.beta)
                policy = self.selector.pick(self.rng)
                boundary = policy.apply(ctx)
                if boundary is None:
                    # policy was a no-op: fall back to full restart
                    boundary = 0

            history.append(best.profit)
            last_packing = current

        # final memetic intensification: refine the incumbent's physical
        # layout once, after the global search has used its budget. Kept
        # OUTSIDE the loop because reinjecting + re-scoring mid-run
        # resets the learning and hurts; as a post-process it only adds.
        pre_layout_volume = best.used_volume  # incumbent before local search
        if getattr(self.p, "layout_search", False) and self.knapsack.is_3d:
            from .layout_search import LayoutSearch
            budget = max(0.5, self.p.time_limit - (time.time() - start))
            ls = LayoutSearch(self.knapsack, self.items,
                              self.p.allow_rotation, depth=1,
                              max_seeds=12, time_budget=budget)
            refined = ls.improve(best)
            if refined.used_volume > best.used_volume:
                best = refined

        return GASPResult(best_packing=best, best_profit=best.profit,
                          iterations=iterations,
                          elapsed=time.time() - start, history=history,
                          seed_volume=seed_volume,
                          pre_layout_volume=pre_layout_volume)
