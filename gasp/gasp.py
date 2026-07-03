"""GASP - Greedy Adaptive Search Procedure (Section 3 of the paper).

The metaheuristic separates the feasibility phase (handled by the
EP-KPH greedy) from the optimality phase (the order in which items are
presented to the greedy, driven by item scores).

Main loop (Fig. 1):
  1. Build an initial solution with the PCH and set it as best (BS).
  2. Initialise the scores.
  3. Until the stopping conditions are met:
       - sort items by non-increasing score, run the greedy -> CS;
       - on too many non-improving iterations, apply the Long-term
         Score Reinitialization (which also cycles the merit
         function); otherwise apply the Score Update on CS;
       - if CS > BS, set BS = CS;
       - apply the Parameter Update.

Stopping conditions: best solution equals a known optimum/upper bound,
or the time limit is reached (10 s by default, as in the paper).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .geometry import Item, Knapsack, Packing
from .greedy import ep_kph
from .merit import MERIT_SEQUENCE
from .sorting import SORTING_RULES


@dataclass
class GASPParams:
    # Defaults below are the cfg03 configuration selected by the
    # parameter tuning campaign (examples/tune_params.py): it ranked
    # first on BOTH the BR (3D-CLP) and the classic 2D validation
    # families, beating the paper baseline. Paper/original values are
    # noted in comments for reproducibility.
    time_limit: float = 10.0          # seconds (paper: 10 s for 2D/3D-KP)
    alpha: float = 0.2                # score decrease (paper: 0.1)
    beta: float = 0.1                 # score increase (Section 3.4)
    k_init: int = 2                   # initial score multiplier (paper: 3)
    non_improving_limit: int = 10     # reinit threshold (paper-style: 30)
    pch_deltas: tuple = (10,)         # delta values tried by the clustered PCH rules
    exact_repair: bool = False        # exchange-master repair at stagnation
    basin_probe: bool = False         # columnar CP probe at deep stagnation
    layer_greedy: bool = False        # VNS-style layer front-stage in the initial solution
    use_ems: bool = False             # Empty Maximal Spaces placement backend (3D)
    layout_search: bool = False       # physical-layout local search at stagnation (3D)
    parreno_seed: bool = False        # seed initial solution with Parreno block constructive (3D)
    block_mode: str = "off"           # EP block loading: off|max|bestfit (3D)
    reinit_swaps: int = 3             # random score swaps in the reinit (was 5)
    allow_rotation: bool = False
    seed: Optional[int] = None
    known_optimum: Optional[float] = None  # stop early if reached
    # --- subset-based score update policies -------------------------
    # "classic": the single (j, l) pair swap of Section 3.4
    # "band":    core-style perturbation restricted to the critical
    #            band around the loaded/unloaded frontier
    # "waste":   spatial policy driven by the residual free spaces
    # "adaptive": ALNS-style roulette over the three policies, with
    #            weights rewarded on best-solution improvements
    update_policy: str = "classic"
    band_fraction: float = 0.05       # band size as fraction of n (min 2)
    policy_reward: float = 1.0        # weight added on improvement
    policy_decay: float = 0.99        # multiplicative decay per iteration


@dataclass
class GASPResult:
    best_packing: Packing
    best_profit: float
    iterations: int
    elapsed: float
    history: List[float] = field(default_factory=list)
    # stage decomposition (3D Container Loading diagnostics): used volume
    # of the initial seed and of the incumbent just before the final
    # layout-search post-process. None when not applicable.
    seed_volume: Optional[int] = None
    pre_layout_volume: Optional[int] = None


class GASP:
    def __init__(self, items: List[Item], knapsack: Knapsack,
                 params: Optional[GASPParams] = None):
        self.items = items
        self.knapsack = knapsack
        self.p = params or GASPParams()
        self.rng = random.Random(self.p.seed)

        self.scores: Dict[int, float] = {}
        self.k = self.p.k_init
        # long-term memories f^l (loaded) and f^u (unloaded), Section 3.4
        self.f_loaded: Dict[int, int] = {i.idx: 0 for i in items}
        self.f_unloaded: Dict[int, int] = {i.idx: 0 for i in items}
        self.merit_idx = 0  # index in MERIT_SEQUENCE, starts from RS
        self.policy_weights: Dict[str, float] = {
            "classic": 1.0, "band": 1.0, "waste": 1.0}

    # ------------------------------------------------------------------
    # Section 3.2 - Initial solution (PCH)
    def _place(self, ordered, criterion, return_eps=False):
        """Placement backend selector: EMS for 3D when enabled, block
        greedy when a block_mode is set, else the classical EP greedy.
        Both alternatives ignore the EP-specific return_eps channel (the
        waste policy is EP-only), returning (packing, None)."""
        if getattr(self.p, "use_ems", False) and self.knapsack.is_3d:
            from .ems import EMSGreedy
            crit = "BSS" if criterion in ("RS", "TP") else "VOL"
            pk = EMSGreedy(self.knapsack, crit,
                           self.p.allow_rotation).run(ordered)
            return (pk, None) if return_eps else pk
        bmode = getattr(self.p, "block_mode", "off")
        if bmode != "off" and self.knapsack.is_3d:
            from .block_greedy import ep_kph_blocks
            pk = ep_kph_blocks(ordered, self.knapsack, criterion,
                               self.p.allow_rotation, bmode)
            return (pk, None) if return_eps else pk
        return ep_kph(ordered, self.knapsack, criterion=criterion,
                      allow_rotation=self.p.allow_rotation,
                      return_eps=return_eps)

    def initial_solution(self) -> Packing:
        best = None
        if getattr(self.p, "parreno_seed", False) and self.knapsack.is_3d:
            from .parreno_construct import parreno_construct
            best = parreno_construct(self.items, self.knapsack,
                                     self.p.allow_rotation, "bestvol")
        if getattr(self.p, "layer_greedy", False) and self.knapsack.is_3d:
            from .layer_greedy import LayerGreedy
            lg = LayerGreedy(self.knapsack, "RS", self.p.allow_rotation,
                             min_layer_fill=0.45, max_second_types=2)
            cand = lg.run(self.items)
            if best is None or cand.profit > best.profit:
                best = cand
        for rule in SORTING_RULES:
            clustered = "clustered" in rule.__name__
            deltas = self.p.pch_deltas if clustered else (10,)
            for delta in deltas:
                ordered = rule(self.items, self.knapsack, delta=delta)
                packing = self._place(ordered, "RS")
                if best is None or packing.profit > best.profit:
                    best = packing
        return best

    # ------------------------------------------------------------------
    # Section 3.3 - Score initialization
    def init_scores(self, reference: Packing, k: Optional[int] = None) -> None:
        k = k if k is not None else self.p.k_init
        loaded = reference.loaded_ids
        for item in self.items:
            self.scores[item.idx] = k * item.profit if item.idx in loaded else item.profit

    # ------------------------------------------------------------------
    # Section 3.4 - Score update (classic single-pair policy)
    def update_scores(self, current: Packing,
                      update_memories: bool = True) -> None:
        loaded = current.loaded_ids
        if update_memories:
            for item in self.items:
                if item.idx in loaded:
                    self.f_loaded[item.idx] += 1
                else:
                    self.f_unloaded[item.idx] += 1

        loaded_items = [i for i in self.items if i.idx in loaded]
        unloaded_items = [i for i in self.items if i.idx not in loaded]
        if not loaded_items or not unloaded_items:
            return

        # least valuable loaded item: minimise p/(w*l) * (1 + f^l)
        j = min(loaded_items,
                key=lambda i: (i.profit / i.base_area) * (1 + self.f_loaded[i.idx]))
        # most valuable unloaded item: maximise p / (w*l*(1 + f^u))
        l = max(unloaded_items,
                key=lambda i: i.profit / (i.base_area * (1 + self.f_unloaded[i.idx])))

        self.scores[j.idx] *= (1 - self.p.alpha)
        self.scores[l.idx] *= (1 + self.p.beta)
        self.scores[j.idx], self.scores[l.idx] = self.scores[l.idx], self.scores[j.idx]

    # ------------------------------------------------------------------
    # Band policy: perturb only the critical band around the frontier
    # (core-problem idea: Balas-Zemel / Pisinger expanding core)
    def update_scores_band(self, current: Packing) -> None:
        loaded = current.loaded_ids
        loaded_items = [i for i in self.items if i.idx in loaded]
        unloaded_items = [i for i in self.items if i.idx not in loaded]
        if not loaded_items or not unloaded_items:
            return
        b = max(2, int(len(self.items) * self.p.band_fraction))

        # frontier band: the b least efficient loaded items and the b
        # most efficient unloaded ones (long-term memories included to
        # avoid cycling on the same subset)
        worst_in = sorted(
            loaded_items,
            key=lambda i: (i.profit / i.base_area) * (1 + self.f_loaded[i.idx])
        )[:b]
        best_out = sorted(
            unloaded_items,
            key=lambda i: -i.profit / (i.base_area * (1 + self.f_unloaded[i.idx]))
        )[:b]

        # swap a random number of pairs inside the band only
        n_pairs = self.rng.randint(1, min(len(worst_in), len(best_out)))
        self.rng.shuffle(worst_in)
        self.rng.shuffle(best_out)
        for j, l in zip(worst_in[:n_pairs], best_out[:n_pairs]):
            self.scores[j.idx] *= (1 - self.p.alpha)
            self.scores[l.idx] *= (1 + self.p.beta)
            self.scores[j.idx], self.scores[l.idx] = \
                self.scores[l.idx], self.scores[j.idx]

    # ------------------------------------------------------------------
    # Waste policy: spatial diagnosis of the current packing.
    # Promote unloaded items that fit into the largest residual free
    # spaces; demote the loaded items with the lowest profit density.
    def update_scores_waste(self, current: Packing, epm) -> None:
        loaded = current.loaded_ids
        unloaded_items = [i for i in self.items if i.idx not in loaded]
        loaded_items = [i for i in self.items if i.idx in loaded]
        if not unloaded_items or not loaded_items or epm is None:
            return

        # largest residual free boxes among the surviving EPs
        free_boxes = sorted(
            ((ep.rs_x, ep.rs_y, ep.rs_z) for ep in epm.eps),
            key=lambda r: -(r[0] * r[1] * r[2]))[:5]
        if not free_boxes:
            return

        is_3d = self.knapsack.is_3d

        def fits_somewhere(item) -> bool:
            for (rx, ry, rz) in free_boxes:
                for (w, d, h) in item.rotations(self.p.allow_rotation, is_3d):
                    if w <= rx and d <= ry and h <= rz:
                        return True
            return False

        candidates = [i for i in unloaded_items if fits_somewhere(i)]
        if not candidates:
            return
        candidates.sort(key=lambda i: -i.profit / i.volume)
        demote = sorted(loaded_items, key=lambda i: i.profit / i.volume)

        for j, l in zip(demote[:3], candidates[:3]):
            self.scores[j.idx] *= (1 - self.p.alpha)
            self.scores[l.idx] *= (1 + self.p.beta)
            self.scores[j.idx], self.scores[l.idx] = \
                self.scores[l.idx], self.scores[j.idx]

    # ------------------------------------------------------------------
    # Adaptive policy selection (ALNS-style roulette wheel)
    def _select_policy(self) -> str:
        if self.p.update_policy != "adaptive":
            return self.p.update_policy
        total = sum(self.policy_weights.values())
        r = self.rng.uniform(0, total)
        acc = 0.0
        for name, w in self.policy_weights.items():
            acc += w
            if r <= acc:
                return name
        return "classic"

    # ------------------------------------------------------------------
    # Section 3.5 - Long-term score reinitialization
    def long_term_reinit(self, best: Packing) -> None:
        self.init_scores(best, k=self.p.k_init)
        ids = [i.idx for i in self.items]
        for _ in range(self.p.reinit_swaps):
            a, b = self.rng.sample(ids, 2)
            self.scores[a], self.scores[b] = self.scores[b], self.scores[a]
        # cycle to the next merit function (RS -> MP -> LEV -> FF -> RS ...)
        self.merit_idx = (self.merit_idx + 1) % len(MERIT_SEQUENCE)
        # Section 3.6 - k reset after each reinitialization
        self.k = 1

    # ------------------------------------------------------------------
    def run(self) -> GASPResult:
        start = time.time()

        best = self.initial_solution()
        self.init_scores(best)
        history = [best.profit]

        iterations = 0
        non_improving = 0

        while True:
            elapsed = time.time() - start
            if elapsed >= self.p.time_limit:
                break
            if (self.p.known_optimum is not None
                    and best.profit >= self.p.known_optimum):
                break

            iterations += 1
            ordered = sorted(self.items,
                             key=lambda i: -self.scores[i.idx])
            criterion = MERIT_SEQUENCE[self.merit_idx]
            policy = self._select_policy()
            need_eps = (policy == "waste")
            out = self._place(ordered, criterion, return_eps=need_eps)
            current, epm = out if need_eps else (out, None)

            if current.profit > best.profit:
                best = current
                non_improving = 0
                # Section 3.6 - k increased on each best update
                self.k += 1
                # ALNS-style reward to the policy that produced the move
                if self.p.update_policy == "adaptive":
                    self.policy_weights[policy] += self.p.policy_reward
            else:
                non_improving += 1
                if self.p.update_policy == "adaptive":
                    self.policy_weights[policy] = max(
                        0.1, self.policy_weights[policy] * self.p.policy_decay)

            if non_improving >= self.p.non_improving_limit:
                self.long_term_reinit(best)
                non_improving = 0
            else:
                # long-term memories are shared by all the policies
                loaded = current.loaded_ids
                for item in self.items:
                    if item.idx in loaded:
                        self.f_loaded[item.idx] += 1
                    else:
                        self.f_unloaded[item.idx] += 1
                if policy == "band":
                    self.update_scores_band(current)
                elif policy == "waste":
                    self.update_scores_waste(current, epm)
                else:
                    self.update_scores(current, update_memories=False)

            history.append(best.profit)

        return GASPResult(best_packing=best, best_profit=best.profit,
                          iterations=iterations,
                          elapsed=time.time() - start, history=history)
