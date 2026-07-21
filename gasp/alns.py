"""Adaptive Large Neighborhood Search for the Container Loading Problem.

A COMPLETE, STANDALONE solver -- a contribution distinct from GASP. It
shares only the low-level geometry primitives (maximal spaces, block
placement, lexicographic space selection) with parreno_construct, the
same way any two solvers in this package share geometry. It does NOT use
the GASP loop, scores, policies, or adaptive portfolio: the search is a
self-contained ALNS.

REPRESENTATION  a solution is an ordered list of placed BLOCKS (box +
                member placements), the natural unit for container
                loading with multiplicity; working on blocks (not single
                boxes) keeps the dense column/layer structure that drives
                high fill on the BR instances.
DESTROY (6, adaptively weighted): random, worst (smallest), region
                (half-space), related (Shaw proximity+size), segment
                (contiguous run), radial (closest to a focus).
REPAIR (adaptively weighted): BLOCK reconstruction (Parreno block
                constructive, best-volume / best-fit) AND an EP-style
                single-box reconstruction. The adaptive weights pick
                block-vs-EP per instance -- no a-priori multiplicity
                threshold. Measured: adding EP repair takes ngcut01 and
                ngcut03 from +4.9% / +6.9% gap (blocks only) to +0.0%
                (optimum), because where multiplicity is low blocks
                degenerate and EP fills better; on high-multiplicity BR
                the weights keep favouring blocks (BR7 unchanged ~89%).
ACCEPTANCE      simulated annealing, geometric cooling, reheats.
ADAPTATION      roulette-wheel operator selection with reaction-factor
                weight updates (Ropke-Pisinger ALNS).

Objective selectable via ALNSParams.objective_metric:
  "volume"  (default) -- the CLP objective; on BR7 beats GASP at parity
            of time (+1.85 pts in the prototype).
  "profit"  -- 2D/3D knapsack; block selection, acceptance and the best
            incumbent switch to packed profit. On okp (multiplicity ~3)
            the profit-aware ALNS reaches a +1.13% mean gap vs GASP's
            +1.32%, beating it on several instances. WITHOUT multiplicity
            (ngcut, ~1-2 copies) blocks barely form, so the block ALNS
            stays behind GASP regardless of objective -- the gain comes
            only where multiplicity AND profit!=volume coexist.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .geometry import Item, Knapsack, Packing
from .parreno_construct import _near_corner, _place_block, _apply_box
from .grasp_parreno import _enumerate_configs, _construct
from .greedy import ep_kph


@dataclass
class ALNSParams:
    time_limit: float = 10.0
    max_iter: Optional[int] = None
    seed: int = 1
    allow_rotation: bool = True
    frac_lo: float = 0.15
    frac_hi: float = 0.45
    T0_ratio: float = 0.02
    cooling: float = 0.9995
    reheat_after: int = 1500
    reheat_ratio: float = 0.5
    # weight adaptation: react=reaction factor, seg_update=iterations
    # per weight update. Tuned to (0.1, 25): more frequent updates with a
    # moderate reaction give sharper operator separation than a large
    # reaction factor (e.g. on BR1 the EP repair is correctly pushed to
    # ~0.6 vs blocks ~1.1), while slightly improving fill. Where the two
    # repairs are genuinely equivalent (low multiplicity) the weights
    # stay flat, which is correct -- block==EP there.
    react: float = 0.1
    seg_update: int = 25
    reward_best: float = 4.0
    reward_better: float = 2.0
    reward_accept: float = 1.0
    # objective: "volume" (CLP, default) or "profit" (knapsack). When
    # "profit", acceptance, the best-incumbent, and the block-repair
    # criterion all switch to total packed profit; on container-loading
    # instances profit==volume so the two coincide.
    objective_metric: str = "volume"


@dataclass
class ALNSResult:
    best_packing: Packing
    best_fill: float
    iterations: int
    elapsed: float
    destroy_weights: Dict[str, float] = field(default_factory=dict)
    repair_weights: Dict[str, float] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)


def _block_center(box):
    return ((box[0]+box[3])/2.0, (box[1]+box[4])/2.0, (box[2]+box[5])/2.0)


def _block_vol(box):
    return (box[3]-box[0]) * (box[4]-box[1]) * (box[5]-box[2])


def _cfg_vol(cfg):
    _s, _t, w, d, h, nx, ny, nz, _n = cfg
    return (nx*w) * (ny*d) * (nz*h)


def _construct_blocks(items, ks, allow_rotation, objective, delta, rng):
    placements, raw = _construct(items, ks, allow_rotation, objective,
                                 delta, rng)
    blocks, idx = [], 0
    for (box, count) in raw:
        blocks.append((box, placements[idx:idx+count]))
        idx += count
    return placements, blocks


def _rebuild_ep(keep_blocks, items, ks, allow_rotation, objective):
    """EP-style repair: fill the freed maximal spaces with SINGLE boxes
    (each becomes a block of size 1), instead of multi-copy blocks. This
    is the operator that pays off on low-multiplicity instances, where
    blocks barely form; the adaptive weights decide when to use it.

    Greedy choice per space: among available items fitting the chosen
    maximal space, place the one maximising the objective (volume or
    profit), anchored at the space's near corner. Reuses the same
    maximal-space machinery as the block repair."""
    placements = []
    blocks = list(keep_blocks)
    for (box, pls) in keep_blocks:
        placements.extend(pls)
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    for (box, _pls) in keep_blocks:
        spaces = _apply_box(spaces, box)
    kept_ids = {p.item.idx for (_b, pls) in keep_blocks for p in pls}
    by_type = defaultdict(list)
    for it in items:
        by_type[(it.w, it.d, it.h)].append(it)
    avail = {k: [x for x in v if x.idx not in kept_ids]
             for k, v in by_type.items()}
    want_profit = (objective == "bestprofit")

    while spaces:
        scored = []
        for s in spaces:
            dist, sig = _near_corner(s, ks)
            scored.append((dist, -_block_vol(s), s, sig))
        scored.sort(key=lambda t: (t[0], t[1]))
        dist, nvol, s, sig = scored[0]
        fw, fd, fh = s[3]-s[0], s[4]-s[1], s[5]-s[2]

        # pick the single best item (one copy) that fits this space
        best = None
        for tkey, members in avail.items():
            if not members:
                continue
            rep = members[0]
            for (w, d, h) in rep.rotations(allow_rotation, ks.is_3d):
                if w > fw or d > fd or h > fh:
                    continue
                merit = (rep.profit if want_profit else w * d * h)
                if best is None or merit > best[0]:
                    best = (merit, tkey, w, d, h)
        if best is None:
            spaces = [sp for sp in spaces if sp != s]
            continue
        _m, tkey, w, d, h = best
        members = avail[tkey]
        pls, box = _place_block(s, sig, w, d, h, 1, 1, 1, members, 1)
        placements.extend(pls)
        blocks.append((box, pls))
        avail[tkey] = members[1:]
        spaces = _apply_box(spaces, box)
    return placements, blocks


def _rebuild(keep_blocks, items, ks, allow_rotation, objective, regret=False):
    placements = []
    blocks = list(keep_blocks)
    for (box, pls) in keep_blocks:
        placements.extend(pls)
    spaces = [(0, 0, 0, ks.W, ks.D, ks.H)]
    for (box, _pls) in keep_blocks:
        spaces = _apply_box(spaces, box)
    kept_ids = {p.item.idx for (_b, pls) in keep_blocks for p in pls}
    by_type = defaultdict(list)
    for it in items:
        by_type[(it.w, it.d, it.h)].append(it)
    avail = {k: [x for x in v if x.idx not in kept_ids]
             for k, v in by_type.items()}

    while spaces:
        scored = []
        for s in spaces:
            dist, sig = _near_corner(s, ks)
            scored.append((dist, -_block_vol(s), s, sig))
        scored.sort(key=lambda t: (t[0], t[1]))

        if regret and len(scored) >= 2:
            target, best_reg = None, None
            for (dist, nvol, s, sig) in scored[:6]:
                fw, fd, fh = s[3]-s[0], s[4]-s[1], s[5]-s[2]
                cands = _enumerate_configs(s, sig, fw, fd, fh, avail,
                                           allow_rotation, ks.is_3d, objective)
                if not cands:
                    continue
                cands.sort(key=lambda c: c[0])
                v1 = _cfg_vol(cands[0])
                v2 = _cfg_vol(cands[1]) if len(cands) > 1 else 0.0
                reg = v1 - v2
                if best_reg is None or reg > best_reg:
                    best_reg, target = reg, (s, sig, cands[0])
            if target is None:
                spaces = [sp for sp in spaces if sp != scored[0][2]]
                continue
            s, sig, pick = target
        else:
            dist, nvol, s, sig = scored[0]
            fw, fd, fh = s[3]-s[0], s[4]-s[1], s[5]-s[2]
            cands = _enumerate_configs(s, sig, fw, fd, fh, avail,
                                       allow_rotation, ks.is_3d, objective)
            if not cands:
                spaces = [sp for sp in spaces if sp != s]
                continue
            cands.sort(key=lambda c: c[0])
            pick = cands[0]

        _score, tkey, w, d, h, nx, ny, nz, ncopies = pick
        members = avail[tkey]
        pls, box = _place_block(s, sig, w, d, h, nx, ny, nz, members, ncopies)
        placements.extend(pls)
        blocks.append((box, pls))
        avail[tkey] = members[ncopies:]
        spaces = _apply_box(spaces, box)
    return placements, blocks


def _d_random(blocks, k, rng):
    idx = list(range(len(blocks))); rng.shuffle(idx)
    remove = set(idx[:k])
    return [b for i, b in enumerate(blocks) if i not in remove]


def _d_worst(blocks, k, rng):
    order = sorted(range(len(blocks)), key=lambda i: _block_vol(blocks[i][0]))
    remove = set(order[:k])
    return [b for i, b in enumerate(blocks) if i not in remove]


def _d_region(blocks, k, rng):
    if not blocks:
        return blocks
    axis = rng.randint(0, 2)
    coords = [b[0][axis] for b in blocks]
    cut = rng.uniform(min(coords), max(coords))
    side = rng.random() < 0.5
    keep = [b for b in blocks if (b[0][axis] < cut) != side]
    if not keep or len(keep) == len(blocks):
        return _d_random(blocks, k, rng)
    return keep


def _d_related(blocks, k, rng):
    if len(blocks) <= 1:
        return blocks
    seed = rng.randrange(len(blocks))
    sc = _block_center(blocks[seed][0]); sv = _block_vol(blocks[seed][0])
    box0 = blocks[seed][0]
    diag = math.sqrt(sum((box0[3+a]-box0[a])**2 for a in range(3))) + 1.0

    def rel(b):
        c = _block_center(b[0])
        dist = math.sqrt(sum((c[a]-sc[a])**2 for a in range(3)))
        vdiff = abs(_block_vol(b[0]) - sv) / (sv + 1.0)
        return dist/diag + 0.5*vdiff
    order = sorted(range(len(blocks)), key=lambda i: rel(blocks[i]))
    remove = set(order[:k])
    return [b for i, b in enumerate(blocks) if i not in remove]


def _d_segment(blocks, k, rng):
    if len(blocks) <= k:
        return _d_random(blocks, k, rng)
    start = rng.randint(0, len(blocks)-k)
    remove = set(range(start, start+k))
    return [b for i, b in enumerate(blocks) if i not in remove]


def _d_radial(blocks, k, rng):
    if len(blocks) <= 1:
        return blocks
    focus = _block_center(blocks[rng.randrange(len(blocks))][0])
    order = sorted(range(len(blocks)),
                   key=lambda i: sum((_block_center(blocks[i][0])[a]-focus[a])**2
                                     for a in range(3)))
    remove = set(order[:k])
    return [b for i, b in enumerate(blocks) if i not in remove]


DESTROY = [("random", _d_random), ("worst", _d_worst),
           ("region", _d_region), ("related", _d_related),
           ("segment", _d_segment), ("radial", _d_radial)]

REPAIR = [("greedy_vol", ("bestvol", False, "block")),
          ("greedy_fit", ("bestfit", False, "block")),
          ("ep_vol", ("bestvol", False, "ep"))]

# repair operator pool depends on the objective: profit-greedy block
# selection for knapsack, volume/best-fit for container loading. Each
# pool includes an EP (single-box) repair so the adaptive weights can
# pick block-vs-EP reconstruction per instance: blocks win on high
# multiplicity (BR), EP wins where multiplicity is low (the block
# constructive degenerates there).
REPAIR_PROFIT = [("greedy_profit", ("bestprofit", False, "block")),
                 ("greedy_fit", ("bestfit", False, "block")),
                 ("ep_profit", ("bestprofit", False, "ep"))]


def REPAIR_POOL(profit_mode):
    return REPAIR_PROFIT if profit_mode else REPAIR

# regret-2 repair is implemented in _rebuild (regret=True) and was
# benchmarked, but at a ~10s budget it costs too much per iteration to
# pay off (it roughly halves the iteration count); the greedy block
# constructive with adaptive destroy operators wins. Add the regret
# variant back to REPAIR for long-budget runs where iterations are
# plentiful: REPAIR.append(("regret_vol", ("bestvol", True))).


def _roulette(weights, rng):
    tot = sum(weights); r = rng.random()*tot; acc = 0.0
    for j, w in enumerate(weights):
        acc += w
        if r <= acc:
            return j
    return len(weights)-1


def solve_alns(items, ks, params=None, **kw):
    """Standalone ALNS for container loading (max volume) or knapsack
    (max profit), selected by params.objective_metric."""
    p = params or ALNSParams(**kw)
    rng = random.Random(p.seed)

    profit_mode = (p.objective_metric == "profit")

    def value(pk):
        return pk.profit if profit_mode else pk.used_volume

    # block-repair objective: profit-greedy when maximising profit,
    # else the volume/best-fit pair already in REPAIR.
    init_obj = "bestprofit" if profit_mode else "bestvol"
    
    # Parreno Constructive
    _pls_parr, blocks_parr = _construct_blocks(items, ks, p.allow_rotation,
                                               init_obj, 0.0, rng)
    val_parr = value(Packing(ks, _pls_parr))
    
    # EP Constructive
    items_sorted = sorted(items, key=lambda it: it.profit if profit_mode else it.volume, reverse=True)
    pk_ep = ep_kph(items_sorted, ks, criterion="RS", allow_rotation=p.allow_rotation)
    val_ep = value(pk_ep)
    
    # Take the best
    if val_ep > val_parr:
        _pls = pk_ep.placements
        cur_blocks = []
        for p_ep in _pls:
            box = (p_ep.x, p_ep.y, p_ep.z, p_ep.x2, p_ep.y2, p_ep.z2)
            cur_blocks.append((box, [p_ep]))
    else:
        _pls = _pls_parr
        cur_blocks = blocks_parr

    cur_pls = _pls
    cur_pk = Packing(ks, cur_pls)
    cur_v = value(cur_pk)
    best_pls, best_v = list(cur_pls), cur_v

    nd, nr = len(DESTROY), len(REPAIR_POOL(profit_mode))
    repair = REPAIR_POOL(profit_mode)
    dw, rw = [1.0]*nd, [1.0]*nr
    ds_score, ds_cnt = [0.0]*nd, [0]*nd
    rs_score, rs_cnt = [0.0]*nr, [0]*nr

    # temperature scaled to the objective magnitude
    scale = sum(it.profit for it in items) if profit_mode else ks.volume
    T0 = p.T0_ratio * scale
    T = T0
    start = time.time(); it = 0; since = 0; history = []

    while True:
        if p.max_iter is not None:
            if it >= p.max_iter:
                break
        elif time.time() - start >= p.time_limit:
            break

        di = _roulette(dw, rng); ri = _roulette(rw, rng)
        k = max(1, int(len(cur_blocks) * rng.uniform(p.frac_lo, p.frac_hi)))
        keep = DESTROY[di][1](cur_blocks, k, rng)
        obj, regret, mode = repair[ri][1]
        if mode == "ep":
            new_pls, new_blocks = _rebuild_ep(keep, items, ks,
                                              p.allow_rotation, obj)
        else:
            new_pls, new_blocks = _rebuild(keep, items, ks,
                                           p.allow_rotation, obj, regret)
        new_v = value(Packing(ks, new_pls))

        ds_cnt[di] += 1; rs_cnt[ri] += 1
        reward = 0.0
        if new_v > best_v:
            best_v, best_pls = new_v, list(new_pls)
            reward = p.reward_best; since = 0
        else:
            since += 1
        delta = new_v - cur_v
        accepted = False
        if delta >= 0:
            accepted = True
            if reward == 0.0:
                reward = p.reward_better if delta > 0 else p.reward_accept
        elif rng.random() < math.exp(delta / max(T, 1e-9)):
            accepted = True
            reward = max(reward, p.reward_accept)
        if accepted:
            cur_pls, cur_blocks, cur_v = new_pls, new_blocks, new_v

        ds_score[di] += reward; rs_score[ri] += reward
        history.append(float(best_v))

        it += 1
        if it % p.seg_update == 0:
            for j in range(nd):
                if ds_cnt[j]:
                    dw[j] = (1-p.react)*dw[j] + p.react*(ds_score[j]/ds_cnt[j])
                    ds_score[j] = 0.0; ds_cnt[j] = 0
            for j in range(nr):
                if rs_cnt[j]:
                    rw[j] = (1-p.react)*rw[j] + p.react*(rs_score[j]/rs_cnt[j])
                    rs_score[j] = 0.0; rs_cnt[j] = 0
        if since >= p.reheat_after:
            T = p.reheat_ratio * T0; since = 0
        else:
            T = max(T * p.cooling, 1e-9)

    best_pk = Packing(ks, best_pls)
    return ALNSResult(
        best_packing=best_pk,
        best_fill=100.0 * best_pk.used_volume / ks.volume,
        iterations=it, elapsed=time.time()-start,
        destroy_weights={DESTROY[j][0]: round(dw[j], 3) for j in range(nd)},
        repair_weights={repair[j][0]: round(rw[j], 3) for j in range(nr)},
        history=history)
