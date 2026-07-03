"""Spawn-safe worker for parallel campaigns and parameter tuning.

Each worker process keeps a private cache of parsed instance sets, so
big sets (e.g. ngcutfs) are read once per process instead of once per
task. The function is importable by module path, which is what
multiprocessing's *spawn* start method (the default on Windows)
requires. The numba kernels are compiled with cache=True, so worker
processes load them from the on-disk cache instead of recompiling.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

_CACHE: Dict[str, list] = {}


def _get_instance(set_name: str, inst_idx: int, respect_orientation: bool = False):
    from .readers import load_set
    key = f"{set_name}|ori" if respect_orientation else set_name
    if key not in _CACHE:
        _CACHE[key] = load_set(set_name, respect_orientation=respect_orientation)
    return _CACHE[key][inst_idx]


_WARMED = False


def _ensure_warm():
    """Compile the numba kernels once per process, before any timed run,
    so JIT compilation never falls inside a measured solve. Safe no-op
    when numba is absent (pure-Python backend)."""
    global _WARMED
    if _WARMED:
        return
    try:
        from .fast_greedy import warmup
        warmup()
    except Exception:
        pass
    try:
        from .ems_numba import warmup_dominance
        warmup_dominance()
    except Exception:
        pass
    _WARMED = True


def run_one(set_name: str, inst_idx: int, seed: int,
            time_limit: float, rotation: bool,
            overrides: Optional[dict] = None):
    """Run one solve. Returns (instance_name, profit, fill%, seed_fill%,
    pre_layout_fill%, elapsed_s).

    Backend selected by overrides["solver"]: "gasp" (default, the
    AdaptiveGASP pipeline) or "alns" (the standalone container-loading
    ALNS). For ALNS, seed_fill/pre_fill are None (GASP-specific) and
    `overrides` may carry ALNSParams fields (frac_lo, frac_hi, cooling,
    T0_ratio, ...). For GASP, `overrides` may contain any GASPParams
    field. The reported elapsed time excludes numba JIT warm-up.
    """
    _ensure_warm()
    kw = dict(overrides or {})
    respect = bool(kw.pop("respect_orientation", False))
    solver = kw.pop("solver", "gasp")
    name, items, ks = _get_instance(set_name, inst_idx, respect)

    if solver == "alns":
        from .alns import solve_alns, ALNSParams
        valid = set(ALNSParams.__dataclass_fields__.keys())
        akw = {k: v for k, v in kw.items() if k in valid}
        params = ALNSParams(time_limit=time_limit, seed=seed,
                            allow_rotation=rotation, **akw)
        r = solve_alns(items, ks, params)
        fill = r.best_fill
        profit = r.best_packing.profit
        return (name, float(profit), round(fill, 2), None, None,
                round(r.elapsed, 3))

    from . import GASPParams
    from .adaptive import AdaptiveGASP

    if "pch_deltas" in kw and isinstance(kw["pch_deltas"], list):
        kw["pch_deltas"] = tuple(kw["pch_deltas"])
    params = GASPParams(time_limit=time_limit, seed=seed,
                        allow_rotation=rotation, **kw)
    r = AdaptiveGASP(items, ks, params).run()
    fill = 100.0 * r.best_packing.used_volume / ks.volume
    # stage decomposition (3D): seed fill and pre-layout fill, when known
    seed_fill = (round(100.0 * r.seed_volume / ks.volume, 2)
                 if r.seed_volume is not None else None)
    pre_fill = (round(100.0 * r.pre_layout_volume / ks.volume, 2)
                if r.pre_layout_volume is not None else None)
    return (name, float(r.best_profit), round(fill, 2), seed_fill,
            pre_fill, round(r.elapsed, 3))
