"""CP-SAT slave for the exact-selection matheuristic.

Two entry points built on OR-Tools CP-SAT:

cp_pack(...)      decision version: can THIS subset be packed?
                  Three-outcome protocol (FEASIBLE with coordinates /
                  INFEASIBLE proven / UNKNOWN on budget), used as the
                  exact slave of the exchange master: feasible answers
                  are installed directly as placements, infeasible ones
                  become *exact* permanent no-goods.

cp_solve_kp(...)  optimization version: full exact 2D/3D-KP on a set
                  of items (optional intervals with presence literals,
                  maximise total profit). Used as an optimality probe
                  on selection-dominated plateaus (e.g. okp5): any
                  improving solution or a proven optimum settles the
                  question the heuristic cannot.

Modelling notes
---------------
2D uses the native AddNoOverlap2D global constraint (optional
intervals supported). 3D has no native counterpart: pairwise
six-way disjunctions with OnlyEnforceIf, with rotation handled via
AddAllowedAssignments on the (sx, sy, sz) size variables. Incumbent
coordinates are passed as hints, never fixed, preserving completeness.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

from .geometry import Item, Knapsack, Placement

FEASIBLE, INFEASIBLE, UNKNOWN = "FEASIBLE", "INFEASIBLE", "UNKNOWN"


# ----------------------------------------------------------------------
def _rotations(it: Item, ks: Knapsack, allow_rotation: bool):
    rots = [r for r in it.rotations(allow_rotation, ks.is_3d)
            if r[0] <= ks.W and r[1] <= ks.D and r[2] <= ks.H]
    return rots


# ----------------------------------------------------------------------
def cp_pack(items: Sequence[Item], ks: Knapsack, allow_rotation: bool,
            time_limit: float = 1.0,
            hints: Optional[Dict[int, Tuple[int, int, int]]] = None,
            workers: int = 4):
    """Decision: pack ALL `items` into the knapsack.

    Returns (status, placements|None) with status in
    {FEASIBLE, INFEASIBLE, UNKNOWN}."""
    m = cp_model.CpModel()
    n = len(items)
    xs, ys, zs, sx, sy, sz = [], [], [], [], [], []

    for it in items:
        rots = _rotations(it, ks, allow_rotation)
        if not rots:
            return INFEASIBLE, None
        x = m.NewIntVar(0, ks.W, f"x{it.idx}")
        y = m.NewIntVar(0, ks.D, f"y{it.idx}")
        z = m.NewIntVar(0, max(ks.H - 1, 0), f"z{it.idx}") if ks.is_3d \
            else m.NewConstant(0)
        wv = m.NewIntVarFromDomain(
            cp_model.Domain.FromValues(sorted({r[0] for r in rots})), f"w{it.idx}")
        dv = m.NewIntVarFromDomain(
            cp_model.Domain.FromValues(sorted({r[1] for r in rots})), f"d{it.idx}")
        hv = m.NewIntVarFromDomain(
            cp_model.Domain.FromValues(sorted({r[2] for r in rots})), f"h{it.idx}")
        m.AddAllowedAssignments([wv, dv, hv], rots)
        m.Add(x + wv <= ks.W)
        m.Add(y + dv <= ks.D)
        if ks.is_3d:
            m.Add(z + hv <= ks.H)
        else:
            m.Add(hv == 1)
        xs.append(x); ys.append(y); zs.append(z)
        sx.append(wv); sy.append(dv); sz.append(hv)

    if not ks.is_3d:
        ivx, ivy = [], []
        for i in range(n):
            ex = m.NewIntVar(0, ks.W, f"ex{i}")
            m.Add(ex == xs[i] + sx[i])
            ey = m.NewIntVar(0, ks.D, f"ey{i}")
            m.Add(ey == ys[i] + sy[i])
            ivx.append(m.NewIntervalVar(xs[i], sx[i], ex, f"ix{i}"))
            ivy.append(m.NewIntervalVar(ys[i], sy[i], ey, f"iy{i}"))
        m.AddNoOverlap2D(ivx, ivy)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                b = [m.NewBoolVar(f"s{i}_{j}_{k}") for k in range(6)]
                m.AddBoolOr(b)
                m.Add(xs[i] + sx[i] <= xs[j]).OnlyEnforceIf(b[0])
                m.Add(xs[j] + sx[j] <= xs[i]).OnlyEnforceIf(b[1])
                m.Add(ys[i] + sy[i] <= ys[j]).OnlyEnforceIf(b[2])
                m.Add(ys[j] + sy[j] <= ys[i]).OnlyEnforceIf(b[3])
                m.Add(zs[i] + sz[i] <= zs[j]).OnlyEnforceIf(b[4])
                m.Add(zs[j] + sz[j] <= zs[i]).OnlyEnforceIf(b[5])

    if hints:
        for i, it in enumerate(items):
            if it.idx in hints:
                hx, hy, hz = hints[it.idx]
                m.AddHint(xs[i], hx)
                m.AddHint(ys[i], hy)
                if ks.is_3d:
                    m.AddHint(zs[i], hz)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        placements = [Placement(it, solver.Value(xs[i]), solver.Value(ys[i]),
                                solver.Value(zs[i]) if ks.is_3d else 0,
                                solver.Value(sx[i]), solver.Value(sy[i]),
                                solver.Value(sz[i]))
                      for i, it in enumerate(items)]
        return FEASIBLE, placements
    if st == cp_model.INFEASIBLE:
        return INFEASIBLE, None
    return UNKNOWN, None


# ----------------------------------------------------------------------
def cp_solve_kp(items: Sequence[Item], ks: Knapsack, allow_rotation: bool,
                time_limit: float = 60.0,
                warm: Optional[List[Placement]] = None,
                workers: int = 8):
    """Optimization: exact 2D/3D-KP (subset selection + packing).

    Returns (best_profit, upper_bound, placements, proven_optimal)."""
    m = cp_model.CpModel()
    n = len(items)
    pres, xs, ys, zs, sx, sy, sz = [], [], [], [], [], [], []

    for it in items:
        rots = _rotations(it, ks, allow_rotation)
        p = m.NewBoolVar(f"p{it.idx}")
        if not rots:
            m.Add(p == 0)
            rots = [(it.w, it.d, it.h)]
        x = m.NewIntVar(0, ks.W, f"x{it.idx}")
        y = m.NewIntVar(0, ks.D, f"y{it.idx}")
        z = m.NewIntVar(0, max(ks.H - 1, 0), f"z{it.idx}") if ks.is_3d \
            else m.NewConstant(0)
        wv = m.NewIntVarFromDomain(
            cp_model.Domain.FromValues(sorted({r[0] for r in rots})), f"w{it.idx}")
        dv = m.NewIntVarFromDomain(
            cp_model.Domain.FromValues(sorted({r[1] for r in rots})), f"d{it.idx}")
        hv = m.NewIntVarFromDomain(
            cp_model.Domain.FromValues(sorted({r[2] for r in rots})), f"h{it.idx}")
        m.AddAllowedAssignments([wv, dv, hv], rots)
        m.Add(x + wv <= ks.W).OnlyEnforceIf(p)
        m.Add(y + dv <= ks.D).OnlyEnforceIf(p)
        if ks.is_3d:
            m.Add(z + hv <= ks.H).OnlyEnforceIf(p)
        pres.append(p)
        xs.append(x); ys.append(y); zs.append(z)
        sx.append(wv); sy.append(dv); sz.append(hv)

    if not ks.is_3d:
        ivx, ivy = [], []
        for i in range(n):
            ex = m.NewIntVar(0, ks.W, f"ex{i}")
            m.Add(ex == xs[i] + sx[i])
            ey = m.NewIntVar(0, ks.D, f"ey{i}")
            m.Add(ey == ys[i] + sy[i])
            ivx.append(m.NewOptionalIntervalVar(xs[i], sx[i], ex,
                                                pres[i], f"ix{i}"))
            ivy.append(m.NewOptionalIntervalVar(ys[i], sy[i], ey,
                                                pres[i], f"iy{i}"))
        m.AddNoOverlap2D(ivx, ivy)
        # redundant energy cut
        m.Add(sum(pres[i] * items[i].base_area for i in range(n))
              <= ks.W * ks.D)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                b = [m.NewBoolVar(f"s{i}_{j}_{k}") for k in range(6)]
                m.AddBoolOr(b + [pres[i].Not(), pres[j].Not()])
                m.Add(xs[i] + sx[i] <= xs[j]).OnlyEnforceIf(b[0])
                m.Add(xs[j] + sx[j] <= xs[i]).OnlyEnforceIf(b[1])
                m.Add(ys[i] + sy[i] <= ys[j]).OnlyEnforceIf(b[2])
                m.Add(ys[j] + sy[j] <= ys[i]).OnlyEnforceIf(b[3])
                m.Add(zs[i] + sz[i] <= zs[j]).OnlyEnforceIf(b[4])
                m.Add(zs[j] + sz[j] <= zs[i]).OnlyEnforceIf(b[5])
        m.Add(sum(pres[i] * items[i].volume for i in range(n)) <= ks.volume)

    m.Maximize(sum(pres[i] * items[i].profit for i in range(n)))

    if warm:
        pos = {p.item.idx: p for p in warm}
        for i, it in enumerate(items):
            if it.idx in pos:
                pl = pos[it.idx]
                m.AddHint(pres[i], 1)
                m.AddHint(xs[i], pl.x)
                m.AddHint(ys[i], pl.y)
                if ks.is_3d:
                    m.AddHint(zs[i], pl.z)
            else:
                m.AddHint(pres[i], 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0.0, float("inf"), None, False
    placements = [Placement(it, solver.Value(xs[i]), solver.Value(ys[i]),
                            solver.Value(zs[i]) if ks.is_3d else 0,
                            solver.Value(sx[i]), solver.Value(sy[i]),
                            solver.Value(sz[i]))
                  for i, it in enumerate(items) if solver.Value(pres[i])]
    return (solver.ObjectiveValue(), solver.BestObjectiveBound(),
            placements, st == cp_model.OPTIMAL)
