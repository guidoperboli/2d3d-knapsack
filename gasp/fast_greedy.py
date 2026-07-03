"""Numba-accelerated EP-KPH greedy.

Same semantics as ``gasp.greedy.GreedyState`` (same placements, same
tie-breaking), but the hot loops — feasibility checks, merit
evaluation, EP projection and Residual Space updates — run as njit
kernels over flat numpy arrays instead of Python objects.

State layout
------------
pl[i]  = (x, y, z, w, d, h)              placed items
eps[e] = (x, y, z, rs_x, rs_y, rs_z)     extreme points

Merit criteria are encoded as integers: 0=RS, 1=MP, 2=LEV, 3=FF, with
the same lexicographic tie-break (f, z, y, x) of the pure-Python
implementation (FF compares the EP discovery order only).

Falls back transparently: ``gasp.greedy`` exposes ``make_greedy_state``
which returns this class when numba is importable, the pure-Python one
otherwise (or when GASP_NO_NUMBA=1 is set in the environment).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numba import njit

from . import extreme_points as _epmod
from .extreme_points import ExtremePoint
from .geometry import Item, Knapsack, Packing, Placement

CRIT_ID = {"RS": 0, "MP": 1, "LEV": 2, "FF": 3, "TP": 4}
INF = np.int64(2**62)


# ======================================================================
# njit kernels
# ======================================================================
@njit(cache=True)
def _best_placement(eps, n_eps, pl, n_pl, rots, W, D, H, crit,
                    env_w, env_d):
    """Return (rot_idx, ep_idx) of the best feasible couple, or (-1,-1).

    Tie-break identical to the Python version: minimise (f, z, y, x);
    for FF the discovery order of the EP is the only key and the scan
    stops at the first compatible EP of each rotation."""
    bf0 = INF; bf1 = INF; bf2 = INF; bf3 = INF
    best_r = -1; best_e = -1
    C = (W if W > D else D) + 1

    for r in range(rots.shape[0]):
        w = rots[r, 0]; d = rots[r, 1]; h = rots[r, 2]
        for e in range(n_eps):
            x = eps[e, 0]; y = eps[e, 1]; z = eps[e, 2]
            if x + w > W or y + d > D or z + h > H:
                continue
            feasible = True
            for p in range(n_pl):
                if not (x + w <= pl[p, 0] or pl[p, 0] + pl[p, 3] <= x
                        or y + d <= pl[p, 1] or pl[p, 1] + pl[p, 4] <= y
                        or z + h <= pl[p, 2] or pl[p, 2] + pl[p, 5] <= z):
                    feasible = False
                    break
            if not feasible:
                continue

            if crit == 0:      # RS
                f = (eps[e, 3] - w) + (eps[e, 4] - d) + (eps[e, 5] - h)
                k1 = z; k2 = y; k3 = x
            elif crit == 1:    # MP
                fx = x + w - env_w
                if fx < 0:
                    fx = 0
                fy = y + d - env_d
                if fy < 0:
                    fy = 0
                f = fx + fy; k1 = z; k2 = y; k3 = x
            elif crit == 2:    # LEV
                if x + w > env_w:
                    fx = (x + w - env_w) * C
                else:
                    fx = env_w - (x + w)
                if y + d > env_d:
                    fy = (y + d - env_d) * C
                else:
                    fy = env_d - (y + d)
                f = fx + fy; k1 = z; k2 = y; k3 = x
            elif crit == 3:    # FF: discovery order only
                f = e; k1 = 0; k2 = 0; k3 = 0
            else:              # TP: maximise contact area (negated)
                c = 0
                if x == 0:
                    c += d * h
                if x + w == W:
                    c += d * h
                if y == 0:
                    c += w * h
                if y + d == D:
                    c += w * h
                if z == 0:
                    c += w * d
                if z + h == H:
                    c += w * d
                for p in range(n_pl):
                    px = pl[p, 0]; py = pl[p, 1]; pz = pl[p, 2]
                    pw = pl[p, 3]; pd = pl[p, 4]; ph = pl[p, 5]
                    if x + w == px or px + pw == x:
                        oy = min(y + d, py + pd) - max(y, py)
                        oz = min(z + h, pz + ph) - max(z, pz)
                        if oy > 0 and oz > 0:
                            c += oy * oz
                    if y + d == py or py + pd == y:
                        ox = min(x + w, px + pw) - max(x, px)
                        oz = min(z + h, pz + ph) - max(z, pz)
                        if ox > 0 and oz > 0:
                            c += ox * oz
                    if z + h == pz or pz + ph == z:
                        ox = min(x + w, px + pw) - max(x, px)
                        oy = min(y + d, py + pd) - max(y, py)
                        if ox > 0 and oy > 0:
                            c += ox * oy
                f = -c; k1 = z; k2 = y; k3 = x

            better = False
            if f < bf0:
                better = True
            elif f == bf0:
                if k1 < bf1 or (k1 == bf1 and (k2 < bf2 or
                                (k2 == bf2 and k3 < bf3))):
                    better = True
            if better:
                bf0 = f; bf1 = k1; bf2 = k2; bf3 = k3
                best_r = r; best_e = e
            if crit == 3:
                break  # FF: first compatible EP for this rotation
        if crit == 3 and best_r >= 0:
            break
    return best_r, best_e


@njit(cache=True)
def _pjx(pl, n, a, b, c):
    best = 0
    for p in range(n):
        if (pl[p, 0] + pl[p, 3] <= a and pl[p, 1] <= b < pl[p, 1] + pl[p, 4]
                and pl[p, 2] <= c < pl[p, 2] + pl[p, 5]):
            v = pl[p, 0] + pl[p, 3]
            if v > best:
                best = v
    return best


@njit(cache=True)
def _pjy(pl, n, a, b, c):
    best = 0
    for p in range(n):
        if (pl[p, 1] + pl[p, 4] <= b and pl[p, 0] <= a < pl[p, 0] + pl[p, 3]
                and pl[p, 2] <= c < pl[p, 2] + pl[p, 5]):
            v = pl[p, 1] + pl[p, 4]
            if v > best:
                best = v
    return best


@njit(cache=True)
def _pjz(pl, n, a, b, c):
    best = 0
    for p in range(n):
        if (pl[p, 2] + pl[p, 5] <= c and pl[p, 0] <= a < pl[p, 0] + pl[p, 3]
                and pl[p, 1] <= b < pl[p, 1] + pl[p, 4]):
            v = pl[p, 2] + pl[p, 5]
            if v > best:
                best = v
    return best


@njit(cache=True)
def _add_item(eps, n_eps, pl, n_pl, x, y, z, w, d, h, W, D, H, ext):
    """Update EP/placement arrays after loading an item; returns the
    new number of EPs. ``pl`` must already contain the new item at
    row n_pl - 1 (so projections skip it via index)."""
    n_others = n_pl - 1

    # ---- projections of the three corner points (Crainic et al. 2008)
    # plus composed projections (project the simple EP again along the
    # remaining orthogonal axis) — candidates kept in a fixed order so
    # that FF behaves identically across backends.
    cand = np.empty((28, 3), np.int64)
    nc = 0
    # (x+w, y, z): project on Y and on Z
    py = 0
    for p in range(n_others):
        if (pl[p, 1] + pl[p, 4] <= y and pl[p, 0] <= x + w < pl[p, 0] + pl[p, 3]
                and pl[p, 2] <= z < pl[p, 2] + pl[p, 5]):
            v = pl[p, 1] + pl[p, 4]
            if v > py:
                py = v
    cand[nc, 0] = x + w; cand[nc, 1] = py; cand[nc, 2] = z; nc += 1
    pz = 0
    for p in range(n_others):
        if (pl[p, 2] + pl[p, 5] <= z and pl[p, 0] <= x + w < pl[p, 0] + pl[p, 3]
                and pl[p, 1] <= y < pl[p, 1] + pl[p, 4]):
            v = pl[p, 2] + pl[p, 5]
            if v > pz:
                pz = v
    cand[nc, 0] = x + w; cand[nc, 1] = y; cand[nc, 2] = pz; nc += 1
    # (x, y+d, z): project on X and on Z
    px = 0
    for p in range(n_others):
        if (pl[p, 0] + pl[p, 3] <= x and pl[p, 1] <= y + d < pl[p, 1] + pl[p, 4]
                and pl[p, 2] <= z < pl[p, 2] + pl[p, 5]):
            v = pl[p, 0] + pl[p, 3]
            if v > px:
                px = v
    cand[nc, 0] = px; cand[nc, 1] = y + d; cand[nc, 2] = z; nc += 1
    pz = 0
    for p in range(n_others):
        if (pl[p, 2] + pl[p, 5] <= z and pl[p, 0] <= x < pl[p, 0] + pl[p, 3]
                and pl[p, 1] <= y + d < pl[p, 1] + pl[p, 4]):
            v = pl[p, 2] + pl[p, 5]
            if v > pz:
                pz = v
    cand[nc, 0] = x; cand[nc, 1] = y + d; cand[nc, 2] = pz; nc += 1
    if H > 1:
        # (x, y, z+h): project on X and on Y
        px = 0
        for p in range(n_others):
            if (pl[p, 0] + pl[p, 3] <= x and pl[p, 1] <= y < pl[p, 1] + pl[p, 4]
                    and pl[p, 2] <= z + h < pl[p, 2] + pl[p, 5]):
                v = pl[p, 0] + pl[p, 3]
                if v > px:
                    px = v
        cand[nc, 0] = px; cand[nc, 1] = y; cand[nc, 2] = z + h; nc += 1
        py = 0
        for p in range(n_others):
            if (pl[p, 1] + pl[p, 4] <= y and pl[p, 0] <= x < pl[p, 0] + pl[p, 3]
                    and pl[p, 2] <= z + h < pl[p, 2] + pl[p, 5]):
                v = pl[p, 1] + pl[p, 4]
                if v > py:
                    py = v
        cand[nc, 0] = x; cand[nc, 1] = py; cand[nc, 2] = z + h; nc += 1

        # ---- composed projections (same order as the Python backend)
        # 6: (x+w, py1, projZ)   from candidate 0
        a = cand[0, 0]; b = cand[0, 1]
        v0 = 0
        for p in range(n_others):
            if (pl[p, 2] + pl[p, 5] <= z and pl[p, 0] <= a < pl[p, 0] + pl[p, 3]
                    and pl[p, 1] <= b < pl[p, 1] + pl[p, 4]):
                v = pl[p, 2] + pl[p, 5]
                if v > v0:
                    v0 = v
        cand[nc, 0] = a; cand[nc, 1] = b; cand[nc, 2] = v0; nc += 1
        # 7: (x+w, projY, pz1)   from candidate 1
        a = cand[1, 0]; c = cand[1, 2]
        v0 = 0
        for p in range(n_others):
            if (pl[p, 1] + pl[p, 4] <= y and pl[p, 0] <= a < pl[p, 0] + pl[p, 3]
                    and pl[p, 2] <= c < pl[p, 2] + pl[p, 5]):
                v = pl[p, 1] + pl[p, 4]
                if v > v0:
                    v0 = v
        cand[nc, 0] = a; cand[nc, 1] = v0; cand[nc, 2] = c; nc += 1
        # 8: (px1, y+d, projZ)   from candidate 2
        a = cand[2, 0]; b = cand[2, 1]
        v0 = 0
        for p in range(n_others):
            if (pl[p, 2] + pl[p, 5] <= z and pl[p, 0] <= a < pl[p, 0] + pl[p, 3]
                    and pl[p, 1] <= b < pl[p, 1] + pl[p, 4]):
                v = pl[p, 2] + pl[p, 5]
                if v > v0:
                    v0 = v
        cand[nc, 0] = a; cand[nc, 1] = b; cand[nc, 2] = v0; nc += 1
        # 9: (projX, y+d, pz2)   from candidate 3
        b = cand[3, 1]; c = cand[3, 2]
        v0 = 0
        for p in range(n_others):
            if (pl[p, 0] + pl[p, 3] <= x and pl[p, 1] <= b < pl[p, 1] + pl[p, 4]
                    and pl[p, 2] <= c < pl[p, 2] + pl[p, 5]):
                v = pl[p, 0] + pl[p, 3]
                if v > v0:
                    v0 = v
        cand[nc, 0] = v0; cand[nc, 1] = b; cand[nc, 2] = c; nc += 1
        # 10: (px2, projY, z+h)  from candidate 4
        a = cand[4, 0]; c = cand[4, 2]
        v0 = 0
        for p in range(n_others):
            if (pl[p, 1] + pl[p, 4] <= y and pl[p, 0] <= a < pl[p, 0] + pl[p, 3]
                    and pl[p, 2] <= c < pl[p, 2] + pl[p, 5]):
                v = pl[p, 1] + pl[p, 4]
                if v > v0:
                    v0 = v
        cand[nc, 0] = a; cand[nc, 1] = v0; cand[nc, 2] = c; nc += 1
        # 11: (projX, py2, z+h)  from candidate 5
        b = cand[5, 1]; c = cand[5, 2]
        v0 = 0
        for p in range(n_others):
            if (pl[p, 0] + pl[p, 3] <= x and pl[p, 1] <= b < pl[p, 1] + pl[p, 4]
                    and pl[p, 2] <= c < pl[p, 2] + pl[p, 5]):
                v = pl[p, 0] + pl[p, 3]
                if v > v0:
                    v0 = v
        cand[nc, 0] = v0; cand[nc, 1] = b; cand[nc, 2] = c; nc += 1

        # ---- fixed-point continuations of the composed candidates
        # (gated: tested with no measurable gain over composed-only)
        if ext == 1:
            spec_idx = (6, 7, 8, 9, 10, 11)
            spec_a1 = (1, 2, 0, 2, 0, 1)   # 0=x, 1=y, 2=z
            spec_a2 = (2, 1, 2, 0, 1, 0)
            for s in range(6):
                cx = cand[spec_idx[s], 0]
                cy = cand[spec_idx[s], 1]
                cz = cand[spec_idx[s], 2]
                emitted = 0
                stall = 0
                t = 0
                while emitted < 2 and stall < 2:
                    a = spec_a1[s] if t % 2 == 0 else spec_a2[s]
                    t += 1
                    moved = False
                    if a == 0:
                        nv = _pjx(pl, n_others, cx, cy, cz)
                        if nv < cx:
                            cx = nv; moved = True
                    elif a == 1:
                        nv = _pjy(pl, n_others, cx, cy, cz)
                        if nv < cy:
                            cy = nv; moved = True
                    else:
                        nv = _pjz(pl, n_others, cx, cy, cz)
                        if nv < cz:
                            cz = nv; moved = True
                    if moved:
                        cand[nc, 0] = cx; cand[nc, 1] = cy; cand[nc, 2] = cz
                        nc += 1
                        emitted += 1
                        stall = 0
                    else:
                        stall += 1

            # ---- diagonal-edge candidates (single orthogonal axis each)
            cand[nc, 0] = x + w; cand[nc, 1] = y + d
            cand[nc, 2] = _pjz(pl, n_others, x + w, y + d, z); nc += 1
            cand[nc, 0] = x + w; cand[nc, 1] = _pjy(pl, n_others, x + w, y, z + h)
            cand[nc, 2] = z + h; nc += 1
            cand[nc, 0] = _pjx(pl, n_others, x, y + d, z + h)
            cand[nc, 1] = y + d; cand[nc, 2] = z + h; nc += 1

    # ---- drop EPs covered by the new item (compact in place)
    m = 0
    for e in range(n_eps):
        inside = (x <= eps[e, 0] < x + w and y <= eps[e, 1] < y + d
                  and z <= eps[e, 2] < z + h)
        if not inside:
            if m != e:
                for c in range(6):
                    eps[m, c] = eps[e, c]
            m += 1
    n_eps = m

    # ---- update RS of survivors against the new item
    for e in range(n_eps):
        ex = eps[e, 0]; ey = eps[e, 1]; ez = eps[e, 2]
        if x >= ex and y <= ey < y + d and z <= ez < z + h:
            v = x - ex
            if v < eps[e, 3]:
                eps[e, 3] = v
        if y >= ey and x <= ex < x + w and z <= ez < z + h:
            v = y - ey
            if v < eps[e, 4]:
                eps[e, 4] = v
        if z >= ez and x <= ex < x + w and y <= ey < y + d:
            v = z - ez
            if v < eps[e, 5]:
                eps[e, 5] = v

    # ---- append new EPs (dedup, bounds, RS vs all placements)
    for c in range(nc):
        nx = cand[c, 0]; ny = cand[c, 1]; nz = cand[c, 2]
        if nx >= W or ny >= D or nz >= H:
            continue
        dup = False
        for e in range(n_eps):
            if eps[e, 0] == nx and eps[e, 1] == ny and eps[e, 2] == nz:
                dup = True
                break
        if dup:
            continue
        rsx = W - nx; rsy = D - ny; rsz = H - nz
        for p in range(n_pl):
            if (pl[p, 0] >= nx and pl[p, 1] <= ny < pl[p, 1] + pl[p, 4]
                    and pl[p, 2] <= nz < pl[p, 2] + pl[p, 5]):
                v = pl[p, 0] - nx
                if v < rsx:
                    rsx = v
            if (pl[p, 1] >= ny and pl[p, 0] <= nx < pl[p, 0] + pl[p, 3]
                    and pl[p, 2] <= nz < pl[p, 2] + pl[p, 5]):
                v = pl[p, 1] - ny
                if v < rsy:
                    rsy = v
            if (pl[p, 2] >= nz and pl[p, 0] <= nx < pl[p, 0] + pl[p, 3]
                    and pl[p, 1] <= ny < pl[p, 1] + pl[p, 4]):
                v = pl[p, 2] - nz
                if v < rsz:
                    rsz = v
        eps[n_eps, 0] = nx; eps[n_eps, 1] = ny; eps[n_eps, 2] = nz
        eps[n_eps, 3] = rsx; eps[n_eps, 4] = rsy; eps[n_eps, 5] = rsz
        n_eps += 1

    return n_eps


# ======================================================================
class FastGreedyState:
    """Drop-in replacement for gasp.greedy.GreedyState (numba backend)."""

    def __init__(self, knapsack: Knapsack, criterion: str = "RS",
                 allow_rotation: bool = False, _alloc: bool = True):
        self.knapsack = knapsack
        self.criterion = criterion
        self.allow_rotation = allow_rotation
        self.n_processed = 0
        if _alloc:
            self.n_pl = 0
            self.n_eps = 1
            self.pl = np.zeros((8, 6), np.int64)
            self.eps = np.zeros((16, 6), np.int64)
            self.eps[0] = (0, 0, 0, knapsack.W, knapsack.D, knapsack.H)
            self.placed_items: List[tuple] = []   # (Item, x,y,z,w,d,h)
            self._env_w = 0
            self._env_d = 0

    # ------------------------------------------------------------------
    def clone(self) -> "FastGreedyState":
        st = FastGreedyState(self.knapsack, self.criterion,
                             self.allow_rotation, _alloc=False)
        st.n_pl = self.n_pl
        st.n_eps = self.n_eps
        st.pl = self.pl.copy()
        st.eps = self.eps.copy()
        st.placed_items = list(self.placed_items)
        st._env_w = self._env_w
        st._env_d = self._env_d
        st.n_processed = self.n_processed
        return st

    # ------------------------------------------------------------------
    def _grow(self):
        if self.n_pl + 1 >= self.pl.shape[0]:
            self.pl = np.vstack((self.pl, np.zeros_like(self.pl)))
        if self.n_eps + 32 >= self.eps.shape[0]:
            self.eps = np.vstack((self.eps, np.zeros_like(self.eps)))

    def place(self, item: Item) -> bool:
        self._grow()
        rots = np.array(item.rotations(self.allow_rotation,
                                       self.knapsack.is_3d), np.int64)
        r, e = _best_placement(self.eps, self.n_eps, self.pl, self.n_pl,
                               rots, self.knapsack.W, self.knapsack.D,
                               self.knapsack.H, CRIT_ID[self.criterion],
                               self._env_w, self._env_d)
        self.n_processed += 1
        if r < 0:
            return False
        w, d, h = int(rots[r, 0]), int(rots[r, 1]), int(rots[r, 2])
        x, y, z = (int(self.eps[e, 0]), int(self.eps[e, 1]),
                   int(self.eps[e, 2]))
        self.pl[self.n_pl] = (x, y, z, w, d, h)
        self.n_pl += 1
        self.placed_items.append((item, x, y, z, w, d, h))
        if x + w > self._env_w:
            self._env_w = x + w
        if y + d > self._env_d:
            self._env_d = y + d
        self.n_eps = _add_item(self.eps, self.n_eps, self.pl, self.n_pl,
                               x, y, z, w, d, h, self.knapsack.W,
                               self.knapsack.D, self.knapsack.H,
                               1 if _epmod.EXTENDED_PROJECTIONS else 0)
        return True

    # ------------------------------------------------------------------
    def run(self, items: List[Item]) -> Packing:
        for item in items:
            self.place(item)
        return self.packing

    @property
    def packing(self) -> Packing:
        return Packing(self.knapsack,
                       [Placement(it, x, y, z, w, d, h)
                        for (it, x, y, z, w, d, h) in self.placed_items])

    @property
    def residual_eps(self) -> List[ExtremePoint]:
        return [ExtremePoint(*map(int, self.eps[e]))
                for e in range(self.n_eps)]


def warmup():
    """Trigger JIT compilation on a tiny dummy problem."""
    ks = Knapsack(4, 4, 4)
    st = FastGreedyState(ks, "RS", True)
    st.place(Item(0, 2, 2, 2, 1.0))
    st.place(Item(1, 2, 2, 2, 1.0))
    for crit in ("MP", "LEV", "FF", "TP"):
        s2 = FastGreedyState(ks, crit, False)
        s2.place(Item(0, 2, 2, 2, 1.0))
