"""Readers for the benchmark instances bundled in ``data/``.

All sets come from the OR-Library (J.E. Beasley) as redistributed by the
ESICUP datasets collection (https://github.com/ESICUP/datasets).

2D sets (Section 4.1.1 of the paper)
------------------------------------
gcut1-13   Beasley (1985a), one problem per file:
               m / L W / per piece: l w value
cgcut1-3   Christofides & Whitlock (1977), one problem per file:
               m / L W / per piece: l w Q value     (Q = max copies)
okp1-5     Fekete & Schepers (2004), same format as cgcut
hccut1-5   Hadjiconstantinou & Christofides (1995), same format as cgcut
               (files 1..5 are problems 9, 3, 11, 8, 12 of the paper)
ngcutap    Beasley (2004) "assorted problems": 21 problems in one file,
               problems 1-12 being ngcut01-12 (Beasley, 1985b):
               P / for each: m / L W / per piece: l w min max value
ngcutcon   same multi-problem format (constrained variants)
ngcutfs1-3 Beasley (2004) large random sets, Type I/II/III,
               210 problems per file, same multi-problem format

3D set (Section 4.1.4)
----------------------
thpack1-7  Bischoff & Ratcliff (1995) BR1-BR7 container loading sets,
           100 problems per file:
               P / for each: id seed / L W H / n /
               per box type: i l xv w yv h zv count
           (xv/yv/zv are 0/1 vertical-orientation permissions; the pure
           3D-CLP with free rotation ignores them, profits = volumes)

Every reader returns a list of ``(name, items, knapsack)`` tuples.
Pieces with multiple copies (Q > 1) are expanded into Q individual
items, as customary in the 2D-KP literature.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .geometry import Item, Knapsack

Instance = Tuple[str, List[Item], Knapsack]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _tokens(path) -> List[int]:
    return [int(t) for t in Path(path).read_text().split()]


# ----------------------------------------------------------------------
# Single-problem files: gcut (no copies), cgcut/okp/hccut (with copies)
# ----------------------------------------------------------------------
def read_gcut(path) -> Instance:
    t = _tokens(path)
    m, L, W = t[0], t[1], t[2]
    items, idx, pos = [], 0, 3
    for _ in range(m):
        l, w, v = t[pos], t[pos + 1], t[pos + 2]
        pos += 3
        items.append(Item(idx, l, w, 1, profit=v))
        idx += 1
    return Path(path).stem, items, Knapsack(L, W, 1)


def read_constrained_2d(path) -> Instance:
    """cgcut / okp / hccut format: piece-type records ``l w Q v``.

    The first token is the number of piece types in cgcut/hccut but the
    total number of pieces (sum of the copies) in okp, so the parser
    simply consumes 4-token records until the file is exhausted.
    """
    t = _tokens(path)
    L, W = t[1], t[2]
    items, idx, pos = [], 0, 3
    while pos + 3 < len(t):
        l, w, q, v = t[pos], t[pos + 1], t[pos + 2], t[pos + 3]
        pos += 4
        for _ in range(max(1, q)):
            items.append(Item(idx, l, w, 1, profit=v))
            idx += 1
    return Path(path).stem, items, Knapsack(L, W, 1)


# ----------------------------------------------------------------------
# Multi-problem files: ngcutap, ngcutcon, ngcutfs1-3
# ----------------------------------------------------------------------
def read_ngcut_multi(path, use_max_copies: bool = True) -> List[Instance]:
    t = _tokens(path)
    stem = Path(path).stem
    n_problems, pos = t[0], 1
    instances = []
    for p in range(1, n_problems + 1):
        m = t[pos]; pos += 1
        L, W = t[pos], t[pos + 1]; pos += 2
        items, idx = [], 0
        for _ in range(m):
            l, w, qmin, qmax, v = t[pos:pos + 5]; pos += 5
            copies = qmax if use_max_copies else max(qmin, 1)
            for _ in range(max(1, copies)):
                items.append(Item(idx, l, w, 1, profit=v))
                idx += 1
        instances.append((f"{stem}-{p:03d}", items, Knapsack(L, W, 1)))
    return instances


# ----------------------------------------------------------------------
# 3D container loading: thpack1-7 (BR1-BR7)
# ----------------------------------------------------------------------
def read_thpack(path, respect_orientation: bool = False) -> List[Instance]:
    """BR sets: profits are set equal to volumes (3D-CLP).

    If respect_orientation is True, the per-dimension vertical-orientation
    flags (xv/yv/zv) are honoured, matching the constrained protocol used
    by the container-loading literature; otherwise rotation is free."""
    t = _tokens(path)
    stem = Path(path).stem
    n_problems, pos = t[0], 1
    instances = []
    for _ in range(n_problems):
        pid, _seed = t[pos], t[pos + 1]; pos += 2
        L, W, H = t[pos], t[pos + 1], t[pos + 2]; pos += 3
        n_types = t[pos]; pos += 1
        items, idx = [], 0
        for _ in range(n_types):
            (_i, l, xv, w, yv, h, zv, count) = t[pos:pos + 8]; pos += 8
            vflags = (bool(xv), bool(yv), bool(zv)) if respect_orientation else None
            for _ in range(count):
                items.append(Item(idx, l, w, h, profit=l * w * h,
                                  vflags=vflags))
                idx += 1
        instances.append((f"{stem}-{pid:03d}", items, Knapsack(L, W, H)))
    return instances


# ----------------------------------------------------------------------
# Egeblad-Pisinger 2D/3D-KP instances (.2kp / .3kp)
# ----------------------------------------------------------------------
def read_ep(path) -> Instance:
    """EP format: comma-separated lines.

    2D: header ``dim, W, H`` then ``rect, i, w, h, p, c``
    3D: header ``dim, W, H, D`` then ``box, i, w, h, d, p, c``
    where c is the number of copies of the piece (expanded here).
    Note the EP convention orders the knapsack sizes (W, H[, D]); we
    map them onto our (W, D, H) box with the third 2D dimension = 1.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()
             if ln.strip()]
    head = [t.strip() for t in lines[0].split(",")]
    dims = [int(t) for t in head[1:]]
    is_3d = len(dims) == 3
    if is_3d:
        W, H, D = dims
        ks = Knapsack(W, H, D)
    else:
        W, H = dims
        ks = Knapsack(W, H, 1)

    items, idx = [], 0
    for ln in lines[1:]:
        tok = [t.strip() for t in ln.split(",")]
        if tok[0].lower() not in ("rect", "box"):
            continue  # skip metadata/comment lines in some files
        vals = [int(t) for t in tok[1:] if t and t.lstrip("-").isdigit()]
        if not vals:
            continue
        if len(vals) == 4 and not is_3d:
            vals = vals  # i,w,h,p with c missing -> handled below
        if not is_3d and len(vals) == 5:
            _i, w, h, p, c = vals
        elif not is_3d and len(vals) == 4:
            w, h, p, c = vals  # letter index dropped
        elif is_3d and len(vals) == 6:
            _i, w, h, d, p, c = vals
        elif is_3d and len(vals) == 5:
            w, h, d, p, c = vals
        else:
            continue
        if is_3d:
            for _ in range(max(1, c)):
                items.append(Item(idx, w, h, d, profit=p))
                idx += 1
        else:
            for _ in range(max(1, c)):
                items.append(Item(idx, w, h, 1, profit=p))
                idx += 1
    return Path(path).stem, items, ks


# ----------------------------------------------------------------------
# Convenience loaders over the bundled data directory
# ----------------------------------------------------------------------
def load_set(name: str, data_dir: Path = DATA_DIR,
             respect_orientation: bool = False) -> List[Instance]:
    """Load a whole benchmark set by name.

    Valid names: 'gcut', 'cgcut', 'okp', 'hccut', 'ngcut' (the 12
    classic problems extracted from ngcutap), 'ngcutap', 'ngcutcon',
    'ngcutfs1', 'ngcutfs2', 'ngcutfs3', 'thpack1' ... 'thpack7',
    'br' (alias for all thpack files = BR1-BR15).

    respect_orientation (thpack/BR only): honour the per-box vertical
    orientation flags, matching the constrained literature protocol.
    """
    d2, d3 = data_dir / "2d", data_dir / "3d"

    if name in ("ep2", "ep_other"):
        return [read_ep(p) for p in sorted((d2 / name).glob("*.2kp"))]
    if name == "ep3":
        return [read_ep(p) for p in sorted((d3 / "ep3").glob("*.3kp"))]
    if name == "gcut":
        return [read_gcut(p) for p in sorted((d2 / "gcut").glob("gcut*.txt"),
                                             key=lambda p: int(p.stem[4:]))]
    if name in ("cgcut", "okp", "hccut"):
        return [read_constrained_2d(p)
                for p in sorted((d2 / name).glob(f"{name}*.txt"),
                                key=lambda p: int(p.stem[len(name):]))]
    if name == "ngcut":
        # ngcut01-12 are the first 12 problems of ngcutap
        all_ap = read_ngcut_multi(d2 / "ngcut" / "ngcutap.txt")[:12]
        return [(f"ngcut{int(n.split('-')[1]):02d}", items, ks)
                for (n, items, ks) in all_ap]
    if name in ("ngcutap", "ngcutcon"):
        return read_ngcut_multi(d2 / "ngcut" / f"{name}.txt")
    if name.startswith("ngcutfs"):
        return read_ngcut_multi(d2 / "ngcutfs" / f"{name}.txt")
    if name.startswith("thpack"):
        return read_thpack(d3 / "thpack" / f"{name}.txt",
                           respect_orientation=respect_orientation)
    if name == "br":
        out = []
        for k in range(1, 16):  # BR1-BR15 (thpack8-15 = Davies-Bischoff extended)
            out.extend(read_thpack(d3 / "thpack" / f"thpack{k}.txt", respect_orientation=respect_orientation))
        return out
    raise ValueError(f"Unknown benchmark set: {name}")
