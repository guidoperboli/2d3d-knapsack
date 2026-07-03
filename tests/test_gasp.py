"""Sanity tests for the GASP implementation.

Run with:  python -m pytest tests/  (or simply python tests/test_gasp.py)
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasp import (GASP, GASPParams, Item, Knapsack, ep_kph,
                  generate_2d, generate_3d)
from gasp.sorting import SORTING_RULES


def assert_valid(packing):
    ks = packing.knapsack
    for p in packing.placements:
        assert p.x >= 0 and p.y >= 0 and p.z >= 0
        assert p.x2 <= ks.W and p.y2 <= ks.D and p.z2 <= ks.H, "out of knapsack"
    for a, b in combinations(packing.placements, 2):
        assert not a.overlaps(b), f"overlap between items {a.item.idx} and {b.item.idx}"
    ids = [p.item.idx for p in packing.placements]
    assert len(ids) == len(set(ids)), "item loaded twice"


def test_greedy_2d_valid():
    items, ks = generate_2d(n=40, geom_class="D", seed=1)
    for rule in SORTING_RULES:
        packing = ep_kph(rule(items, ks), ks, criterion="RS")
        assert_valid(packing)


def test_greedy_3d_valid_all_merits():
    items, ks = generate_3d(n=30, geom_class="D", seed=2)
    for crit in ("RS", "MP", "LEV", "FF"):
        packing = ep_kph(items, ks, criterion=crit, allow_rotation=True)
        assert_valid(packing)
        assert packing.profit > 0


def test_tiny_exact():
    # Two 5x5 items fit exactly in a 10x5 knapsack; a 6x6 cannot.
    items = [Item(0, 5, 5, 1, 10), Item(1, 5, 5, 1, 10), Item(2, 6, 6, 1, 100)]
    ks = Knapsack(10, 5, 1)
    res = GASP(items, ks, GASPParams(time_limit=1.0, seed=0,
                                     known_optimum=20)).run()
    assert res.best_profit == 20
    assert_valid(res.best_packing)


def test_gasp_not_worse_than_initial():
    items, ks = generate_3d(n=40, geom_class="U", seed=5)
    solver = GASP(items, ks, GASPParams(time_limit=3.0, seed=0,
                                        allow_rotation=True))
    initial = solver.initial_solution()
    res = solver.run()
    assert res.best_profit >= initial.profit
    assert_valid(res.best_packing)


def test_readers_and_optima():
    from gasp.readers import load_set
    from gasp.best_known import optimum

    ng = load_set("ngcut")
    assert len(ng) == 12 and ng[0][0] == "ngcut01"
    assert len(load_set("gcut")) == 13
    assert len(load_set("cgcut")) == 3
    assert len(load_set("okp")) == 5
    br = load_set("thpack1")
    assert len(br) == 100
    # CLP: profits equal volumes
    assert all(it.profit == it.volume for it in br[0][1])

    # GASP reaches the proven optimum on an easy classic instance
    name, items, ks = ng[0]
    opt = optimum(name).value
    res = GASP(items, ks, GASPParams(time_limit=5.0, seed=0,
                                     known_optimum=opt)).run()
    assert res.best_profit == opt
    assert_valid(res.best_packing)


if __name__ == "__main__":
    test_greedy_2d_valid()
    test_greedy_3d_valid_all_merits()
    test_tiny_exact()
    test_gasp_not_worse_than_initial()
    test_readers_and_optima()
    print("All tests passed.")
