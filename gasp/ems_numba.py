"""Numba-accelerated dominance elimination for maximal spaces.

Profiling the faithful GRASP on strongly heterogeneous instances (BR15)
showed that ~58% of a construction's time is spent in the O(n^2)
dominance check that prunes contained empty spaces. That check is pure
integer arithmetic over a list of (x,y,z,x2,y2,z2) tuples -- an ideal
numba target. This module provides a compiled kernel that takes a
(n,6) int array of spaces (already filtered by min dimension and sorted
by volume descending) and returns a boolean mask of the survivors.

If numba is unavailable the caller falls back to the pure-Python loop.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco if not args else args[0]


@njit(cache=True)
def _keep_mask(arr):
    """arr: (n,6) int array, sorted by volume desc. Returns a boolean
    array; entry i is True if space i is not contained in any earlier
    kept space (lower index = larger or equal volume)."""
    n = arr.shape[0]
    keep = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if not keep[i]:
            continue
        ax0 = arr[i, 0]; ay0 = arr[i, 1]; az0 = arr[i, 2]
        ax1 = arr[i, 3]; ay1 = arr[i, 4]; az1 = arr[i, 5]
        # compare against earlier KEPT spaces (j < i): is i inside j?
        for j in range(i):
            if not keep[j]:
                continue
            if (arr[j, 0] <= ax0 and arr[j, 1] <= ay0 and arr[j, 2] <= az0
                    and arr[j, 3] >= ax1 and arr[j, 4] >= ay1
                    and arr[j, 5] >= az1):
                keep[i] = False
                break
    return keep


def warmup_dominance():
    """Trigger JIT compilation on a tiny dummy array."""
    if _HAVE_NUMBA:
        _keep_mask(np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64))
