"""Optimal and best-known solution values for the bundled benchmarks.

The values below are numerical results widely reported in the 2D-KP
literature (Beasley 1985a/b; Fekete & Schepers 2004; Hadjiconstantinou
& Iori 2007; Alvarez-Valdes et al. 2007; Bortfeldt & Winter 2009).

Conventions
-----------
- A value marked ``proven=True`` is a proven optimum.
- ``proven=False`` marks the best-known value (e.g. gcut13, for which
  no proven optimum is available).
- ``None`` means the value is not bundled here; the 1D-knapsack upper
  bound (gasp.instances.knapsack_upper_bound) can be used instead, as
  done for the ngcutfs sets and the rotation variants in the paper.
- For the BR sets (thpack1-7) the literature reports mean volume
  utilisation per set rather than per-instance optima; the reference
  means of Table 4 of the paper are given in BR_MEAN_VOLUME.

NOTE: values flagged ``verify=True`` are commonly cited but should be
double-checked against the original tables before publication use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BestKnown:
    value: float
    proven: bool = True
    verify: bool = False


# ---------------------------------------------------------------- 2D-KP
OPTIMA: Dict[str, BestKnown] = {
    # ngcut01-12 (Beasley 1985b), proven optima
    "ngcut01": BestKnown(164),  "ngcut02": BestKnown(230),
    "ngcut03": BestKnown(247),  "ngcut04": BestKnown(268),
    "ngcut05": BestKnown(358),  "ngcut06": BestKnown(289),
    "ngcut07": BestKnown(430),  "ngcut08": BestKnown(834),
    "ngcut09": BestKnown(924),  "ngcut10": BestKnown(1452),
    "ngcut11": BestKnown(1688), "ngcut12": BestKnown(1865),

    # cgcut1-3 (Christofides & Whitlock 1977), proven optima
    "cgcut1": BestKnown(244), "cgcut2": BestKnown(2892),
    "cgcut3": BestKnown(1860),

    # gcut1-12 (Beasley 1985a), proven optima; gcut13 best known only
    "gcut1": BestKnown(48368),  "gcut2": BestKnown(59798),
    "gcut3": BestKnown(61275),  "gcut4": BestKnown(61380),
    "gcut5": BestKnown(195582), "gcut6": BestKnown(236305),
    "gcut7": BestKnown(240143), "gcut8": BestKnown(245758),
    "gcut9": BestKnown(939600), "gcut10": BestKnown(937349),
    "gcut11": BestKnown(969709), "gcut12": BestKnown(979521),
    "gcut13": BestKnown(8763696, proven=False, verify=True),

    # okp1-5 (Fekete & Schepers 2004), proven optima
    "okp1": BestKnown(27718, verify=True),
    "okp2": BestKnown(22502, verify=True),
    "okp3": BestKnown(24019, verify=True),
    "okp4": BestKnown(32893, verify=True),
    "okp5": BestKnown(27923, verify=True),

    # wang20 (Wang 1983) - the data file is not in the ESICUP mirror;
    # value kept for reference
    "wang20": BestKnown(2726, verify=True),

    # hccut1-5 (= problems 9, 3, 11, 8, 12 of Hadjiconstantinou &
    # Christofides 1995): per-instance optima are reported in that
    # paper; not bundled here.

    # ep3 cube instances, n=20: optima CERTIFIED in this project
    # (50-fill: chain-cut bound coincides with the GASP solution;
    # 90-fill: CP-SAT proof, see gasp/cp_slave.py)
    "ep3-20-C-C-50": BestKnown(65308),
    "ep3-20-C-R-50": BestKnown(62364),
    "ep3-20-C-C-90": BestKnown(80124),
    "ep3-20-C-R-90": BestKnown(66844),
}

# ------------------------------------------------------------ 3D-CLP
# Mean volume utilisation (%) per BR set: state-of-the-art reference
# values from Table 4 of the GASP paper (best column = VNS_PAO for
# BR1-7), useful to benchmark mean fill rates.
BR_MEAN_VOLUME: Dict[str, float] = {
    "thpack1": 94.93, "thpack2": 95.16, "thpack3": 94.99,
    "thpack4": 94.71, "thpack5": 94.33, "thpack6": 94.04,
    "thpack7": 93.53,
}


def optimum(instance_name: str) -> Optional[BestKnown]:
    """Return the BestKnown record for an instance, or None."""
    return OPTIMA.get(instance_name)
