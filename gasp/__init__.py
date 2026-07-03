"""GASP - Greedy Adaptive Search Procedure for Multi-Dimensional
Knapsack problems (2D-KP, 3D-KP, 3D-CLP).

Python implementation of the metaheuristic described in:
G. Perboli, "An Efficient Metaheuristic for Multi-Dimensional
Knapsack Problems".
"""

from .adaptive import AdaptiveGASP
from .best_known import BR_MEAN_VOLUME, OPTIMA, optimum
from .gasp import GASP, GASPParams, GASPResult
from .geometry import Item, Knapsack, Packing, Placement
from .greedy import ep_kph
from .instances import generate_2d, generate_3d, knapsack_upper_bound
from .readers import load_set

__all__ = [
    "GASP", "AdaptiveGASP", "GASPParams", "GASPResult",
    "Item", "Knapsack", "Packing", "Placement",
    "ep_kph",
    "generate_2d", "generate_3d", "knapsack_upper_bound",
    "load_set", "optimum", "OPTIMA", "BR_MEAN_VOLUME",
]
__version__ = "0.1.0"
