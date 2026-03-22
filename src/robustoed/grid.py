from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple
import numpy as np
import sympy as sp


def make_grid_equidistant(
    spec: Dict[sp.Symbol, Tuple[float, float, int]]
) -> List[Dict[sp.Symbol, float]]:
    """Create a Cartesian grid of parameter scenarios (Ω).

    Parameters
    ----------
    spec:
        Mapping {param_symbol: (low, high, n_points)}. For each parameter,
        create n_points equally spaced values in [low, high], then take the
        Cartesian product across parameters.

    Returns
    -------
    grid:
        List of scenarios. Each scenario is {param_symbol: value}.
    """
    symbols = list(spec.keys())
    axes = []
    for s in symbols:
        low, high, n = spec[s]
        if n < 2:
            raise ValueError(f"n_points must be >= 2 for {s}. Got {n}.")
        axes.append(np.linspace(low, high, int(n), dtype=float))

    grid = []
    for values in product(*axes):
        grid.append({s: float(v) for s, v in zip(symbols, values)})
    return grid