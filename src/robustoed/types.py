from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Design:
    """Approximate experimental design.

    A design is represented by a finite set of support points in the design space
    and associated nonnegative weights summing to 1.

    Parameters
    ----------
    points :
        Support points with shape (m, d), where m is the number of support points
        and d is the dimension of the design space.
    weights :
        Nonnegative weights with shape (m,) that sum to 1.

    Notes
    -----
    In approximate design theory, the information matrix typically has the form

        M(ξ; θ) = Σ_i w_i f(x_i, θ) f(x_i, θ)^T

    where f(x, θ) is the gradient (Jacobian) of the mean function with respect to
    model parameters at design point x.
    """
    points: Array
    weights: Array

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        if pts.ndim != 2:
            raise ValueError(f"points must have shape (m, d). Got {pts.shape}.")
        if w.ndim != 1:
            raise ValueError(f"weights must have shape (m,). Got {w.shape}.")
        if pts.shape[0] != w.shape[0]:
            raise ValueError("points and weights must have same length (m).")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")
        s = float(w.sum())
        if not np.isfinite(s) or abs(s - 1.0) > 1e-8:
            raise ValueError(f"weights must sum to 1. Got sum={s}.")