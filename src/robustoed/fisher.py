from __future__ import annotations

import numpy as np
from .types import Design


Array = np.ndarray


def information_matrix(model, design: Design, theta: Array) -> Array:
    """Compute the Fisher information matrix for a scalar-response model.

    The Fisher information matrix for an approximate design ξ with support points
    {x_i} and weights {w_i} is computed as:

        M(ξ; θ) = Σ_i w_i f(x_i, θ) f(x_i, θ)^T

    where f(x, θ) is the gradient of the mean function η with respect to θ.

    Parameters
    ----------
    model :
        Model object providing `jacobian(x, theta) -> array of shape (p,)`.
    design :
        Approximate design with `points` shape (m, d) and `weights` shape (m,).
    theta :
        Parameter vector with shape (p,).

    Returns
    -------
    M :
        Fisher information matrix with shape (p, p).

    Notes
    -----
    This implementation assumes:
    - scalar response η(x, θ)
    - homoscedastic errors (or that f already includes the proper weighting).
    """
    pts = np.asarray(design.points, dtype=float)
    w = np.asarray(design.weights, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    # Evaluate first to get p
    f0 = model.jacobian(pts[0], theta)
    p = f0.shape[0]
    M = np.zeros((p, p), dtype=float)

    for xi, wi in zip(pts, w):
        f = model.jacobian(xi, theta).reshape(p, 1)
        M += wi * (f @ f.T)
    return M