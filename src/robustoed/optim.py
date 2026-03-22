from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import dual_annealing

from .types import Design
from .fisher import information_matrix
from .regularize import AutoRegularization, regularize_if_needed
from .criterion_d import logdet_psd


Array = np.ndarray
Bounds = List[Tuple[float, float]]


@dataclass
class WynnFedorovOptions:
    """Options for the Wynn–Fedorov style local D-optimal design algorithm.

    Parameters
    ----------
    max_iter :
        Maximum number of outer iterations.
    tol :
        Stopping tolerance. Used both for equivalence-theorem gap and relative
        logdet improvements.
    rounding_digits :
        Number of decimal digits used to round support points.
    merge_tol :
        Distance threshold used to merge nearby support points after each update.
        If None, a sensible default based on rounding_digits is used.
    min_weight :
        Prune support points with weight < min_weight.
    weight_digits :
        Round weights to this number of decimals for stable, readable output.
    seed :
        Random seed for the global optimizer.
    regularization :
        Auto-regularization configuration for near-singular information matrices.
    da_maxiter :
        Max iterations for SciPy dual_annealing used to maximize the sensitivity
        function.

    Notes
    -----
    This is a practical implementation intended for nonlinear models and bounded
    design spaces. It can be extended to other optimality criteria by swapping the
    sensitivity function and stopping rule.
    """
    max_iter: int = 200
    tol: float = 1e-6
    rounding_digits: int = 2
    seed: Optional[int] = None
    regularization: AutoRegularization = AutoRegularization()
    merge_tol: float | None = None
    weight_digits: int = 6       # para presentación/reproducibilidad
    min_weight: float = 1e-4     # elimina pesos menores a esto (ajustable)
    # dual annealing options (keep minimal for MVP)
    da_maxiter: int = 200


@dataclass
class WynnFedorovResult:
    design: Design
    n_iter: int
    history_logdet: List[float]
    stop_reason: str


def _round_points(points: Array, digits: int) -> Array:
    return np.round(points.astype(float), decimals=digits)


def _aggregate_design(points: Array, weights: Array, *, digits: int, merge_tol: float | None) -> Design:
    """Aggregate duplicate/close points after rounding.

    If merge_tol is None: only exact matches after rounding are merged.
    If merge_tol is provided: points within merge_tol are merged (in Euclidean norm).
    """
    pts = _round_points(points, digits)
    w = np.asarray(weights, dtype=float)

    # sort lexicographically for stable merging
    idx = np.lexsort(pts.T[::-1])
    pts = pts[idx]
    w = w[idx]

    if merge_tol is None:
        # exact merge by key
        keys = [tuple(row.tolist()) for row in pts]
        acc: Dict[Tuple[float, ...], float] = {}
        order: List[Tuple[float, ...]] = []
        for k, wi in zip(keys, w):
            if k not in acc:
                acc[k] = float(wi)
                order.append(k)
            else:
                acc[k] += float(wi)
        out_points = np.array(order, dtype=float)
        out_weights = np.array([acc[k] for k in order], dtype=float)
        out_weights = out_weights / out_weights.sum()
        return Design(out_points, out_weights)

    # tolerance-based merge
    merged_pts: List[Array] = []
    merged_w: List[float] = []

    cur_pt = pts[0].copy()
    cur_w = float(w[0])

    for pt, wi in zip(pts[1:], w[1:]):
        dist = float(np.linalg.norm(pt - cur_pt))
        if dist <= merge_tol:
            # merge: keep weighted average location (more stable than picking one)
            new_w = cur_w + float(wi)
            cur_pt = (cur_w * cur_pt + float(wi) * pt) / new_w
            cur_w = new_w
        else:
            merged_pts.append(cur_pt)
            merged_w.append(cur_w)
            cur_pt = pt.copy()
            cur_w = float(wi)

    merged_pts.append(cur_pt)
    merged_w.append(cur_w)

    out_points = np.vstack(merged_pts)
    out_points = _round_points(out_points, digits)   

    out_weights = np.array(merged_w, dtype=float)
    out_weights = out_weights / out_weights.sum()
    return Design(out_points, out_weights)


def _prune_and_round_weights(design: Design, *, min_weight: float, weight_digits: int) -> Design:
    pts = np.asarray(design.points, dtype=float)
    w = np.asarray(design.weights, dtype=float)

    keep = w >= float(min_weight)
    if not np.any(keep):
        # si por alguna razón todos quedan fuera, conserva el máximo
        keep = np.zeros_like(w, dtype=bool)
        keep[np.argmax(w)] = True

    pts = pts[keep]
    w = w[keep]

    # renormaliza
    w = w / w.sum()

    # redondea pesos y renormaliza otra vez para que sumen 1 exacto
    w = np.round(w, decimals=weight_digits)
    s = w.sum()
    if s <= 0:
        # fallback: no redondear si algo raro
        w = w.astype(float)
        w = w / w.sum()
    else:
        w = w / s

    return Design(pts, w)


def d_sensitivity_value(
    model,
    design: Design,
    theta: Array,
    x: Array,
    *,
    reg: AutoRegularization,
) -> float:
    """D-opt 'variance function' f^T M^{-1} f (scalar)."""
    M = information_matrix(model, design, theta)
    M, _ = regularize_if_needed(M, reg=reg)
    f = model.jacobian(np.asarray(x, dtype=float), theta).reshape(-1, 1)
    # solve instead of inv
    v = float((f.T @ np.linalg.solve(M, f))[0, 0])
    return v


def wynn_fedorov_d_opt(
    model,
    theta: Array,
    bounds: Bounds,
    *,
    init_design: Optional[Design] = None,
    n_support: int = 6,
    options: WynnFedorovOptions = WynnFedorovOptions(),
) -> WynnFedorovResult:
    """Compute a locally D-optimal approximate design (bounded design space).

    This routine iteratively updates an approximate design by repeatedly:
    1) maximizing the D-sensitivity (variance function) over the design space,
    2) adding the maximizer to the support with a stable step size,
    3) aggregating nearby points and pruning tiny weights.

    Parameters
    ----------
    model :
        Model object implementing `jacobian(x, theta) -> (p,)`.
    theta :
        Parameter vector of shape (p,).
    bounds :
        Box constraints for the design space as a list of (low, high) pairs of
        length d.
    init_design :
        Optional initial design. If None, starts from random support points.
    n_support :
        Number of support points used for random initialization if init_design is None.
    options :
        Algorithm configuration.

    Returns
    -------
    result :
        WynnFedorovResult containing the final design and diagnostics.

    Notes
    -----
    For D-optimal designs, the equivalence theorem implies that at the optimum
    max_x f(x)^T M(ξ)^{-1} f(x) = p, where p is the number of parameters. We use
    this as a stopping criterion (within tolerance).
    """
    rng = np.random.default_rng(options.seed)
    d = len(bounds)

    if init_design is None:
        pts = np.array(
            [rng.uniform(low, high, size=d) for (low, high) in [b for b in bounds] for _ in []],
            dtype=float,
        )
        # Above line is awkward; do it clearly:
        pts = np.vstack([rng.uniform(bounds[j][0], bounds[j][1], size=n_support) for j in range(d)]).T
        w = np.ones(n_support, dtype=float) / n_support
        design = Design(_round_points(pts, options.rounding_digits), w)
    else:
        design = init_design

    history: List[float] = []
    stop_reason = "max_iter"

    p = model.p if hasattr(model, "p") else len(theta)

    for it in range(1, options.max_iter + 1):
        # Compute logdet of current information matrix (regularized)
        M = information_matrix(model, design, theta)
        M_reg, _ = regularize_if_needed(M, reg=options.regularization)
        try:
            ld = logdet_psd(M_reg)
        except np.linalg.LinAlgError:
            # If still not PD, push a bit more regularization
            M_reg = M_reg + 1e-8 * np.eye(M_reg.shape[0])
            ld = logdet_psd(M_reg)
        history.append(ld)

        # Objective for dual annealing: maximize sensitivity => minimize negative
        def obj(x_flat: Array) -> float:
            x_vec = np.asarray(x_flat, dtype=float).reshape(d)
            v = d_sensitivity_value(model, design, theta, x_vec, reg=options.regularization)
            return -v

        da = dual_annealing(
            obj,
            bounds=bounds,
            seed=options.seed,
            maxiter=options.da_maxiter,
        )
        x_star = np.asarray(da.x, dtype=float).reshape(d)
        x_star = np.round(x_star, options.rounding_digits)

        # Sensitivity value at maximizer
        v_star = d_sensitivity_value(model, design, theta, x_star, reg=options.regularization)

        # D-opt equivalence theorem: for approximate designs, max f^T M^{-1} f == p at optimum
        gap = v_star - p
        if gap <= options.tol:
            stop_reason = "tol"
            # still add the point in a stable way? Usually we can stop directly.
            break

        # Update: add point with step a_n = gap / (p*(gap + 1)) (common stable choice)
        # This keeps weights positive and tends to converge.
        a_n = gap / (p * (gap + 1.0))

        X = np.vstack([design.points, x_star.reshape(1, d)])
        w_old = (1.0 - a_n) * design.weights
        w_new = np.append(w_old, a_n)

        merge_tol = options.merge_tol
        if merge_tol is None:
            # sensible default tied to rounding resolution
            merge_tol = 10 ** (-options.rounding_digits)
        design = _aggregate_design(X, w_new, digits=options.rounding_digits, merge_tol=merge_tol)

        design = _prune_and_round_weights(
            design,
            min_weight=options.min_weight,
            weight_digits=options.weight_digits,
        )

        # optional convergence check on logdet improvement
        if it >= 5:
            rel = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12)
            if rel < options.tol:
                stop_reason = "logdet_rel"
                break

        design = _prune_and_round_weights(
            design,
            min_weight=options.min_weight,
            weight_digits=options.weight_digits,
        )

    return WynnFedorovResult(design=design, n_iter=len(history), history_logdet=history, stop_reason=stop_reason)