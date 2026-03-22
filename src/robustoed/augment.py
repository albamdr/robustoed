from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import dual_annealing

from .types import Design
from .grid import make_grid_equidistant
from .fisher import information_matrix
from .optim import WynnFedorovOptions, wynn_fedorov_d_opt
from .regularize import AutoRegularization, regularize_if_needed
from .criterion_d import logdet_psd, d_efficiency_relative_shared_ridge

Array = np.ndarray
Bounds = List[Tuple[float, float]]
PointSet = Array  # shape (n_per_step, d)


@dataclass
class RobustAugmentResult:
    design_base: Design
    design_augm: Design
    x1_star: PointSet
    x2_star: PointSet
    delta: List[PointSet]
    meta: Dict[str, object]


def _theta_dict_to_vector(theta_symbols: Sequence[sp.Symbol], theta_dict: Dict[sp.Symbol, float]) -> Array:
    return np.array([float(theta_dict[s]) for s in theta_symbols], dtype=float)


def _round_points(points: Array, digits: int) -> Array:
    return np.round(np.asarray(points, dtype=float), decimals=digits)


def _canonicalize_set(X: Array, digits: int) -> Array:
    """Round and sort rows lexicographically so sets have canonical order."""
    X = _round_points(X, digits)
    idx = np.lexsort(X.T[::-1])
    return X[idx]


def _merge_close_support(points: Array, weights: Array, *, digits: int, merge_tol: float) -> Design:
    """
    Round, sort and merge nearby points, summing their weights.

    Two points are merged if their Euclidean distance is <= merge_tol
    after rounding.
    """
    pts = _round_points(points, digits)
    w = np.asarray(weights, dtype=float)

    if pts.ndim != 2:
        raise ValueError("points must have shape (m, d).")
    if w.ndim != 1 or w.shape[0] != pts.shape[0]:
        raise ValueError("weights must have shape (m,) matching points.")

    idx = np.lexsort(pts.T[::-1])
    pts = pts[idx]
    w = w[idx]

    merged_pts: List[Array] = []
    merged_w: List[float] = []

    cur_pt = pts[0].copy()
    cur_w = float(w[0])

    for pt, wi in zip(pts[1:], w[1:]):
        dist = float(np.linalg.norm(pt - cur_pt))
        if dist <= merge_tol:
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

    out_pts = _round_points(np.vstack(merged_pts), digits)
    out_w = np.asarray(merged_w, dtype=float)
    out_w = out_w / out_w.sum()
    return Design(out_pts, out_w)


def _equal_weight_same_support(design: Design) -> Design:
    pts = np.asarray(design.points, dtype=float)
    m = pts.shape[0]
    w = np.ones(m, dtype=float) / m
    return Design(pts, w)


def _augment_design(
    base: Design,
    X_sets: List[PointSet],
    *,
    alpha_mode: str,
    alpha: float,
    rounding_digits: int,
    merge_tol: float,
) -> Design:
    """
    Build augmented design and then merge nearby support points.

    fixed:
        (1-alpha) * base  +  alpha spread equally among all added points

    equal_weight:
        merge support first, then assign equal weights to final merged support.
    """
    if alpha_mode not in ("fixed", "equal_weight"):
        raise ValueError("alpha_mode must be 'fixed' or 'equal_weight'.")

    base_pts = np.asarray(base.points, dtype=float)
    base_w = np.asarray(base.weights, dtype=float)

    added_pts_list = []
    for X in X_sets:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            continue
        if X.ndim != 2:
            raise ValueError("Each X must have shape (n, d).")
        added_pts_list.append(X)

    if len(added_pts_list) == 0:
        return _merge_close_support(base_pts, base_w, digits=rounding_digits, merge_tol=merge_tol)

    all_added = np.vstack(added_pts_list)

    if alpha_mode == "equal_weight":
        pts = np.vstack([base_pts, all_added])
        w = np.ones(pts.shape[0], dtype=float)
        merged = _merge_close_support(pts, w, digits=rounding_digits, merge_tol=merge_tol)
        m = merged.points.shape[0]
        return Design(merged.points, np.ones(m, dtype=float) / m)

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1) when alpha_mode='fixed'.")

    n_added = all_added.shape[0]
    pts = np.vstack([base_pts, all_added])
    w = np.concatenate([
        (1.0 - alpha) * base_w,
        np.ones(n_added, dtype=float) * (alpha / n_added),
    ])
    return _merge_close_support(pts, w, digits=rounding_digits, merge_tol=merge_tol)


def _d_efficiency_formal(
    model,
    design_test: Design,
    design_opt: Design,
    theta_vec: Array,
    *,
    reg: AutoRegularization,
) -> float:
    M_test = information_matrix(model, design_test, theta_vec)
    M_opt = information_matrix(model, design_opt, theta_vec)
    p = theta_vec.shape[0]

    tr = float(np.trace(M_opt))
    if (not np.isfinite(tr)) or tr <= 0:
        tr = float(np.trace(M_test))
    if (not np.isfinite(tr)) or tr <= 0:
        tr = 1.0

    ridge = float(reg.eps0) * tr / p
    eff_raw = d_efficiency_relative_shared_ridge(M_test, M_opt, p, ridge=ridge)
    return float(min(1.0, eff_raw))


def _argmax_one_point_variance(
    model,
    base_design: Design,
    theta_vec: Array,
    bounds: Bounds,
    *,
    reg: AutoRegularization,
    seed: Optional[int],
    maxiter: int,
) -> Array:
    """
    For D-opt and one-point augmentation, Algorithm 1 step 2 reduces to
    maximizing f(x)^T M^{-1} f(x).
    """
    d = len(bounds)
    M = information_matrix(model, base_design, theta_vec)
    M, _ = regularize_if_needed(M, reg=reg)

    def obj(x_flat: Array) -> float:
        x = np.asarray(x_flat, dtype=float).reshape(d)
        f = model.jacobian(x, theta_vec).reshape(-1, 1)
        v = float((f.T @ np.linalg.solve(M, f))[0, 0])
        return -v

    res = dual_annealing(obj, bounds=bounds, seed=seed, maxiter=maxiter)
    return np.asarray(res.x, dtype=float).reshape(d)


def _logdet_augmented_for_set(
    model,
    base_design_for_weights: Design,
    theta_vec: Array,
    X: PointSet,
    *,
    alpha_mode: str,
    alpha: float,
    rounding_digits: int,
    merge_tol: float,
    reg: AutoRegularization,
) -> float:
    xi1 = _augment_design(
        base_design_for_weights,
        [X],
        alpha_mode=alpha_mode,
        alpha=alpha,
        rounding_digits=rounding_digits,
        merge_tol=merge_tol,
    )
    M = information_matrix(model, xi1, theta_vec)
    M, _ = regularize_if_needed(M, reg=reg)
    return logdet_psd(M)


def _argmax_two_point_set(
    model,
    base_design_for_weights: Design,
    theta_vec: Array,
    bounds: Bounds,
    *,
    alpha_mode: str,
    alpha: float,
    rounding_digits: int,
    merge_tol: float,
    reg: AutoRegularization,
    seed: Optional[int],
    maxiter: int,
) -> PointSet:
    """
    For n_per_step=2, maximize logdet(M(xi1(X))) directly over X={x1,x2}.
    """
    d = len(bounds)

    def obj(u: Array) -> float:
        u = np.asarray(u, dtype=float).reshape(2 * d)
        x1 = u[:d]
        x2 = u[d:]
        X = np.vstack([x1, x2])
        return -_logdet_augmented_for_set(
            model,
            base_design_for_weights,
            theta_vec,
            X,
            alpha_mode=alpha_mode,
            alpha=alpha,
            rounding_digits=rounding_digits,
            merge_tol=merge_tol,
            reg=reg,
        )

    res = dual_annealing(obj, bounds=bounds * 2, seed=seed, maxiter=maxiter)
    u = np.asarray(res.x, dtype=float).reshape(2 * d)
    x1 = u[:d]
    x2 = u[d:]
    return _canonicalize_set(np.vstack([x1, x2]), rounding_digits)


def _support_overlap_count(new_design: Design, base_design: Design, *, merge_tol: float) -> int:
    """
    Count how many support points in new_design lie within merge_tol of the base support.
    """
    base_pts = np.asarray(base_design.points, dtype=float)
    new_pts = np.asarray(new_design.points, dtype=float)
    count = 0
    for pt in new_pts:
        dists = np.linalg.norm(base_pts - pt.reshape(1, -1), axis=1)
        if np.any(dists <= merge_tol):
            count += 1
    return int(count)


def robust_augment_two_step(
    model,
    *,
    theta_symbols: Sequence[sp.Symbol],
    theta0: Dict[sp.Symbol, float],
    bounds: Bounds,
    uncertain_params: Sequence[sp.Symbol],
    grid_spec: Optional[Dict[sp.Symbol, Tuple[float, float, int]]] = None,
    rel_range: float = 0.15,
    grid_points: int = 21,
    n_per_step: int = 1,            # supported: 1 or 2
    alpha_mode: str = "fixed",      # "fixed" or "equal_weight"
    alpha: float = 0.25,
    wynn_base: Optional[WynnFedorovOptions] = None,
    wynn_ref: Optional[WynnFedorovOptions] = None,
    reg: AutoRegularization = AutoRegularization(),
    rounding_digits: int = 3,
    merge_tol: Optional[float] = None,
    da_maxiter: int = 250,
    seed: Optional[int] = 1,
) -> RobustAugmentResult:
    if n_per_step not in (1, 2):
        raise ValueError("Currently only n_per_step=1 or n_per_step=2 is supported.")
    if alpha_mode not in ("fixed", "equal_weight"):
        raise ValueError("alpha_mode must be 'fixed' or 'equal_weight'.")

    if merge_tol is None:
        merge_tol = 10 ** (-int(rounding_digits))

    if wynn_base is None:
        wynn_base = WynnFedorovOptions(
            max_iter=60, tol=1e-5, rounding_digits=2, merge_tol=0.01, seed=seed, da_maxiter=150
        )
    if wynn_ref is None:
        wynn_ref = WynnFedorovOptions(
            max_iter=120, tol=1e-7, rounding_digits=3, merge_tol=0.001, seed=seed, da_maxiter=300
        )

    # Build Ω
    if grid_spec is None:
        spec: Dict[sp.Symbol, Tuple[float, float, int]] = {}
        for s in uncertain_params:
            if s not in theta0:
                raise ValueError(f"theta0 missing value for parameter {s}.")
            v = float(theta0[s])
            spec[s] = (v * (1.0 - rel_range), v * (1.0 + rel_range), int(grid_points))
        grid = make_grid_equidistant(spec)
    else:
        grid = make_grid_equidistant(grid_spec)

    theta0_vec = _theta_dict_to_vector(theta_symbols, theta0)

    # Base design ξ0*
    res0 = wynn_fedorov_d_opt(model=model, theta=theta0_vec, bounds=bounds, n_support=6, options=wynn_base)
    xi0 = _merge_close_support(
        res0.design.points,
        res0.design.weights,
        digits=rounding_digits,
        merge_tol=merge_tol,
    )

    xi0_for_weights = _equal_weight_same_support(xi0) if alpha_mode == "equal_weight" else xi0
    d = xi0.points.shape[1]

    # Scenario-optimal designs for formal efficiency comparisons
    theta_vecs: List[Array] = []
    xi_opt: List[Design] = []
    for scen in grid:
        theta_s = dict(theta0)
        theta_s.update(scen)
        th = _theta_dict_to_vector(theta_symbols, theta_s)
        theta_vecs.append(th)

        res_opt = wynn_fedorov_d_opt(model=model, theta=th, bounds=bounds, n_support=6, options=wynn_ref)
        xi_opt.append(
            _merge_close_support(
                res_opt.design.points,
                res_opt.design.weights,
                digits=rounding_digits,
                merge_tol=merge_tol,
            )
        )

    # Step 2: Δ = {X_beta_z}
    delta_raw: List[PointSet] = []
    for th in theta_vecs:
        if n_per_step == 1:
            x = _argmax_one_point_variance(
                model,
                xi0_for_weights,
                th,
                bounds,
                reg=reg,
                seed=seed,
                maxiter=da_maxiter,
            )
            X = _canonicalize_set(x.reshape(1, d), rounding_digits)
        else:
            X = _argmax_two_point_set(
                model,
                xi0_for_weights,
                th,
                bounds,
                alpha_mode=alpha_mode,
                alpha=alpha,
                rounding_digits=rounding_digits,
                merge_tol=merge_tol,
                reg=reg,
                seed=seed,
                maxiter=da_maxiter,
            )
        delta_raw.append(X)

    # unique Δ
    uniq: Dict[Tuple[Tuple[float, ...], ...], PointSet] = {}
    for X in delta_raw:
        Xc = _canonicalize_set(X, rounding_digits)
        key = tuple(tuple(row.tolist()) for row in Xc)
        uniq[key] = Xc
    Delta = list(uniq.values())

    # Base efficiencies
    base_effs = [
        _d_efficiency_formal(model, xi0_for_weights, xi_opt[i], theta_vecs[i], reg=reg)
        for i in range(len(grid))
    ]

    # Step 3: choose X1*
    best_X1 = None
    best_score = -np.inf
    for X in Delta:
        xi1 = _augment_design(
            xi0_for_weights,
            [X],
            alpha_mode=alpha_mode,
            alpha=alpha,
            rounding_digits=rounding_digits,
            merge_tol=merge_tol,
        )
        effs1 = [
            _d_efficiency_formal(model, xi1, xi_opt[i], theta_vecs[i], reg=reg)
            for i in range(len(grid))
        ]
        improvements = [effs1[i] - base_effs[i] for i in range(len(grid))]
        score = float(np.max(improvements))
        if score > best_score:
            best_score = score
            best_X1 = X

    assert best_X1 is not None
    X1_star = _canonicalize_set(np.asarray(best_X1, dtype=float).reshape(n_per_step, d), rounding_digits)

    # Step 5: choose X2*
    best_X2 = None
    best_min_eff = -np.inf
    for X2 in Delta:
        xi2 = _augment_design(
            xi0_for_weights,
            [X1_star, X2],
            alpha_mode=alpha_mode,
            alpha=alpha,
            rounding_digits=rounding_digits,
            merge_tol=merge_tol,
        )
        effs2 = [
            _d_efficiency_formal(model, xi2, xi_opt[i], theta_vecs[i], reg=reg)
            for i in range(len(grid))
        ]
        score = float(np.min(effs2))
        if score > best_min_eff:
            best_min_eff = score
            best_X2 = X2

    assert best_X2 is not None
    X2_star = _canonicalize_set(np.asarray(best_X2, dtype=float).reshape(n_per_step, d), rounding_digits)

    xi_augm = _augment_design(
        xi0_for_weights,
        [X1_star, X2_star],
        alpha_mode=alpha_mode,
        alpha=alpha,
        rounding_digits=rounding_digits,
        merge_tol=merge_tol,
    )

    overlap_count = _support_overlap_count(xi_augm, xi0_for_weights, merge_tol=merge_tol)
    n_new_support = int(xi_augm.points.shape[0] - xi0_for_weights.points.shape[0])

    meta = {
        "criterion": "D",
        "alpha_mode": alpha_mode,
        "alpha": float(alpha) if alpha_mode == "fixed" else None,
        "n_per_step": int(n_per_step),
        "n_added_total": int(2 * n_per_step),
        "uncertain_params": [str(s) for s in uncertain_params],
        "n_scenarios": len(grid),
        "delta_size": len(Delta),
        "rounding_digits": int(rounding_digits),
        "merge_tol": float(merge_tol),
        "n_base_support": int(xi0_for_weights.points.shape[0]),
        "n_aug_support": int(xi_augm.points.shape[0]),
        "n_new_support": n_new_support,
        "overlap_with_base_support": overlap_count,
        "stop_signal_reassigns_base_weight": bool(n_new_support <= 0),
        "wynn_base": wynn_base,
        "wynn_ref": wynn_ref,
    }

    return RobustAugmentResult(
        design_base=xi0_for_weights,
        design_augm=xi_augm,
        x1_star=X1_star,
        x2_star=X2_star,
        delta=Delta,
        meta=meta,
    )
