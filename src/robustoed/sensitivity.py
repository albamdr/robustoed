from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import sympy as sp

from .types import Design
from .grid import make_grid_equidistant
from .fisher import information_matrix
from .criterion_d import d_efficiency_relative_safe, d_efficiency_relative_shared_ridge
from .regularize import AutoRegularization


Array = np.ndarray


@dataclass
class SensitivityReport:
    """Summary of design robustness over a parameter-uncertainty set Ω.

    Attributes
    ----------
    grid :
        List of scenarios (dict symbol->value) used in the analysis.
    efficiencies :
        Array of relative efficiencies of the base design over the grid, shape (R,).
    summary :
        Dictionary with keys like 'min', 'mean', 'max', 'p05', 'p95'.
    meta :
        Metadata (criterion, params varied, grid spec, etc.).
    """
    grid: List[Dict[sp.Symbol, float]]
    efficiencies: Array
    summary: Dict[str, float]
    meta: Dict[str, object]


def _theta_dict_to_vector(theta_symbols: Sequence[sp.Symbol], theta_dict: Dict[sp.Symbol, float]) -> Array:
    """Convert {Symbol: value} to theta vector ordered as theta_symbols."""
    return np.array([float(theta_dict[s]) for s in theta_symbols], dtype=float)


def _scenario_from_theta0(theta0: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
    return {k: float(v) for k, v in theta0.items()}


def sensitivity_report_d(
    model,
    base_design: Design,
    theta_symbols: Sequence[sp.Symbol],
    theta0: Dict[sp.Symbol, float],
    *,
    uncertain_params: Sequence[sp.Symbol],
    rel_range: float = 0.15,
    grid_points: int = 21,
    grid_spec: Optional[Dict[sp.Symbol, Tuple[float, float, int]]] = None,
    reg: AutoRegularization = AutoRegularization(),
) -> SensitivityReport:
    """Compute a D-efficiency robustness report of a fixed design over a parameter grid.

    This report helps identify which parameters should be treated as uncertain
    (i.e., which ones cause the largest efficiency loss when perturbed).

    Parameters
    ----------
    model :
        Model providing `jacobian(x, theta_vec)`.
    base_design :
        Design ξ to be evaluated (fixed).
    theta_symbols :
        Parameter symbols in a fixed order (consistent with model definition).
    theta0 :
        Nominal parameter values as a dict {symbol: value}.
    uncertain_params :
        Subset of parameters to vary. Parameters not listed stay fixed at theta0.
    rel_range :
        Relative range for each varied parameter (±rel_range) around theta0,
        used only if grid_spec is None.
    grid_points :
        Number of points per varied parameter axis (>=2), used only if grid_spec is None.
    grid_spec :
        Explicit grid specification {symbol: (low, high, n)}. If provided, overrides
        rel_range/grid_points for those symbols.
    reg :
        Auto-regularization configuration.

    Returns
    -------
    report :
        SensitivityReport with efficiencies over Ω and summary statistics.

    Notes
    -----
    Efficiencies are computed relative to the locally D-optimal design at theta0
    (i.e., the design is compared to itself at theta0). This is a pragmatic first
    step; later we can compute efficiency relative to the scenario-specific optimum.
    """
    # Build grid Ω
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

    # Reference information matrix at theta0 (for base design)
    theta0_vec = _theta_dict_to_vector(theta_symbols, theta0)
    M_ref = information_matrix(model, base_design, theta0_vec)

    effs = []
    for scen in grid:
        # build full theta dict (vary only uncertain_params)
        theta_s = dict(theta0)
        theta_s.update(scen)

        theta_vec = _theta_dict_to_vector(theta_symbols, theta_s)
        M_test = information_matrix(model, base_design, theta_vec)

        p = theta_vec.shape[0]
        eff = d_efficiency_relative_safe(M_test, M_ref, p, reg=reg)
        effs.append(float(eff))

    effs_arr = np.array(effs, dtype=float)

    summary = {
        "min": float(np.min(effs_arr)),
        "mean": float(np.mean(effs_arr)),
        "max": float(np.max(effs_arr)),
        "p05": float(np.quantile(effs_arr, 0.05)),
        "p95": float(np.quantile(effs_arr, 0.95)),
    }

    meta = {
        "criterion": "D",
        "rel_range": rel_range,
        "grid_points": grid_points,
        "uncertain_params": [str(s) for s in uncertain_params],
        "n_scenarios": len(grid),
        "max_eff_raw": float(max_raw),
        "n_clipped": int(n_clipped),
    }

    return SensitivityReport(grid=grid, efficiencies=effs_arr, summary=summary, meta=meta)

from .optim import wynn_fedorov_d_opt, WynnFedorovOptions


def sensitivity_report_d_vs_scenario_optimum(
    model,
    base_design: Design,
    theta_symbols: Sequence[sp.Symbol],
    theta0: Dict[sp.Symbol, float],
    *,
    bounds: List[Tuple[float, float]],
    uncertain_params: Sequence[sp.Symbol],
    rel_range: float = 0.15,
    grid_points: int = 21,
    grid_spec: Optional[Dict[sp.Symbol, Tuple[float, float, int]]] = None,
    wynn_options: Optional[WynnFedorovOptions] = None,
    reg: AutoRegularization = AutoRegularization(),
) -> SensitivityReport:
    """Compute D-efficiency of a fixed design relative to the scenario-specific D-optimum.

    For each scenario β_z in Ω:
    1) compute the locally D-optimal design ξ*_βz (via Wynn–Fedorov),
    2) compute the relative D-efficiency of `base_design` at β_z:

        eff_z = ( det M(base_design; β_z) / det M(ξ*_βz; β_z) )^(1/p)

    This is the efficiency notion used in the theoretical robust-augmentation workflow:
    efficiencies are expected to satisfy eff_z <= 1 (up to numerical tolerances).

    Parameters
    ----------
    model :
        Model providing `jacobian(x, theta_vec)`.
    base_design :
        Design to be evaluated (fixed).
    theta_symbols :
        Parameter symbols in a fixed order (consistent with model definition).
    theta0 :
        Nominal parameter values {symbol: value}.
    bounds :
        Design space box constraints as a list of (low, high) pairs of length d.
    uncertain_params :
        Subset of parameters to vary across Ω (others fixed at theta0).
    rel_range, grid_points, grid_spec :
        Control the construction of Ω (equidistant grid by default).
    wynn_options :
        Options for the local D-optimal design computation. If None, defaults are used.
    reg :
        Auto-regularization configuration used for efficiency computations.

    Returns
    -------
    report :
        SensitivityReport with per-scenario efficiencies and summary statistics.
    """
    if wynn_options is None:
        wynn_options = WynnFedorovOptions()

    # Build grid Ω
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

    # Precompute base-design information over scenarios
    effs: List[float] = []

    n_clipped = 0
    max_raw = -np.inf
    for scen in grid:
        theta_s = dict(theta0)
        theta_s.update(scen)
        theta_vec = _theta_dict_to_vector(theta_symbols, theta_s)

        # Compute scenario-specific D-optimal design ξ*_βz
        res_opt = wynn_fedorov_d_opt(
            model=model,
            theta=theta_vec,
            bounds=bounds,
            init_design=None,
            n_support=6,
            options=wynn_options,
        )
        design_opt = res_opt.design

        # Compute information matrices at scenario θ
        M_test = information_matrix(model, base_design, theta_vec)
        M_opt = information_matrix(model, design_opt, theta_vec)

        p = theta_vec.shape[0]
        # Shared ridge based on a common scale (trace of M_opt, fallback to M_test)

        tr = float(np.trace(M_opt))

        if (not np.isfinite(tr)) or tr <= 0:

            tr = float(np.trace(M_test))

        if (not np.isfinite(tr)) or tr <= 0:

            tr = 1.0

        ridge = float(reg.eps0) * tr / p

        eff_raw = d_efficiency_relative_shared_ridge(M_test, M_opt, p, ridge=ridge)

        max_raw = max(max_raw, float(eff_raw))

        if eff_raw > 1.0 + 1e-6:

            n_clipped += 1

        eff = min(1.0, float(eff_raw))
        effs.append(float(eff))

    effs_arr = np.array(effs, dtype=float)

    summary = {
        "min": float(np.min(effs_arr)),
        "mean": float(np.mean(effs_arr)),
        "max": float(np.max(effs_arr)),
        "p05": float(np.quantile(effs_arr, 0.05)),
        "p95": float(np.quantile(effs_arr, 0.95)),
    }

    meta = {
        "criterion": "D",
        "comparison": "scenario_optimum",
        "rel_range": rel_range,
        "grid_points": grid_points,
        "uncertain_params": [str(s) for s in uncertain_params],
        "n_scenarios": len(grid),
        "max_eff_raw": float(max_raw),
        "n_clipped": int(n_clipped),
        "wynn_options": wynn_options,
    }

    return SensitivityReport(grid=grid, efficiencies=effs_arr, summary=summary, meta=meta)