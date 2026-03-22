from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import sympy as sp

from .types import Design
from .sensitivity import SensitivityReport, sensitivity_report_d_vs_scenario_optimum
from .optim import WynnFedorovOptions


@dataclass
class ParameterScreeningResult:
    """
    Result of one-at-a-time parameter screening.

    Attributes
    ----------
    reports :
        Per-parameter sensitivity reports.
    ranking :
        Sorted list of tuples (parameter_name, score), where lower score means
        more sensitivity. By default the score is the minimum efficiency.
    meta :
        Metadata describing the screening setup.
    """
    reports: Dict[str, SensitivityReport]
    ranking: List[Tuple[str, float]]
    meta: Dict[str, object]


def screen_uncertain_parameters_d(
    model,
    base_design: Design,
    *,
    theta_symbols: Sequence[sp.Symbol],
    theta0: Dict[sp.Symbol, float],
    bounds: List[Tuple[float, float]],
    param_specs: Dict[sp.Symbol, Tuple[float, float, int]],
    wynn_options: WynnFedorovOptions,
) -> ParameterScreeningResult:
    """
    Screen parameter sensitivity one parameter at a time using formal D-efficiency.

    For each parameter s in param_specs, this function varies only s over the user-
    specified interval (low, high, n_points), keeping all other parameters fixed at
    their nominal values in theta0.

    The fixed base design is compared against the scenario-specific D-optimal design
    at each scenario using sensitivity_report_d_vs_scenario_optimum.

    Parameters
    ----------
    model :
        Model object.
    base_design :
        Fixed design to evaluate.
    theta_symbols :
        Ordered parameter symbols.
    theta0 :
        Nominal parameter values.
    bounds :
        Design-space bounds.
    param_specs :
        Dict mapping each parameter symbol to a tuple (low, high, n_points).
    wynn_options :
        Options used to compute the scenario-specific D-optimal designs.

    Returns
    -------
    ParameterScreeningResult
        reports :
            Per-parameter sensitivity reports.
        ranking :
            Sorted by minimum efficiency (ascending).
        meta :
            Screening metadata.
    """
    reports: Dict[str, SensitivityReport] = {}
    ranking: List[Tuple[str, float]] = []

    for s, (low, high, n_points) in param_specs.items():
        if s not in theta0:
            raise ValueError(f"theta0 missing nominal value for parameter {s}.")
        if int(n_points) < 2:
            raise ValueError(f"n_points for parameter {s} must be >= 2.")
        if float(low) >= float(high):
            raise ValueError(f"For parameter {s}, low must be < high.")

        rep = sensitivity_report_d_vs_scenario_optimum(
            model=model,
            base_design=base_design,
            theta_symbols=theta_symbols,
            theta0=theta0,
            bounds=bounds,
            uncertain_params=[s],
            grid_spec={s: (float(low), float(high), int(n_points))},
            wynn_options=wynn_options,
        )

        key = str(s)
        reports[key] = rep
        ranking.append((key, float(rep.summary["min"])))

    ranking.sort(key=lambda t: t[1])

    meta = {
        "criterion": "D",
        "comparison": "scenario_optimum",
        "n_parameters_screened": len(param_specs),
        "parameter_order": [str(s) for s in param_specs.keys()],
        "ranking_score": "min_efficiency",
    }

    return ParameterScreeningResult(
        reports=reports,
        ranking=ranking,
        meta=meta,
    )
