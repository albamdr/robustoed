import numpy as np
import sympy as sp

from robustoed import SympyModel
from robustoed.augment import robust_augment_two_step
from robustoed.optim import WynnFedorovOptions
from robustoed.sensitivity import sensitivity_report_d_vs_scenario_optimum

# Model: eta(x) = a * exp(-b*x)
x = sp.Symbol("x")
a, b = sp.symbols("a b")

def eta(x):
    return a * sp.exp(-b * x)

model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])

theta0 = {a: 1.0, b: 0.5}
bounds = [(0.0, 10.0)]

opts_base = WynnFedorovOptions(
    max_iter=60, tol=1e-5,
    rounding_digits=2, merge_tol=0.01,
    min_weight=1e-4, weight_digits=6,
    seed=1, da_maxiter=150
)

opts_ref = WynnFedorovOptions(
    max_iter=120, tol=1e-7,
    rounding_digits=3, merge_tol=0.001,
    min_weight=1e-6, weight_digits=8,
    seed=1, da_maxiter=300
)

def run_case(n_per_step: int, alpha_mode: str):
    res = robust_augment_two_step(
        model,
        theta_symbols=[a, b],
        theta0=theta0,
        bounds=bounds,
        uncertain_params=[b],
        rel_range=0.15,
        grid_points=11,
        n_per_step=n_per_step,
        alpha_mode=alpha_mode,
        alpha=0.25,
        wynn_base=opts_base,
        wynn_ref=opts_ref,
        rounding_digits=3,
        merge_tol=0.01,
        da_maxiter=250,
        seed=1,
    )
    print(f"\n=== CASE n_per_step={n_per_step}, alpha_mode={alpha_mode} ===")
    print("x1*:", res.x1_star.tolist())
    print("x2*:", res.x2_star.tolist())
    print("aug points:", res.design_augm.points.ravel().tolist())
    print("aug weights:", res.design_augm.weights.tolist())
    print("meta:", res.meta)

    report = sensitivity_report_d_vs_scenario_optimum(
        model=model,
        base_design=res.design_augm,
        theta_symbols=[a, b],
        theta0=theta0,
        bounds=bounds,
        uncertain_params=[b],
        rel_range=0.15,
        grid_points=11,
        wynn_options=opts_ref,
    )
    print("Formal sensitivity summary (augmented):", report.summary)
    print("max efficiency (augmented):", float(report.efficiencies.max()))
    print("meta (augmented report):", report.meta)

for n in (1, 2):
    for mode in ("fixed", "equal_weight"):
        run_case(n, mode)
