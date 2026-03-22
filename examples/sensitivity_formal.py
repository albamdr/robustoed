import numpy as np
import sympy as sp

from robustoed import SympyModel
from robustoed.optim import wynn_fedorov_d_opt, WynnFedorovOptions
from robustoed.sensitivity import sensitivity_report_d_vs_scenario_optimum

# Model: eta(x) = a * exp(-b*x)
x = sp.Symbol("x")
a, b = sp.symbols("a b")

def eta(x):
    return a * sp.exp(-b * x)

model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])

theta0 = {a: 1.0, b: 0.5}
theta0_vec = np.array([theta0[a], theta0[b]], dtype=float)
bounds = [(0.0, 10.0)]

# Base design = local D-opt at theta0
opts_base = WynnFedorovOptions(
    max_iter=60,
    tol=1e-5,
    rounding_digits=2,
    merge_tol=0.01,
    min_weight=1e-4,
    weight_digits=6,
    seed=1,
    da_maxiter=150,
)

opts_ref = WynnFedorovOptions(
    max_iter=120,
    tol=1e-7,
    rounding_digits=3,
    merge_tol=0.001,
    min_weight=1e-6,
    weight_digits=8,
    seed=1,
    da_maxiter=300,
)
res0 = wynn_fedorov_d_opt(model=model, theta=theta0_vec, bounds=bounds, n_support=4, options=opts_base)
design0 = res0.design

print("Base design points:", design0.points.ravel().tolist())
print("Base design weights:", design0.weights.tolist())

# Formal sensitivity: compare base design to scenario-optimum at each scenario
report = sensitivity_report_d_vs_scenario_optimum(
    model=model,
    base_design=design0,
    theta_symbols=[a, b],
    theta0=theta0,
    bounds=bounds,
    uncertain_params=[b],
    rel_range=0.15,
    grid_points=11,       # pon 11 para que sea rápido
    wynn_options=opts_ref,
)

print("Formal sensitivity summary:", report.summary)
print("max efficiency (should be <= 1):", float(report.efficiencies.max()))
print("meta:", report.meta)
