import numpy as np
import sympy as sp

from robustoed import SympyModel, screen_uncertain_parameters_d
from robustoed.optim import WynnFedorovOptions, wynn_fedorov_d_opt

# Example model: eta(x) = a * exp(-b*x)
x = sp.Symbol("x")
a, b = sp.symbols("a b")

def eta(x):
    return a * sp.exp(-b * x)

model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])

theta0 = {a: 1.0, b: 0.5}
theta0_vec = np.array([theta0[a], theta0[b]], dtype=float)
bounds = [(0.0, 10.0)]

# Base design
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

res0 = wynn_fedorov_d_opt(
    model=model,
    theta=theta0_vec,
    bounds=bounds,
    n_support=4,
    options=opts_base,
)
design0 = res0.design

print("Base design points:", design0.points.ravel().tolist())
print("Base design weights:", design0.weights.tolist())

# Strict options for scenario-specific optima
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

# User-defined ranges per parameter: (low, high, n_points)
param_specs = {
    a: (0.85, 1.15, 11),
    b: (0.425, 0.575, 11),
}

screen = screen_uncertain_parameters_d(
    model=model,
    base_design=design0,
    theta_symbols=[a, b],
    theta0=theta0,
    bounds=bounds,
    param_specs=param_specs,
    wynn_options=opts_ref,
)

print("\n=== PARAMETER SCREENING RANKING ===")
print("(lower min_eff => more sensitive)")
for name, score in screen.ranking:
    print(f"{name}: min_eff={score:.6f}")

print("\n=== PER-PARAMETER SUMMARIES ===")
for name, rep in screen.reports.items():
    print(f"\nParameter: {name}")
    print("summary:", rep.summary)
    print("meta:", rep.meta)

print("\nScreening meta:", screen.meta)
