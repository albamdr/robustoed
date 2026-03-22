import sympy as sp
import numpy as np

from robustoed import SympyModel, screen_uncertain_parameters_d
from robustoed.augment import robust_augment_two_step
from robustoed.optim import WynnFedorovOptions, wynn_fedorov_d_opt
from robustoed.sensitivity import sensitivity_report_d_vs_scenario_optimum

# Example model: eta(x) = a * exp(-b*x)
x = sp.Symbol("x")
a, b = sp.symbols("a b")

def eta(x):
    return a * sp.exp(-b * x)

model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])

theta0 = {a: 1.0, b: 0.5}
theta0_vec = np.array([theta0[a], theta0[b]], dtype=float)
bounds = [(0.0, 10.0)]

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

# Base design
res0 = wynn_fedorov_d_opt(
    model=model,
    theta=theta0_vec,
    bounds=bounds,
    n_support=4,
    options=opts_base,
)
design0 = res0.design

print("=== BASE DESIGN ===")
print("points:", design0.points.ravel().tolist())
print("weights:", design0.weights.tolist())

# -------- STEP 1: SCREENING --------
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

print("\n=== SCREENING RANKING ===")
print("(lower min_eff => more sensitive)")
for name, score in screen.ranking:
    print(f"{name}: min_eff={score:.6f}")

print("\n=== SCREENING SUMMARIES ===")
for name, rep in screen.reports.items():
    print(f"\nParameter: {name}")
    print("summary:", rep.summary)
    print("meta:", rep.meta)

# -------- STEP 2: USER CHOOSES PARAMETERS MANUALLY --------
# This is the key point:
# the package does NOT decide automatically.
# The user reviews the screening output and chooses the parameters to robustify.

selected_params = [b]
selected_grid_spec = {
    b: (0.425, 0.575, 11),
}

print("\n=== USER-SELECTED PARAMETERS FOR ROBUSTIFICATION ===")
print([str(s) for s in selected_params])

# -------- STEP 3: ROBUST AUGMENT --------
aug = robust_augment_two_step(
    model,
    theta_symbols=[a, b],
    theta0=theta0,
    bounds=bounds,
    uncertain_params=selected_params,
    grid_spec=selected_grid_spec,
    n_per_step=1,
    alpha_mode="fixed",
    alpha=0.25,
    wynn_base=opts_base,
    wynn_ref=opts_ref,
    rounding_digits=3,
    merge_tol=0.01,
    da_maxiter=250,
    seed=1,
)

print("\n=== AUGMENT RESULT ===")
print("x1*:", aug.x1_star.tolist())
print("x2*:", aug.x2_star.tolist())
print("aug points:", aug.design_augm.points.ravel().tolist())
print("aug weights:", aug.design_augm.weights.tolist())
print("augment meta:", aug.meta)

# -------- STEP 4: FORMAL CHECK OF AUGMENTED DESIGN --------
report_aug = sensitivity_report_d_vs_scenario_optimum(
    model=model,
    base_design=aug.design_augm,
    theta_symbols=[a, b],
    theta0=theta0,
    bounds=bounds,
    uncertain_params=selected_params,
    grid_spec=selected_grid_spec,
    wynn_options=opts_ref,
)

print("\n=== FORMAL ROBUSTNESS OF AUGMENTED DESIGN ===")
print("summary:", report_aug.summary)
print("meta:", report_aug.meta)
