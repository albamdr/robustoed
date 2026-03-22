import numpy as np
import sympy as sp

from robustoed import SympyModel, Design
from robustoed.optim import wynn_fedorov_d_opt, WynnFedorovOptions
from robustoed.sensitivity import sensitivity_report_d

# Model: eta(x) = a * exp(-b*x)
x = sp.Symbol("x")
a, b = sp.symbols("a b")

def eta(x):
    return a * sp.exp(-b * x)

model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])

theta0 = {a: 1.0, b: 0.5}
theta0_vec = np.array([theta0[a], theta0[b]], dtype=float)

bounds = [(0.0, 10.0)]
opts = WynnFedorovOptions(max_iter=50, tol=1e-5, rounding_digits=2, seed=1, da_maxiter=100)

# Compute local D-opt at theta0
res = wynn_fedorov_d_opt(model=model, theta=theta0_vec, bounds=bounds, n_support=4, options=opts)
design0 = res.design

print("Base design points:", design0.points.ravel().tolist())
print("Base design weights:", design0.weights.tolist())

# Sensitivity: vary parameter b by ±15%
report = sensitivity_report_d(
    model=model,
    base_design=design0,
    theta_symbols=[a, b],
    theta0=theta0,
    uncertain_params=[b],
    rel_range=0.15,
    grid_points=21,
)

print("Sensitivity summary:", report.summary)