import numpy as np
import sympy as sp
from robustoed import SympyModel, Design
from robustoed.optim import wynn_fedorov_d_opt, WynnFedorovOptions

# 1D example: eta(x) = a * exp(-b*x)
x = sp.Symbol("x")
a, b = sp.symbols("a b")

def eta(x):
    return a * sp.exp(-b * x)

model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])

theta0 = np.array([1.0, 0.5])
bounds = [(0.0, 10.0)]

opts = WynnFedorovOptions(max_iter=50, tol=1e-5, rounding_digits=2, seed=1, merge_tol = 0.01, da_maxiter=100)

res = wynn_fedorov_d_opt(model=model, theta=theta0, bounds=bounds, n_support=4, options=opts)

print("stop_reason:", res.stop_reason)
print("n_iter:", res.n_iter)
print("design points:", res.design.points.ravel().tolist())
print("weights:", res.design.weights.tolist())