import numpy as np
import sympy as sp

from robustoed.model_sympy import SympyModel
from robustoed.optim import WynnFedorovOptions, wynn_fedorov_d_opt
from robustoed.types import Design
from robustoed.screening import screen_uncertain_parameters_d


# ============================================================
# 1) BARANYI MODEL
# ============================================================

t = sp.Symbol("t", real=True)

y0, ymax, mu_max, h0, nu, m = sp.symbols(
    "y0 ymax mu_max h0 nu m", positive=True, real=True
)

def eta_baranyi(t):
    L = sp.log(
        sp.exp(-nu * t) + sp.exp(-h0) - sp.exp(-nu * t - h0)
    )

    term1 = y0 + mu_max * t + (1 / mu_max) * L

    exp_arg = m * mu_max * t + (m / mu_max) * L

    term2 = (1 / m) * sp.log(
        1 + (sp.exp(exp_arg) - 1) / sp.exp(m * (ymax - y0))
    )

    return term1 - term2


model = SympyModel(
    eta=eta_baranyi,
    x_symbols=[t],
    theta_symbols=[y0, ymax, mu_max, h0, nu, m],
)

# ============================================================
# 2) NOMINAL VALUES FROM THE PAPER
# ============================================================

theta0 = {
    y0: 2.364,
    ymax: 21.097,
    mu_max: 1.089,
    h0: 2.657,
    nu: 1.089,
    m: 1.0,
}

theta_symbols = [y0, ymax, mu_max, h0, nu, m]
theta0_vec = np.array([theta0[s] for s in theta_symbols], dtype=float)

bounds = [(0.0, 28.0)]


# ============================================================
# 3) SMALL HELPERS TO CLEAN THE BASE DESIGN
# ============================================================

def clean_design(points, weights, *, merge_tol=0.25, prune_tol=1e-3, digits=2):
    pts = np.round(np.asarray(points, dtype=float), digits)
    w = np.asarray(weights, dtype=float)

    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    w = w[idx]

    merged_pts = []
    merged_w = []

    cur_pt = pts[0].copy()
    cur_w = float(w[0])

    for pt, wi in zip(pts[1:], w[1:]):
        if abs(float(pt[0] - cur_pt[0])) <= merge_tol:
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

    pts = np.round(np.vstack(merged_pts), digits)
    w = np.asarray(merged_w, dtype=float)

    keep = w >= prune_tol
    pts = pts[keep]
    w = w[keep]

    w = w / w.sum()
    return Design(pts, w)


def force_equal_weights_if_saturated(design: Design, p: int) -> Design:
    pts = np.asarray(design.points, dtype=float)
    if pts.shape[0] == p:
        w = np.ones(p, dtype=float) / p
        return Design(pts, w)
    return design


# ============================================================
# 4) BASE DESIGN OPTIONS
# ============================================================

opts_base = WynnFedorovOptions(
    max_iter=120,
    tol=1e-6,
    rounding_digits=2,
    merge_tol=0.05,
    min_weight=1e-5,
    weight_digits=8,
    seed=1,
    da_maxiter=220,
)

# Much lighter options for screening, otherwise it is too slow
opts_ref = WynnFedorovOptions(
    max_iter=50,
    tol=1e-5,
    rounding_digits=2,
    merge_tol=0.05,
    min_weight=1e-4,
    weight_digits=6,
    seed=1,
    da_maxiter=120,
)


# ============================================================
# 5) BUILD BASE DESIGN
# ============================================================

print("Construyendo diseño D-óptimo base para Baranyi...")
res0 = wynn_fedorov_d_opt(
    model=model,
    theta=theta0_vec,
    bounds=bounds,
    n_support=6,
    options=opts_base,
)

design0_raw = res0.design
design0 = clean_design(
    design0_raw.points,
    design0_raw.weights,
    merge_tol=0.25,
    prune_tol=1e-3,
    digits=2,
)
design0 = force_equal_weights_if_saturated(design0, p=len(theta_symbols))

print("\n=== BASE DESIGN RAW ===")
print("points:", design0_raw.points.ravel().tolist())
print("weights:", design0_raw.weights.tolist())
print("n_support:", len(design0_raw.weights))
print("stop_reason:", res0.stop_reason)

print("\n=== BASE DESIGN CLEANED ===")
print("points:", design0.points.ravel().tolist())
print("weights:", design0.weights.tolist())
print("n_support:", len(design0.weights))

if len(design0.weights) != len(theta_symbols):
    print("\nAVISO: el diseño limpio todavía no es saturado con 6 soportes.")
    print("No tiene sentido lanzar aún el screening completo del paper.")
    raise SystemExit(0)


# ============================================================
# 6) PARAMETER RANGES: ±15% AS IN THE PAPER
# ============================================================

def pm15(v: float):
    return (0.85 * float(v), 1.15 * float(v), 11)


param_specs = {
    y0: pm15(theta0[y0]),
    ymax: pm15(theta0[ymax]),
    mu_max: pm15(theta0[mu_max]),
    h0: pm15(theta0[h0]),
    nu: pm15(theta0[nu]),
    m: pm15(theta0[m]),
}


# ============================================================
# 7) SCREENING
# ============================================================

print("\nLanzando screening de sensibilidad (versión ligera)...")
screen = screen_uncertain_parameters_d(
    model=model,
    base_design=design0,
    theta_symbols=theta_symbols,
    theta0=theta0,
    bounds=bounds,
    param_specs=param_specs,
    wynn_options=opts_ref,
)

print("\n=== PARAMETER SCREENING RANKING (Baranyi) ===")
print("(menor min_eff => mayor sensibilidad)")
for name, score in screen.ranking:
    print(f"{name}: min_eff={score:.6f}")

print("\n=== PER-PARAMETER SUMMARIES ===")
for name, rep in screen.reports.items():
    print(f"\nParameter: {name}")
    print("summary:", rep.summary)
    print("meta:", rep.meta)

print("\nScreening meta:", screen.meta)
