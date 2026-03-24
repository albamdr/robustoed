import numpy as np
import sympy as sp

from robustoed.model_sympy import SympyModel
from robustoed.optim import WynnFedorovOptions, wynn_fedorov_d_opt
from robustoed.types import Design
from robustoed.augment import robust_augment_two_step
from robustoed.sensitivity import sensitivity_report_d_vs_scenario_optimum


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
# 3) HELPERS TO CLEAN THE BASE DESIGN
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


def print_design(label, design):
    print(f"\n=== {label} ===")
    print("points:", design.points.ravel().tolist())
    print("weights:", design.weights.tolist())
    print("n_support:", len(design.weights))


# ============================================================
# 4) OPTIONS
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

opts_ref = WynnFedorovOptions(
    max_iter=80,
    tol=1e-6,
    rounding_digits=2,
    merge_tol=0.05,
    min_weight=1e-5,
    weight_digits=8,
    seed=1,
    da_maxiter=180,
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

print_design("BASE DESIGN RAW", design0_raw)
print_design("BASE DESIGN CLEANED", design0)

if len(design0.weights) != len(theta_symbols):
    raise RuntimeError(
        "El diseño base limpio no tiene 6 soportes; no seguimos con la augmentación."
    )


# ============================================================
# 6) USER-SELECTED UNCERTAIN PARAMETERS
#    As in the paper sensitivity analysis: ymax and mu_max
# ============================================================

selected_params = [ymax, mu_max]
selected_grid_spec = {
    ymax: (0.85 * theta0[ymax], 1.15 * theta0[ymax], 11),
    mu_max: (0.85 * theta0[mu_max], 1.15 * theta0[mu_max], 11),
}

print("\nParámetros seleccionados para robustificación:")
print([str(s) for s in selected_params])


# ============================================================
# 7) BASE FORMAL ROBUSTNESS (reference)
# ============================================================

report_base = sensitivity_report_d_vs_scenario_optimum(
    model=model,
    base_design=design0,
    theta_symbols=theta_symbols,
    theta0=theta0,
    bounds=bounds,
    uncertain_params=selected_params,
    grid_spec=selected_grid_spec,
    wynn_options=opts_ref,
)

print("\n=== BASE DESIGN ROBUSTNESS ===")
print("summary:", report_base.summary)
print("meta:", report_base.meta)


# ============================================================
# 8) AUGMENTATION CASES
# ============================================================

cases = [
    ("2pt_fixed", 1, "fixed", 0.25),
    ("2pt_equal_weight", 1, "equal_weight", 0.25),
    ("4pt_fixed", 2, "fixed", 0.25),
    ("4pt_equal_weight", 2, "equal_weight", 0.25),
]

all_results = []

for case_name, n_per_step, alpha_mode, alpha in cases:
    print(f"\n\n########## CASE: {case_name} ##########")

    aug = robust_augment_two_step(
        model,
        theta_symbols=theta_symbols,
        theta0=theta0,
        bounds=bounds,
        base_design=design0,
        uncertain_params=selected_params,
        grid_spec=selected_grid_spec,
        n_per_step=n_per_step,
        alpha_mode=alpha_mode,
        alpha=alpha,
        wynn_base=opts_base,
        wynn_ref=opts_ref,
        rounding_digits=2,
        merge_tol=0.25,
        da_maxiter=220,
        seed=1,
    )

    print("x1*:", aug.x1_star.ravel().tolist())
    print("x2*:", aug.x2_star.ravel().tolist())
    print_design(f"AUGMENTED DESIGN ({case_name})", aug.design_augm)
    print("augment meta:", aug.meta)

    report_aug = sensitivity_report_d_vs_scenario_optimum(
        model=model,
        base_design=aug.design_augm,
        theta_symbols=theta_symbols,
        theta0=theta0,
        bounds=bounds,
        uncertain_params=selected_params,
        grid_spec=selected_grid_spec,
        wynn_options=opts_ref,
    )

    print(f"\n=== FORMAL ROBUSTNESS ({case_name}) ===")
    print("summary:", report_aug.summary)
    print("meta:", report_aug.meta)

    all_results.append((case_name, aug, report_aug))


# ============================================================
# 9) COMPACT COMPARISON
# ============================================================

print("\n\n================ FINAL COMPARISON ================")
print("Base min efficiency:", report_base.summary["min"])
for case_name, aug, report_aug in all_results:
    print(
        f"{case_name}: "
        f"min={report_aug.summary['min']:.6f}, "
        f"mean={report_aug.summary['mean']:.6f}, "
        f"max={report_aug.summary['max']:.6f}"
    )
