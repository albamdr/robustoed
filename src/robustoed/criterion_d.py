from __future__ import annotations
from .regularize import AutoRegularization, regularize_if_needed

import numpy as np


Array = np.ndarray


def logdet_psd(M: Array) -> float:
    """Compute log(det(M)) robustly for positive definite matrices.

    Parameters
    ----------
    M :
        Square matrix of shape (p, p).

    Returns
    -------
    logdet :
        Natural logarithm of the determinant.

    Raises
    ------
    numpy.linalg.LinAlgError
        If M is not positive definite (sign <= 0) or slogdet is not finite.

    Notes
    -----
    For numerical stability we use `numpy.linalg.slogdet` rather than `det`.
    """
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0 or not np.isfinite(ld):
        # caller can regularize; we fail loudly here
        raise np.linalg.LinAlgError("Matrix is not positive definite for logdet.")
    return float(ld)


def d_efficiency_relative(M_test: Array, M_opt: Array, p: int) -> float:
    """Compute relative D-efficiency between two information matrices.

    Parameters
    ----------
    M_test :
        Information matrix of the test design, shape (p, p).
    M_opt :
        Information matrix of the reference (optimal) design, shape (p, p).
    p :
        Number of parameters.

    Returns
    -------
    eff :
        Relative D-efficiency:
            eff = (det(M_test) / det(M_opt))^(1/p)

    Notes
    -----
    Computed using log-determinants for numerical stability.
    """
    ld_test = logdet_psd(M_test)
    ld_opt = logdet_psd(M_opt)
    return float(np.exp((ld_test - ld_opt) / p))


def d_efficiency_relative_safe(M_test: Array, M_opt: Array, p: int, *, reg: AutoRegularization = AutoRegularization()) -> float:
    """Compute relative D-efficiency with auto-regularization fallback.

    This function applies diagonal regularization to both matrices before computing
    D-efficiency, which prevents failures when matrices are near-singular.

    Parameters
    ----------
    M_test, M_opt :
        Information matrices, shape (p, p).
    p :
        Number of parameters.
    reg :
        Auto-regularization configuration.

    Returns
    -------
    eff :
        Relative D-efficiency (float).
    """
    M_test_reg, _ = regularize_if_needed(M_test, reg=reg)
    M_opt_reg, _ = regularize_if_needed(M_opt, reg=reg)
    return d_efficiency_relative(M_test_reg, M_opt_reg, p)

def d_efficiency_relative_shared_ridge(M_test: Array, M_opt: Array, p: int, *, ridge: float) -> float:
    """Relative D-efficiency using the same diagonal ridge for both matrices.

    eff = exp((logdet(M_test + ridge I) - logdet(M_opt + ridge I))/p)
    """
    I = np.eye(p, dtype=float)
    ld_test = logdet_psd(M_test + ridge * I)
    ld_opt = logdet_psd(M_opt + ridge * I)
    return float(np.exp((ld_test - ld_opt) / p))