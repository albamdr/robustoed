from __future__ import annotations

from dataclasses import dataclass
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class AutoRegularization:
    """Automatic diagonal regularization for near-singular information matrices.

    When an information matrix M is singular or ill-conditioned, computations like
    logdet(M) or solve(M, f) may fail. This helper defines an *auto-scaled* diagonal
    regularization:

        M_reg = M + λ I
        λ = eps * trace(M) / p

    where p is the matrix dimension. The scale uses trace(M) so λ adapts to the
    magnitude of M.

    Parameters
    ----------
    eps0 :
        Initial relative strength `eps`.
    max_steps :
        Maximum number of times to multiply eps by 10 if conditioning is poor.
    cond_tol :
        If the reciprocal condition number is below this threshold, M is treated
        as ill-conditioned.

    Notes
    -----
    Regularization is a numerical safeguard. For ill-posed designs (e.g., insufficient
    support size), strong regularization can bias criterion values. We expose the
    settings so advanced users can tune them.
    """
    eps0: float = 1e-12
    max_steps: int = 6
    cond_tol: float = 1e-14


def regularize_if_needed(M: Array, reg: AutoRegularization = AutoRegularization()) -> tuple[Array, float]:
    """Regularize a square matrix if it is ill-conditioned.

    Parameters
    ----------
    M :
        Input matrix of shape (p, p).
    reg :
        Regularization configuration.

    Returns
    -------
    M_reg :
        Regularized matrix (possibly unchanged), shape (p, p).
    eps_used :
        Final eps value used in λ = eps * trace(M)/p.
    """
    M = np.asarray(M, dtype=float)
    p = M.shape[0]
    if M.shape != (p, p):
        raise ValueError("M must be square.")

    # quick check
    eps = reg.eps0
    tr = float(np.trace(M))
    if not np.isfinite(tr) or tr <= 0:
        tr = 1.0

    for _ in range(reg.max_steps + 1):
        try:
            rcond = float(np.linalg.cond(M) ** -1) if np.all(np.isfinite(M)) else 0.0
        except Exception:
            rcond = 0.0

        if rcond >= reg.cond_tol:
            return M, eps

        lam = eps * tr / p
        M = M + lam * np.eye(p)
        eps *= 10.0

    return M, eps