from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
import sympy as sp


Array = np.ndarray


@dataclass
class SympyModel:
    """Scalar-response nonlinear model defined symbolically with SymPy.

    This class supports the "case 2" interface: the user provides a callable
    `eta(*x_symbols)` returning a SymPy expression (sympy.Expr) that depends on
    the parameter symbols in `theta_symbols`.

    The class builds the symbolic gradient with respect to parameters and compiles
    it to a fast numerical function using `sympy.lambdify`. After initialization,
    the core algorithms only need numerical Jacobian evaluations.

    Parameters
    ----------
    eta :
        Callable returning a SymPy expression, signature `eta(*x_symbols)`.
    x_symbols :
        Sequence of design-variable symbols `(x1, x2, ..., xd)`.
    theta_symbols :
        Sequence of parameter symbols `(θ1, ..., θp)`.

    Notes
    -----
    - This MVP assumes a *scalar* model response (one mean function η). If you later
      need multi-response models, the Jacobian should return shape (q, p) and the
      Fisher information matrix will incorporate the response covariance.
    - Internally, argument order for lambdify is `[x_symbols..., theta_symbols...]`.

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol("x")
    >>> a, b = sp.symbols("a b")
    >>> def eta(x): return a * sp.exp(-b*x)
    >>> model = SympyModel(eta=eta, x_symbols=[x], theta_symbols=[a, b])
    >>> model.jacobian([1.0], [1.0, 0.5]).shape
    (2,)
    """
    eta: Callable[..., sp.Expr]
    x_symbols: Sequence[sp.Symbol]
    theta_symbols: Sequence[sp.Symbol]

    def __post_init__(self) -> None:
        self.x_symbols = list(self.x_symbols)
        self.theta_symbols = list(self.theta_symbols)
        if len(self.x_symbols) < 1:
            raise ValueError("x_symbols must contain at least one symbol.")
        if len(self.theta_symbols) < 1:
            raise ValueError("theta_symbols must contain at least one symbol.")

        # Build symbolic expression and Jacobian wrt parameters
        eta_expr = self.eta(*self.x_symbols)
        if not isinstance(eta_expr, sp.Expr):
            raise TypeError("eta(*x_symbols) must return a sympy.Expr.")

        J = sp.Matrix([sp.diff(eta_expr, th) for th in self.theta_symbols])  # (p, 1)

        # Compile to numerical functions
        # Input order: x_symbols then theta_symbols
        args = list(self.x_symbols) + list(self.theta_symbols)
        self._jac_num = sp.lambdify(args, J, modules="numpy")

    @property
    def d(self) -> int:
        return len(self.x_symbols)

    @property
    def p(self) -> int:
        return len(self.theta_symbols)

    def jacobian(self, x: Array, theta: Array) -> Array:
        """Evaluate the gradient of η with respect to parameters.

        Parameters
        ----------
        x :
            Design point with shape (d,).
        theta :
            Parameter vector with shape (p,).

        Returns
        -------
        f :
            Gradient vector f(x, θ) = ∂η(x, θ)/∂θ with shape (p,).

        Raises
        ------
        ValueError
            If shapes of `x` or `theta` do not match model dimensions.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x must have length d={self.d}. Got {x.shape[0]}.")
        if theta.shape[0] != self.p:
            raise ValueError(f"theta must have length p={self.p}. Got {theta.shape[0]}.")

        vals = list(x) + list(theta)
        J = np.asarray(self._jac_num(*vals), dtype=float).reshape(self.p)
        return J