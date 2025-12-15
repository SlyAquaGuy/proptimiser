import jax
import jax.numpy as jnp
from typing import Callable

from init import Inputs

class NewtonSolver:
    """
    JAX-compatible Newton solver for small nonlinear systems.

    Solves:
        F(x, *args) = 0

    Solver behaviour is configured via `inputs`.
    """

    def __init__(self, fun: Callable):
        """
        Parameters
        ----------
        fun : callable
            Residual function F(x, *args) -> scalar or 1D array
        """
        self.fun = fun

        # Pull numerical settings from inputs (snapshot at construction time)
        self.n_iter = Inputs.newton_max_iter
        self.damping = Inputs.newton_damping
        self.eps = Inputs.newton_eps

        # Autodiff Jacobian
        # jacfwd is usually best for small systems
        self._jac = jax.jacfwd(fun)

    def step(self, x, *args):
        F = self.fun(x, *args)           # (m,)
        J = self._jac(x, *args)          # (m,m)

        dx = jnp.linalg.solve(
            J + self.eps * jnp.eye(J.shape[0], dtype=J.dtype),
            F,
        )
        return x - self.damping * dx


    def solve(self, x0, *args):
        """
        Fixed-iteration Newton solve (JIT-safe).
        """
        x = x0
        for _ in range(self.n_iter):
            x = self.step(x, *args)
        return x
2