import jax
import jax.numpy as jnp
from typing import Callable

from init import inputs

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
        self.n_iter = inputs.newton_n_iter
        self.damping = inputs.newton_damping
        self.eps = inputs.newton_eps

        # Autodiff Jacobian
        # jacfwd is usually best for small systems
        self._jac = jax.jacfwd(fun)

    def step(self, x, *args):
        """
        Perform one Newton update.
        """
        F = self.fun(x, *args)
        J = self._jac(x, *args)

        # Scalar unknown
        def scalar_update():
            return F / (J + self.eps)

        # Vector unknown
        def vector_update():
            return jnp.linalg.solve(
                J + self.eps * jnp.eye(J.shape[0], dtype=J.dtype),
                F,
            )

        dx = jax.lax.cond(
            jnp.ndim(F) == 0,
            lambda _: scalar_update(),
            lambda _: vector_update(),
            operand=None,
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