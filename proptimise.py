# Use Optax for optimization
import optax
from jax import numpy as jnp
from jax import jit, vmap, grad
from jax.scipy.optimize import minimize
from jax import random
from typing import Callable
# Internal Libraries
from airfoil import parametric_coeffs
from solvers import NewtonSolver
### Propeller Optimization Framework
