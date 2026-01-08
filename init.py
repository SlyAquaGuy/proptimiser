from dataclasses import dataclass
import jax
import jax.numpy as jnp

# Freeze configuration parameters at runtime
@dataclass(frozen=True)
class Inputs:
    # Optimisation / design settings which remain constant during execution

    # Outer-loop optimisation (Optax / similar)
    opt_max_iter: int = 200
    opt_tol: float = 1e-4

    # Newton solver settings
    newton_max_iter: int = 100
    newton_eps: float = 1e-4
    newton_damping: float = 0.1

    # Thrust constraint
    thrust: float = 10.0 # Newtons

    ## Discretisation settings
    # Radial discretisation
    n_r: int = 15
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 20

    ## Physical / modelling flags
    use_prandtl_tip_loss: bool = False
    use_prandtl_root_loss: bool = False

    # Diagnostics / debugging
    verbose: bool = True
    record_convergence_history: bool = False

@dataclass(frozen=True)
class Params:
    # Domain of Integration
    r: jnp.ndarray          # radial stations
    dr: jnp.ndarray         # radial spacing
    psi: jnp.ndarray        # azimuthal stations
    dpsi: jnp.ndarray       # azimuthal spacing


    # Geometry & discretization
    N: jnp.ndarray          # number of blades
    R: jnp.ndarray          # propeller radius
    c: jnp.ndarray          # chord length at radial stations
    beta: jnp.ndarray       # blade twist angle (radians)


    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity
    V_yz: jnp.ndarray       # tangential / side inflow velocity
    omega: jnp.ndarray      # rotational speed

    V_ia_0: jnp.ndarray     # initial guess/storage for V_ia_0


    # Physical properties
    rho: float = 1.225      # air density

    # Model Selection
    inflow_model: int = 1 # 0 = basic, 1 = Pitt & Peters, 2 = DTU
    coeff_method: int = 0 # 0 = parametric, 1 = lookup table
    tip_loss_model: int = 0 # 0 = none, 1 = Prandtl
    root_loss_model: int = 0 # 0 = none, 1 = Prandtl

