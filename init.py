from dataclasses import dataclass
import jax.numpy as jnp

# Freeze configuration parameters at runtime
@dataclass(frozen=True)
class inputs:

    # Optimisation / design settings

    # Outer-loop optimisation (Optax / similar)
    opt_max_iter: int = 200
    opt_tol: float = 1e-6

    # Thrust constraint
    thrust = 10 # Newtons


    ## Discretisation settings

    # Radial discretisation
    n_r: int = 5

    # Azimuthal discretisation (for skewed inflow)
    n_phi: int = 5

    ## Physical / modelling flags
    use_prandtl_tip_loss: bool = False
    use_prandtl_root_loss: bool = False
    use_skew_inflow_model: bool = True

    # Diagnostics / debugging

    verbose: bool = False
    record_convergence_history: bool = False

# Freeze runtime parameters for the propeller model
@dataclass(frozen=True)
class params:
    # Geometry & discretization
    N: int                  # number of blades
    c: jnp.ndarray          # chord length at radial stations
    phi: jnp.ndarray        # local pitch angle (rad)
    r: jnp.ndarray          # radial stations
    dr: jnp.ndarray         # radial spacing
    psi: jnp.ndarray        # azimuthal stations
    dpsi: jnp.ndarray       # azimuthal spacing

    # Flow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity
    V_yz: jnp.ndarray       # tangential / side inflow velocity
    omega: jnp.ndarray      # rotational speed

    # Physical properties
    rho: float              # air density
    C_L: jnp.ndarray        # lift coefficient at stations
    C_D: jnp.ndarray        # drag coefficient at stations

    # Optional fields for future models
    V_ia0: jnp.ndarray = None     # initial guess for induced velocity
    xi: jnp.ndarray = None        # wake skew angle
    tip_loss: jnp.ndarray = None  # Prandtl tip loss factor
    root_loss: jnp.ndarray = None # Prandtl root loss factor

    #
