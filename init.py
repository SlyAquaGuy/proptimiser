from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp

# Freeze configuration parameters at runtime
class Inputs(eqx.Module):
    # Optimisation / design settings which remain constant during execution

    # Outer-loop optimisation (Optax / similar)
    opt_max_iter: int = 200
    opt_tol: float = 1e-3

    # Newton solver settings
    newton_max_iter: int = 100
    newton_eps: float = 1e-4
    newton_damping: float = 0.1

    # Thrust constraint
    solvertype: str = "T"
    thrust: float = 100.0 # Newtons

    # Normalisation for Power (update with flat-plate guess)
    power_scale: float = 1000

    # Smoothness Constraints
    lambda_beta_smooth: float = 1e-4*power_scale
    lambda_chord_smooth: float = 1e-4*power_scale



    ## Discretisation settings
    # Radial discretisation
    n_r: int = 15
    # Azimuthal discretisation (for skewed inflow)
    n_psi: int = 20

    ## Physical / modelling flags
    use_prandtl_tip_loss: bool = False
    use_prandtl_root_loss: bool = False

    # Diagnostics / debugging
    verbose: bool = False
    record_convergence_history: bool = False

    # Model Selection
    inflow_model: int = 1   # 0 = basic, 1 = Pitt & Peters, 2 = DTU
    coeff_method: int = 0 # 0 = parametric, 1 = lookup table
    tip_loss_model: int = 0 # 0 = none, 1 = Prandtl
    root_loss_model: int = 0 # 0 = none, 1 = Prandtl

class Params(eqx.Module):
    '''
    Parameters for a given blade solution.
    Ensure that all inputs have specified shape upon assignment.
    ([None,:] for azimuthal stops, [:,None] for radial stops)

    '''
    # Domain of Integration
    r: jnp.ndarray          # radial stations
    dr: jnp.ndarray         # radial spacing (m)
    psi: jnp.ndarray        # azimuthal stations
    dpsi: jnp.ndarray       # azimuthal spacing (m)


    # Geometry & discretization
    N: jnp.ndarray          # number of blades
    R: jnp.ndarray          # propeller radius (m)
    c: jnp.ndarray          # chord length at radial stations (m)
    beta: jnp.ndarray       # blade twist angle (rad)


    # Inflow / rotor state
    V_x: jnp.ndarray        # axial inflow velocity (ms^{-1})
    V_yz: jnp.ndarray       # tangential / side inflow velocity (ms^{-1})
    omega: jnp.ndarray      # rotational speed (rad/s)

    V_ia_0: jnp.ndarray     # initial guess/storage for V_ia_0 (ms^{-1})

    # Physical properties
    rho: float = 1.225      # air density (kgm^{-3})

def blade_design_mask(params: Params):
    """
    Mask specifying which Params fields are optimized.
    True  -> optimized
    False -> frozen
    """
    return Params(
        # Domain of integration
        r=False,
        dr=False,
        psi=False,
        dpsi=False,

        # Geometry & discretization (DESIGN VARIABLES)
        N=False,
        R=False,
        c=True,
        beta=True,
        # Inflow / rotor state
        V_x=False,
        V_yz=False,
        omega=False,
        V_ia_0=False,

        # Physical properties
        rho=False
    )


