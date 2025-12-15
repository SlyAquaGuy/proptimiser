## Define All Aerodynamic Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax
import matplotlib.pyplot as plt

from dataclasses import replace

# Internal Libraries
from init import Inputs, Params
from airfoil import aero_coeffs
from airflow import tip_root_loss_factors, inflow_model
from solvers import NewtonSolver


### Integrand Functions
def radial_stops(R, r_hub, n):
    # Return radial node centers and dr for integration
    r = jnp.linspace(r_hub + (R - r_hub) / (2 * n), R - (R - r_hub) / (2 * n), n)
    dr = (R - r_hub) / n
    return r, dr

def angular_stops(n_psi):
    # Return angular node centers and dpsi for integration
    psi = jnp.linspace((jnp.pi / n_psi), (2 * jnp.pi) - (jnp.pi / n_psi), n_psi)
    dpsi = (2 * jnp.pi) / n_psi
    return psi, dpsi

def thrust_residual(tr_params):
    """
    Calculate thrust residual for an annular element at radius r.
    """
    # Unpack parameters
    N, c, r, dr, psi, dpsi = tr_params.N, tr_params.c, tr_params.r, tr_params.dr, tr_params.psi, tr_params.dpsi, 
    V_x, V_yz, omega = tr_params.V_x, tr_params.V_yz, tr_params.omega
    rho = tr_params.rho

    # Calculate Local Flow Properties
    V_yz_perp = V_yz * jnp.sin(psi)         # Perpendicular component of velocity in yz plane
    V_ia = inflow_model(tr_params)          # Calculate Inflow Velocity using selected mode
    phi = jnp.arctan((V_x+V_ia)/(omega * r + V_yz_perp))         # Angle of attack
    V_rel = jnp.sqrt((V_x + V_ia) ** 2 + (omega * r + V_yz_perp) ** 2)  # relative velocity
    
    # Update Parameters with updated phi
    tr_params = replace(tr_params, phi=phi)

    # Get Aerodynamic Coefficients
    C_L, C_D, C_M = aero_coeffs(tr_params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(tr_params)

    # Momentum Theory about annular element 
    dT_momentum = (2 * rho * r * V_ia * jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2) * dpsi * dr)  
    # Blade Element Theory about annular element
    dT_blade =( 
        (N / (4 * jnp.pi)) * rho * c * V_rel ** 2 
        * (C_L * jnp.cos(phi) + C_D * jnp.sin(phi))* dpsi* dr
        # tip and root loss factors
        * tip_loss * root_loss
        )
    res = jnp.sum(dT_momentum - dT_blade)

    return jnp.atleast_1d(res)


def solve_annulus(params):
    # Unpack Initial Guess
    V_ia_0_init = params.V_ia_0

    # Minimise Thrust Residual to Solve for Induced Velocity

    def thrust_residual_fun(V_ia_guess, tr_params):
        # Update params with current V_ia_0
        tr_params_updated = replace(tr_params, V_ia_0=V_ia_guess)
        # Compute Residual
        return thrust_residual(tr_params_updated)

    solver = NewtonSolver(thrust_residual_fun)

    # Test Case Single Annulus
    V_ia_solution = solver.solve(V_ia_0_init, params)
    
    # Plot Induced Velocity Distribution about singular annulus
    plt.plot(jnp.degrees(params.psi), inflow_model(params))
    plt.title("Induced Velocity Distribution around Propeller Annulus")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Induced Velocity (m/s)")
    plt.show()

    return V_ia_solution 

if __name__ == "__main__":
    # Single Annulus Solver
    psi,dpsi = angular_stops(Inputs.n_psi)

    #Update Parameters Class
    params = Params(
                N = 2,
                R = jnp.array([0.5]),
                c = jnp.array([0.01]),
                beta = jnp.array([jnp.radians(20)]),

                r = jnp.array([0.3]),
                dr = jnp.array([0.05]),

                psi = psi,
                dpsi = dpsi,

                V_x = jnp.array([10.0]),
                V_yz = jnp.array([3.0]),
                omega = jnp.array([400]),

                V_ia_0 = jnp.array([5.0]),
                phi = jnp.array([0])
            )
    # Register dataclass as a PyTree
    jax.tree_util.register_dataclass(Params)
    print()

    # Solve
    solve_annulus(params)





