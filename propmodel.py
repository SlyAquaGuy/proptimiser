## Define All Aerodynamic Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax

import optax

import jaxopt

import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

from dataclasses import replace

# Internal Libraries
from init import Inputs, Params
from airfoil import aero_coeffs
from airflow import tip_root_loss_factors, inflow_model
#from solvers import NewtonSolver


### Integrand Functions
def radial_stops(R, r_hub, n):
    # Return radial node centers and dr for integration
    r = jnp.linspace(r_hub + (R - r_hub) / (2 * n), R - (R - r_hub) / (2 * n), n)
    # Simple Homogeneous radial spacing
    dr = jnp.ones_like(r)*(R - r_hub) / n
    return r, dr

def angular_stops(n_psi):
    # Return angular node centers and dpsi for integration
    psi = jnp.linspace((jnp.pi / n_psi), (2 * jnp.pi) - (jnp.pi / n_psi), n_psi)
    dpsi = jnp.ones_like(psi)*(2 * jnp.pi) / n_psi
    return psi, dpsi

def thrust_residual(V_ia_0, tr_params):
    """
    Calculate thrust residual for an annular element at radius r.
    """
    # Unpack parameters
    N, c, r, psi, dpsi = tr_params.N, tr_params.c[:,None], tr_params.r[:,None], tr_params.psi[None,:], tr_params.dpsi[None,:]
    V_x, V_yz, omega = tr_params.V_x, tr_params.V_yz, tr_params.omega
    rho = tr_params.rho

    # Calculate Local Flow Properties
    V_yz_perp = V_yz * jnp.sin(psi)                                     # Perpendicular component of velocity in yz plane
    V_ia = inflow_model(V_ia_0, tr_params)                              # Calculate Inflow Velocity using selected mode
    phi = jnp.arctan((V_x + V_ia)/(omega * r + V_yz_perp))              # Angle of attack
    V_rel = jnp.sqrt((V_x + V_ia) ** 2 + (omega * r + V_yz_perp) ** 2)  # relative velocity
    # Get Aerodynamic Coefficients
    C_L, C_D, C_M = aero_coeffs(V_ia, tr_params)

    # Tip and Root Loss Factors
    tip_loss, root_loss = tip_root_loss_factors(tr_params)

    # Momentum Theory about annular element 
    dT_momentum = (2 * rho * r * V_ia * jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2))  
    # Blade Element Theory about annular element
    dT_blade =( 
        (N / (4 * jnp.pi)) * rho 
        * c * V_rel ** 2 
        * (C_L * jnp.cos(phi) + C_D * jnp.sin(phi))
        # tip and root loss factors
        * tip_loss[:,None] * root_loss[:,None]
        )
    # integrate thrust residuals about annulus
    res = jnp.trapezoid((dT_momentum - dT_blade), psi, axis=1)[:,None]

    
    return res

def induced_velocity(params):
    V_ia_init = params.V_ia_0   # shape (Nr, 1)

    # Debug Check Initial Residuals
    res_check = thrust_residual(V_ia_init, params)
    print(res_check)

    print("residual shape", jnp.shape(res_check))
    print("Residual min/max:", res_check.min(), res_check.max())

    # Newton Solver for Induced Velocity
    solver = jaxopt.Broyden(fun=thrust_residual, maxiter=100, tol=Inputs.newton_eps, max_stepsize=0.8, implicit_diff=True)

    (V_ia_0, info) = solver.run(V_ia_init, params)

    # Debug Check Final Residuals
    if Inputs.verbose:
        res_check = thrust_residual(V_ia_0, params)
        print("Residuals Check", res_check)
    V_ia = inflow_model(V_ia_0, params)

    return V_ia

if __name__ == "__main__":
    ## Demo Cases of BEMT
    # single is single annulus solver
    # flat is simple flat blade solver
    # input allows importing of a json file 
    demo_solve = "flat"

    # Single Annulus Solver
    if demo_solve == "single":

        psi,dpsi = angular_stops(Inputs.n_psi)
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_debug_nans", True)
        #Update Parameters Class
        params = Params(
                    N = 2,
                    R = jnp.array([0.5]),
                    c = jnp.array([0.05]),
                    beta = jnp.array([jnp.radians(20)]),

                    r = jnp.array([0.3]),
                    dr = jnp.array([0.05]),

                    psi = psi,
                    dpsi = dpsi,

                    V_x = jnp.array([10.0]),
                    V_yz = jnp.array([3.0]),
                    omega = jnp.array([400]),

                    V_ia_0 = jnp.array([20.0])[:,None]
                )
        # Register dataclass as a PyTree
        jax.tree_util.register_dataclass(Params)
        print()

        # Solve
        V_ia = induced_velocity(params)


        # Plot Induced Velocity Distribution about singular annulus
        plt.plot(jnp.degrees(params.psi), V_ia[1])
        plt.title("Induced Velocity Distribution around Propeller Annulus")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Induced Velocity (m/s)")
        plt.show()

    elif demo_solve == "flat":
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_debug_nans", True)
        psi,dpsi = angular_stops(Inputs.n_psi)
        r,dr = radial_stops(0.5, 0.05, Inputs.n_r)
        V_ia_0 = jnp.full_like(r,jnp.array([10.0]))[:,None]

        #Update Parameters Class
        params = Params(
                    N = 2,
                    R = jnp.array([0.5]),
                    c = jnp.full_like(r, 0.01),
                    beta = jnp.full_like(r, jnp.radians(20)),

                    r = r,
                    dr = dr,

                    psi = psi,
                    dpsi = dpsi,

                    V_x = jnp.array([10.0]),
                    V_yz = jnp.array([2.0]),
                    omega = jnp.array([400]),

                    V_ia_0 = V_ia_0
                )
        # Register dataclass as a PyTree
        jax.tree_util.register_dataclass(Params)
        
        # Solve Induced Velocity
        V_ia = induced_velocity(params)  # shape (Nr, Npsi)
        
        # Create radial and azimuthal grids
        #r, theta = np.meshgrid(np.array(params.r), np.array(params.psi)), 

        # Plot heatmap
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plt.contourf(psi, r, V_ia, shading='auto', cmap='viridis')
        plt.colorbar(label='Induced Velocity [m/s]')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Induced Velocity over Propeller Disk')
        plt.show()