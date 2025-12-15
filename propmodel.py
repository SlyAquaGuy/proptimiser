## Define All Aerodynamic Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax

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

def thrust_residual(tr_params):
    """
    Calculate thrust residual for an annular element at radius r.
    """
    # Unpack parameters
    N, c, r, dr, psi, dpsi = tr_params.N, tr_params.c[:,None], tr_params.r[:,None], tr_params.dr[:,None], tr_params.psi[None,:], tr_params.dpsi[None,:], 
    V_x, V_yz, omega = tr_params.V_x, tr_params.V_yz, tr_params.omega
    rho = tr_params.rho

    # Calculate Local Flow Properties
    V_yz_perp = V_yz * jnp.sin(psi)                                     # Perpendicular component of velocity in yz plane
    V_ia = inflow_model(tr_params)                                      # Calculate Inflow Velocity using selected mode
    phi = jnp.arctan((V_x + V_ia)/(omega * r + V_yz_perp))              # Angle of attack
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
        (N / (4 * jnp.pi)) * rho 
        * c * V_rel ** 2 
        * (C_L * jnp.cos(phi) + C_D * jnp.sin(phi))* dpsi* dr
        # tip and root loss factors
        * tip_loss[:,None] * root_loss[:,None]
        )
    res = dT_momentum - dT_blade
    return res

def induced_velocity(params):
    V_ia_init = params.V_ia   # shape (Nr, Npsi)

    # Make Thrust Residual Function of V_ia to Solve for Induced Velocity
    def thrust_residual_fun(V_ia_guess):
        # Update params with current V_ia_0
        tr_params_updated = replace(params, V_ia_0=V_ia_guess)
        # Compute Residual
        return jnp.sum(thrust_residual(tr_params_updated), axis=1)
    # Check Residuals
    res_check = thrust_residual_fun(V_ia_init)
    print("Residual min/max:", res_check.min(), res_check.max())
    # Use Newton Method to Find V_ia that Satisfies Thrust Residual Function

    solver = jaxopt.Broyden(fun=thrust_residual_fun, maxiter=Inputs.newton_max_iter, tol=Inputs.opt_tol, max_stepsize=Inputs.newton_damping)
    
    (V_ia_solution, info) = solver.run(V_ia_init)
    print("V_ia Solution:", V_ia_solution)
    return V_ia_solution
if __name__ == "__main__":
    ## Demo Cases of BEMT
    # single is single annulus solver
    # flat is simple flat blade solver
    # input allows importing of a json file 
    demo_solve = "flat"


    # Single Annulus Solver
    if demo_solve == "single":

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

                    V_ia_0 = jnp.array([0.1]),
                    phi = jnp.array([0])
                )
        # Register dataclass as a PyTree
        jax.tree_util.register_dataclass(Params)
        print()

        # Solve
        solve_annulus(params)
        # Plot Induced Velocity Distribution about singular annulus
        plt.plot(jnp.degrees(params.psi), inflow_model(params))
        plt.title("Induced Velocity Distribution around Propeller Annulus")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Induced Velocity (m/s)")
        plt.show()

    elif demo_solve == "flat":
        
        psi,dpsi = angular_stops(Inputs.n_psi)
        r,dr = radial_stops(0.5, 0.05, Inputs.n_r)
        V_ia_0 = jnp.full_like(r,jnp.array([5.0]))

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
                    V_yz = jnp.array([0]),
                    omega = jnp.array([400]),

                    V_ia_0 = V_ia_0,
                    V_ia = jnp.ones((Inputs.n_r, Inputs.n_psi))*V_ia_0[:, None],
                    phi = jnp.array([0])
                )
        # Register dataclass as a PyTree
        jax.tree_util.register_dataclass(Params)
        
        # Solve Induced Velocity
        V_ia_0 = induced_velocity(params)  # shape (Nr, Npsi)
        print(V_ia_0)
        

        # Create radial and azimuthal grids
        r = np.array(params.r)           # shape (Nr,)
        psi = np.array(params.psi)       # shape (Npsi,)
        R, PSI = np.meshgrid(r, psi, indexing='ij')  # shape (Nr, Npsi)

        # Convert polar to Cartesian
        X = R * np.cos(PSI)
        Y = R * np.sin(PSI)

        # Flatten grids for interpolation
        points = np.column_stack([X.ravel(), Y.ravel()])       # (Nr*Npsi, 2)
        values = np.array(V_ia).ravel()                        # (Nr*Npsi,)

        # Create Cartesian grid for smooth circular plot
        grid_size = 200
        xi = np.linspace(-r[-1], r[-1], grid_size)
        yi = np.linspace(-r[-1], r[-1], grid_size)
        XI, YI = np.meshgrid(xi, yi)

        # Interpolate induced velocity onto Cartesian grid
        ZI = griddata(points, values, (XI, YI), method='cubic')

        # Mask points outside propeller radius
        mask = XI**2 + YI**2 > r[-1]**2
        ZI[mask] = np.nan

        # Plot heatmap
        plt.figure(figsize=(6,6))
        plt.pcolormesh(XI, YI, ZI, shading='auto', cmap='viridis')
        plt.colorbar(label='Induced Velocity [m/s]')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Induced Velocity over Propeller Disk')
        plt.axis('equal')
        plt.show()






