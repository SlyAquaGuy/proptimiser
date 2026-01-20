# Airflow Definitions, Boundary Conditions for Momentum Theory
from dataclasses import replace
import jax.numpy as jnp
from init import Inputs
from jax import lax

## Dimensionless Constants
def calc_mach(V, a):
    # Calculate Mach number
    return V / a
def calc_reynolds(rho, V, c, mu):
    # Calculate Reynolds number
    return (rho * V * c) / mu

def calc_phi(V_ia, params):

    V_x, V_yz, omega, r, psi = params.V_x, params.V_yz, params.omega, params.r, params.psi
    V_yz_perp = V_yz * jnp.sin(psi)
    phi = jnp.arctan((V_x + V_ia)/(omega * r + V_yz_perp))
    
    return phi

## Inflow Model Definitions and Selection
# 0 - Simple uniform inflow
# 1 - Pitt & Peters (1981)
# 2 - DTU model

def simple(V_ia_0, params):
    n_psi = Inputs.n_psi
    # Simple uniform inflow model
    V_ia = jnp.tile(V_ia_0, (1, n_psi))
    return V_ia

def pitt_peters(V_ia_0, params):
    # Unpack parameters
    V_x, V_yz = params.V_x, params.V_yz
    r,psi,R = params.r, params.psi, params.R
    xi = jnp.arctan(V_yz / (V_x + V_ia_0))    # wake skew angle 
    if Inputs.verbose:
        print("Xi Shape", jnp.shape(xi))
        print("V_ia_0", jnp.shape(V_ia_0))
    # Inflow model for skewed rotor (Pitt & Peters 1981)
    V_ia = V_ia_0 * (1 + ((15 * jnp.pi) / 32) * jnp.tan(xi / 2) * (r / R) * jnp.cos(psi))
    return V_ia

def dtu_model(V_ia_0, params):
    # Unpack Parameters
    n_psi = Inputs.n_psi
    # Simple uniform inflow model
    V_ia = jnp.tile(V_ia_0, (1,n_psi))
    # Implement DTU model here to improve accuracy? /doi.org/10.5194/wes-5-1-2020
    return V_ia  # Placeholder, return simple guess for now

def inflow_model(V_ia_0, params):
    # Calculate inflow velocity based on selected model
    return lax.switch(Inputs.inflow_model,
                    [simple, pitt_peters, dtu_model],
                    V_ia_0, params
                    )


## Tip and Root Loss Model Definitions and Selection
# 0 - No loss
# 1 - Prandtl Loss

def no_loss(params):
    # No loss factor
    return jnp.ones(Inputs.n_r)[:,None]

def prandtl_tip_loss(params):
    # Calculate Prandtl tip loss factor
    return jnp.ones(Inputs.n_r)[:,None]  # Placeholder, implement actual model later

def prandtl_root_loss(params):
    # Calculate Prandtl root loss factor
    return jnp.ones(Inputs.n_r)[:,None]  # Placeholder, implement actual model later

def tip_root_loss_factors(params):
    # Calculate tip and root loss factors based on selected models
    tip_loss = lax.switch(Inputs.tip_loss_model,
                      [no_loss, prandtl_tip_loss],
                      params)
    
    root_loss = lax.switch(Inputs.root_loss_model,
                      [no_loss, prandtl_root_loss],
                        params)

    return tip_loss, root_loss 





