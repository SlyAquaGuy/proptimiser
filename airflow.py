from dataclasses import replace
import jax.numpy as jnp
from init import Inputs
from jax import lax

##  Dimensionless Constants 
def calc_mach(V, a):
    ''' 
    
    Computes the Mach number as the ratio of velocity to speed of sound.
    '''
    return V / a

def calc_reynolds(rho, V, c, mu):
    ''' 
    Computes the Reynolds number for fluid flow.
    '''
    return (rho * V * c) / mu

def calc_phi(V_ia, params):
    ''' 
    Computes the inflow angle phi based on local velocity components.
    '''
    V_x, V_yz, omega, r, psi = params.V_x, params.V_yz, params.omega, params.r, params.psi
    V_yz_perp = V_yz * jnp.sin(psi)
    return jnp.arctan((V_x + V_ia) / (omega * r + V_yz_perp))

##  Inflow Model Definitions (0: Simple, 1: Pitt-Peters, 2: DTU) 
def simple(V_ia_0, params):
    ''' 
    Applies a simple uniform inflow model across all azimuthal angles.
    '''
    return jnp.tile(V_ia_0, (1, Inputs.n_psi))

def pitt_peters(V_ia_0, params):
    ''' 
    
    Computes skewed rotor inflow using the Pitt & Peters (1981) model.
    '''
    V_x, V_yz, r, psi, R = params.V_x, params.V_yz, params.r, params.psi, params.R
    xi = jnp.arctan(V_yz / (V_x + V_ia_0))  # wake skew angle
    return V_ia_0 * (1 + ((15 * jnp.pi) / 32) * jnp.tan(xi / 2) * (r / R) * jnp.cos(psi))

def dtu_model(V_ia_0, params):
    ''' 
    Placeholder for the DTU inflow model; currently defaults to uniform inflow.
    '''
    return jnp.tile(V_ia_0, (1, Inputs.n_psi))

def inflow_model(V_ia_0, params):
    ''' 
    Switches between inflow models based on global configuration.
    '''
    return lax.switch(Inputs.inflow_model, [simple, pitt_peters, dtu_model], V_ia_0, params)

##  Tip and Root Loss Models (0: No loss, 1: Prandtl) 
def no_loss(params):
    ''' 
    Returns a unity loss factor (no loss applied).
    '''
    return jnp.ones(Inputs.n_r)[:, None]

def prandtl_tip_loss(params):
    ''' 
    Placeholder for the Prandtl tip loss model. 
    '''
    return jnp.ones(Inputs.n_r)[:, None]

def prandtl_root_loss(params):
    ''' 
    Placeholder for the Prandtl root loss model.
    '''
    return jnp.ones(Inputs.n_r)[:, None]

def tip_root_loss_factors(params):
    ''' 
    Computes and returns both tip and root loss factors via model selection.
    '''
    tip_loss = lax.switch(Inputs.tip_loss_model, [no_loss, prandtl_tip_loss], params)
    root_loss = lax.switch(Inputs.root_loss_model, [no_loss, prandtl_root_loss], params)
    return tip_loss, root_loss