''' solver/physics.py '''
import numpy as np

def cons_to_prim(rho, rhou, rhov, rhoE, gamma, R):
    """ 
    Converts conserved state to primitive. 
    Handles both scalar and vectorized (NumPy array) inputs.
    """
    u = rhou / rho
    v = rhov / rho
    # Internal Energy = Total Energy - Kinetic Energy
    ke = 0.5 * rho * (u**2 + v**2)
    p = (gamma - 1.0) * (rhoE - ke)
    T = p / (rho * R)
    
    return rho, u, v, p, T

def prim_to_cons(rho, u, v, p, T, gamma, R):
    """ Converts primitive state to conserved. """
    rhou = rho * u
    rhov = rho * v
    # Total Energy = p/(gamma-1) + 1/2 * rho * V^2
    rhoE = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    
    return rho, rhou, rhov, rhoE