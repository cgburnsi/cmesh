''' solver/fluxes.py '''
import numpy as np
from .physics import cons_to_prim, prim_to_cons

def compute_rusanov_flux(QL, QR, normal, area, gamma, R):
    """ 
    Computes the Rusanov (LLF) inviscid flux across a face.
    normal: Unit normal vector [nx, ny]
    """
    # 1. Get Primitives
    rhoL, uL, vL, pL, TL = cons_to_prim(QL, gamma, R)
    rhoR, uR, vR, pR, TR = cons_to_prim(QR, gamma, R)
    
    vnL = uL * normal[0] + vL * normal[1]
    vnR = uR * normal[0] + vR * normal[1]
    
    # 2. Physical Fluxes
    FL = np.array([
        rhoL * vnL,
        rhoL * uL * vnL + pL * normal[0],
        rhoL * vL * vnL + pL * normal[1],
        (QL['rhoE'] + pL) * vnL
    ])
    
    FR = np.array([
        rhoR * vnR,
        rhoR * uR * vnR + pR * normal[0],
        rhoR * vR * vnR + pR * normal[1],
        (QR['rhoE'] + pR) * vnR
    ])
    
    # 3. Maximum Wave Speed (Stability)
    aL, aR = np.sqrt(gamma * R * TL), np.sqrt(gamma * R * TR)
    smax = max(abs(vnL) + aL, abs(vnR) + aR)
    
    # 4. Rusanov Dissipation
    return 0.5 * (FL + FR) * area - 0.5 * smax * area * (QR.view('f8') - QL.view('f8'))