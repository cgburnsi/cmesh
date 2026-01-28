''' solver/steady_state.py '''
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_diffusion(face_cells, face_areas, d_PN, bc_groups, cell_volumes, k=1.0, bc_values=None):
    """
    Assembles and solves the Laplacian: div(k grad T) = 0.
    """
    n_cells = len(cell_volumes)
    A = lil_matrix((n_cells, n_cells))
    b = np.zeros(n_cells)
    
    for f_idx in np.where(face_cells[:, 1] != -1)[0]:
        P, N = face_cells[f_idx]
        coeff = (k * face_areas[f_idx]) / d_PN[f_idx]
        A[P, P] -= coeff; A[P, N] += coeff
        A[N, N] -= coeff; A[N, P] += coeff

    if bc_values is not None:
        for tag, T_val in bc_values.items():
            if tag in bc_groups:
                for f_idx in bc_groups[tag]:
                    P = face_cells[f_idx, 0]
                    coeff = (k * face_areas[f_idx]) / d_PN[f_idx]
                    A[P, P] -= coeff
                    b[P]    -= coeff * T_val

    return spsolve(A.tocsr(), b)

def solve_advection_diffusion(face_cells, face_areas, face_normals, d_PN, bc_groups, 
                              cell_volumes, u_field, k=1.0, bc_values=None):
    """
    Solves the steady-state Advection-Diffusion equation with Upwind discretization.
    """
    n_cells = len(cell_volumes)
    A = lil_matrix((n_cells, n_cells))
    b = np.zeros(n_cells)
    
    # 1. Internal Face Fluxes
    internal_mask = face_cells[:, 1] != -1
    for f_idx in np.where(internal_mask)[0]:
        P, N = face_cells[f_idx]
        diff_coeff = (k * face_areas[f_idx]) / d_PN[f_idx]
        
        # Convective mass flux (u . n) * Area
        u_face = u_field[f_idx]
        mass_flux = np.dot(u_face, face_normals[f_idx]) * face_areas[f_idx]
        
        # Matrix Contributions
        A[P, P] -= diff_coeff; A[P, N] += diff_coeff
        A[N, N] -= diff_coeff; A[N, P] += diff_coeff
        
        if mass_flux > 0: # Out of P into N
            A[P, P] -= mass_flux; A[N, P] += mass_flux
        else: # Out of N into P
            A[N, N] += mass_flux; A[P, N] -= mass_flux

    # 2. Boundary Conditions
    if bc_values is not None:
        for tag, val in bc_values.items():
            if tag in bc_groups:
                for f_idx in bc_groups[tag]:
                    P = face_cells[f_idx, 0]
                    diff_coeff = (k * face_areas[f_idx]) / d_PN[f_idx]
                    A[P, P] -= diff_coeff
                    b[P]    -= diff_coeff * val
                    
                    u_face = u_field[f_idx]
                    mass_flux = np.dot(u_face, face_normals[f_idx]) * face_areas[f_idx]
                    if mass_flux > 0: # Outflow
                        A[P, P] -= mass_flux
                    else: # Inflow
                        b[P] -= mass_flux * val

    return spsolve(A.tocsr(), b)