''' solver/steady_state.py '''
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_diffusion(face_cells, face_areas, d_PN, bc_groups, cell_volumes, k=1.0, bc_values=None):
    """
    Assembles and solves the Laplacian: div(k grad T) = 0.
    In Axisymmetric mode, face_areas = L * 2 * pi * r.
    """
    n_cells = len(cell_volumes)
    A = lil_matrix((n_cells, n_cells))
    b = np.zeros(n_cells)
    
    # 1. Internal Face Fluxes
    internal_mask = face_cells[:, 1] != -1
    for f_idx in np.where(internal_mask)[0]:
        P, N = face_cells[f_idx]
        # Flux = k * Area * (T_N - T_P) / d_PN
        coeff = (k * face_areas[f_idx]) / d_PN[f_idx]
        A[P, P] -= coeff; A[P, N] += coeff
        A[N, N] -= coeff; A[N, P] += coeff

    # 2. Boundary Conditions
    if bc_values is not None:
        for tag, T_val in bc_values.items():
            if tag in bc_groups:
                for f_idx in bc_groups[tag]:
                    P = face_cells[f_idx, 0]
                    # Boundary flux uses distance from cell center to face midpoint
                    # which is already captured in d_PN for boundary faces.
                    coeff = (k * face_areas[f_idx]) / d_PN[f_idx]
                    A[P, P] -= coeff
                    b[P]    -= coeff * T_val

    return spsolve(A.tocsr(), b)