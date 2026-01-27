''' solver/steady_state.py '''
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_diffusion(face_cells, face_lengths, d_PN, bc_groups, k=1.0, bc_values=None):
    """
    Assembles and solves the Laplacian: div(k grad T) = 0 with multiple BCs.
    bc_values: Dictionary {tag: temperature}
    """
    n_cells = np.max(face_cells) + 1
    A = lil_matrix((n_cells, n_cells))
    b = np.zeros(n_cells)
    
    # 1. Internal Face Fluxes
    internal_mask = face_cells[:, 1] != -1
    for f_idx in np.where(internal_mask)[0]:
        P, N = face_cells[f_idx]
        coeff = (k * face_lengths[f_idx]) / d_PN[f_idx]
        A[P, P] -= coeff; A[P, N] += coeff
        A[N, N] -= coeff; A[N, P] += coeff

    # 2. Multi-Tag Boundary Conditions
    if bc_values is not None:
        for tag, T_val in bc_values.items():
            if tag in bc_groups:
                for f_idx in bc_groups[tag]:
                    P = face_cells[f_idx, 0]
                    coeff = (k * face_lengths[f_idx]) / d_PN[f_idx]
                    A[P, P] -= coeff
                    b[P]    -= coeff * T_val

    return spsolve(A.tocsr(), b)