''' solver/steady_state.py '''
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_diffusion(face_cells, face_lengths, d_PN, bc_groups, k=1.0, T_wall=500.0):
    """
    Assembles and solves the Laplacian: div(k grad T) = 0.
    Currently assumes Tag 1 is a Dirichlet boundary (Fixed Temperature).
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

    # 2. Boundary Conditions (Tag 1: Dirichlet)
    if 1 in bc_groups:
        for f_idx in bc_groups[1]:
            P = face_cells[f_idx, 0]
            coeff = (k * face_lengths[f_idx]) / d_PN[f_idx]
            
            A[P, P] -= coeff
            b[P]    -= coeff * T_wall

    return spsolve(A.tocsr(), b)