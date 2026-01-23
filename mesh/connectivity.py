''' mesh/connectivity.py '''
import numpy as np

def build_fvm_connectivity(cells):
    """
    Builds robust FVM connectivity arrays using sorted occurrence mapping.
    """
    n_cells = len(cells)
    
    # 1. Gather all potential edges (3 per cell)
    all_edges = np.vstack((
        cells[:, [0, 1]],
        cells[:, [1, 2]],
        cells[:, [2, 0]]
    ))
    all_edges.sort(axis=1)
    
    # 2. Extract unique faces and counts
    unique_faces, inverse, counts = np.unique(
        all_edges, axis=0, return_inverse=True, return_counts=True
    )
    
    n_faces = len(unique_faces)
    face_nodes = unique_faces
    
    # 3. Build Face-to-Cell mapping
    face_cells = np.full((n_faces, 2), -1, dtype=int)
    
    # Create an array matching every edge instance to its Cell ID
    all_cell_indices = np.tile(np.arange(n_cells), 3)
    
    # Sort cell indices by their Face ID to group neighbors together
    order = np.argsort(inverse)
    sorted_faces = inverse[order]
    sorted_cells = all_cell_indices[order]
    
    # Identify the first occurrence of every face in the sorted list
    _, first_indices = np.unique(sorted_faces, return_index=True)
    
    # Every face has at least one cell (the 'owner')
    face_cells[:, 0] = sorted_cells[first_indices]
    
    # Internal faces (count == 2) have a second cell (the 'neighbor')
    # This is always the next element in the sorted array
    internal_mask = (counts == 2)
    face_cells[internal_mask, 1] = sorted_cells[first_indices[internal_mask] + 1]

    # 4. Build Cell-to-Face mapping
    cell_faces = inverse.reshape((3, n_cells)).T
    
    return face_nodes, face_cells, cell_faces