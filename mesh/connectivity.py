''' mesh/connectivity.py '''
import numpy as np
from .distance import get_closest_point_on_segment


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



def map_boundary_faces(points, face_nodes, face_cells, input_nodes, input_faces):
    """
    Groups boundary face indices by their BC tags using geometric proximity.
    """
    # 1. Identify all boundary face indices (where neighbor cell is -1)
    boundary_indices = np.where(face_cells[:, 1] == -1)[0]
    
    # 2. Calculate midpoints of these boundary faces
    p1_coords = points[face_nodes[boundary_indices, 0]]
    p2_coords = points[face_nodes[boundary_indices, 1]]
    midpoints = 0.5 * (p1_coords + p2_coords)
    
    boundary_map = {}
    
    # Create a quick lookup for node coordinates by ID
    node_id_to_idx = {nid: i for i, nid in enumerate(input_nodes['id'])}
    
    # 3. Match each boundary face to an input segment geometrically
    for face in input_faces:
        tag = face['tag']
        
        # Get coordinates of the input segment endpoints
        idx1, idx2 = node_id_to_idx[face['n1']], node_id_to_idx[face['n2']]
        x1, y1 = input_nodes['x'][idx1], input_nodes['y'][idx1]
        x2, y2 = input_nodes['x'][idx2], input_nodes['y'][idx2]
        
        # Check proximity of all boundary midpoints to this segment
        cx, cy = get_closest_point_on_segment(midpoints[:, 0], midpoints[:, 1], x1, y1, x2, y2)
        
        # Squared distance to the segment
        dists_sq = (midpoints[:, 0] - cx)**2 + (midpoints[:, 1] - cy)**2
        
        # If the face is on this segment (tolerance of 1e-8)
        mask = dists_sq < 1e-8
        
        if np.any(mask):
            if tag not in boundary_map:
                boundary_map[tag] = []
            # Map the original face indices back
            boundary_map[tag].extend(boundary_indices[mask].tolist())
            
    return {k: np.array(v) for k, v in boundary_map.items()}