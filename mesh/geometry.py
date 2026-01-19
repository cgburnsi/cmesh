''' mesh/geometry.py '''
import numpy as np

def compute_face_metrics(nodes, faces):
    """
    Computes geometric properties for all faces in the mesh.
    
    Returns a dictionary containing:
        - lengths: (N_faces,) array
        - midpoints: (N_faces, 2) array
        - tangents: (N_faces, 2) unit vectors
        - normals: (N_faces, 2) unit vectors (Assuming Right-Hand Rule)
    """
    # 1. Map Face IDs (1-based) to Node Indices (0-based)
    #    We check if the input is a structured array or just indices
    #    to support both direct calls and your data dict.
    idx_n1 = faces['n1'] - 1
    idx_n2 = faces['n2'] - 1
    
    # 2. Vectorized Fetch of Coordinates
    #    P1 = (x1, y1), P2 = (x2, y2)
    x1 = nodes['x'][idx_n1]
    y1 = nodes['y'][idx_n1]
    
    x2 = nodes['x'][idx_n2]
    y2 = nodes['y'][idx_n2]
    
    # 3. Compute Delta Vectors (The Edge Vector)
    dx = x2 - x1
    dy = y2 - y1
    
    # 4. Compute Lengths
    lengths = np.sqrt(dx**2 + dy**2)
    
    # Prevent division by zero for degenerate edges
    # (replaces 0.0 with 1.0 just for division, doesn't change length)
    safe_lengths = np.where(lengths == 0, 1.0, lengths)
    
    # 5. Compute Midpoints (Average of x, Average of y)
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    
    # 6. Compute Unit Tangents and Normals
    #    Tangent = (dx, dy) / L
    tx = dx / safe_lengths
    ty = dy / safe_lengths
    
    #    Normal = (dy, -dx)  <-- Rotated 90 degrees CW (Standard "Outward" for CCW Loop)
    nx = ty
    ny = -tx
    
    return {
        'lengths': lengths,
        'midpoints': np.column_stack((mid_x, mid_y)),
        'tangents':  np.column_stack((tx, ty)),
        'normals':   np.column_stack((nx, ny))
    }