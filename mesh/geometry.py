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


def compute_triangle_quality(points, simplices):
    """
    Computes the Radius-Edge Ratio (Q) for a set of triangles.
    Q = 1.0 for equilateral, Q -> 0.0 for slivers.
    """
    # 1. Get coordinates for each vertex (A, B, C) of every triangle
    pts = points[simplices] # Shape: (N_tris, 3, 2)
    A, B, C = pts[:, 0], pts[:, 1], pts[:, 2]

    # 2. Compute side lengths
    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(A - C, axis=1)
    c = np.linalg.norm(A - B, axis=1)

    # 3. Compute Area (using cross product magnitude)
    # Area = 0.5 * |(x1(y2-y3) + x2(y3-y1) + x3(y1-y2))|
    area = 0.5 * np.abs(A[:,0]*(B[:,1]-C[:,1]) + B[:,0]*(C[:,1]-A[:,1]) + C[:,0]*(A[:,1]-B[:,1]))
    
    # 4. Compute Radius-Edge Ratio (Q)
    # r_in = Area / semi-perimeter; r_out = (abc) / (4 * Area)
    # Q = 2 * r_in / r_out simplified:
    s = (a + b + c) / 2.0
    # Avoid division by zero for degenerate triangles
    denom = s * a * b * c
    q_values = np.where(denom > 0, (8.0 * area**2) / denom, 0.0)
    
    return q_values