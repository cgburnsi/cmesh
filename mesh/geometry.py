''' mesh/geometry.py '''
import numpy as np

def compute_face_metrics(nodes, faces):
    """ Computes geometric properties for all faces in the mesh. """
    idx_n1 = faces['n1'] - 1
    idx_n2 = faces['n2'] - 1
    
    x1, y1 = nodes['x'][idx_n1], nodes['y'][idx_n1]
    x2, y2 = nodes['x'][idx_n2], nodes['y'][idx_n2]
    
    dx, dy = x2 - x1, y2 - y1
    lengths = np.sqrt(dx**2 + dy**2)
    safe_lengths = np.where(lengths == 0, 1.0, lengths)
    
    tx, ty = dx / safe_lengths, dy / safe_lengths
    nx, ny = ty, -tx
    
    return {
        'lengths': lengths,
        'midpoints': np.column_stack((0.5 * (x1 + x2), 0.5 * (y1 + y2))),
        'tangents':  np.column_stack((tx, ty)),
        'normals':   np.column_stack((nx, ny))
    }

def compute_triangle_quality(points, simplices):
    """ Computes the Radius-Edge Ratio (Q) for a set of triangles. """
    pts = points[simplices]
    A, B, C = pts[:, 0], pts[:, 1], pts[:, 2]

    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(A - C, axis=1)
    c = np.linalg.norm(A - B, axis=1)

    area = 0.5 * np.abs(A[:,0]*(B[:,1]-C[:,1]) + B[:,0]*(C[:,1]-A[:,1]) + C[:,0]*(A[:,1]-B[:,1]))
    s = (a + b + c) / 2.0
    denom = s * a * b * c
    return np.where(denom > 0, (8.0 * area**2) / denom, 0.0)



def compute_cell_metrics(points, cells, mode='axisymmetric'):
    """ 
    Calculates Volume, Centroid, and Planar Area for every cell.
    - Planar: Volume = Area * 1.0
    - Axisymmetric: Volume = Area * 2 * pi * y_centroid
    """
    pts = points[cells] 
    A, B, C = pts[:, 0], pts[:, 1], pts[:, 2]

    # 1. Standard Planar Area
    area = 0.5 * np.abs(A[:,0]*(B[:,1]-C[:,1]) + B[:,0]*(C[:,1]-A[:,1]) + C[:,0]*(A[:,1]-B[:,1]))
    
    # 2. Centroid
    centroid = (A + B + C) / 3.0
    y_centroid = centroid[:, 1]
    
    # 3. Scaling for Axisymmetry
    if mode == 'axisymmetric':
        volume = area * (2.0 * np.pi * y_centroid)
    else:
        volume = area 
    
    return volume, centroid, area


def compute_fvm_face_metrics(points, face_nodes, face_cells, cell_centroids, mode='axisymmetric'):
    """ 
    Calculates Face Areas and Oriented Normals.
    - Planar: Face Area = Length * 1.0
    - Axisymmetric: Face Area = Length * 2 * pi * y_midpoint
    """
    p1 = points[face_nodes[:, 0]]
    p2 = points[face_nodes[:, 1]]
    
    # 1. Length and Midpoint
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    lengths = np.sqrt(dx**2 + dy**2)
    midpoints = 0.5 * (p1 + p2)
    y_mid = midpoints[:, 1]
    
    # 2. Candidate Normal (dy, -dx)
    nx, ny = dy, -dx
    
    # 3. Orientation (Must point from Owner to Neighbor)
    owner_centroids = cell_centroids[face_cells[:, 0]]
    vec_owner_to_face = midpoints - owner_centroids
    dot = vec_owner_to_face[:, 0] * nx + vec_owner_to_face[:, 1] * ny
    flip = np.where(dot < 0, -1.0, 1.0)
    
    # 4. Scaling for Axisymmetry
    if mode == 'axisymmetric':
        # Surface Area of the frustum/ring
        face_areas = lengths * (2.0 * np.pi * y_mid)
    else:
        face_areas = lengths # Planar (unit depth)
        
    # Final Unit Normals
    nx_final = (nx * flip) / (lengths + 1e-12)
    ny_final = (ny * flip) / (lengths + 1e-12)
    
    return face_areas, np.column_stack((nx_final, ny_final)), midpoints

def compute_fvm_weights(face_cells, cell_centroids, face_midpoints):
    """
    Calculates the distances between cell centers and interpolation weights.
    Returns: d_PN (distance P to N), gx (interpolation weight), d_PN_vec
    """
    idx_p = face_cells[:, 0]
    idx_n = face_cells[:, 1]
    
    P = cell_centroids[idx_p]
    
    internal_mask = idx_n != -1
    N = np.zeros_like(P)
    N[internal_mask] = cell_centroids[idx_n[internal_mask]]
    N[~internal_mask] = face_midpoints[~internal_mask]
    
    d_PN_vec = N - P
    d_PN_mag = np.linalg.norm(d_PN_vec, axis=1)
    
    dist_Pf = np.linalg.norm(face_midpoints - P, axis=1)
    gx = dist_Pf / (d_PN_mag + 1e-12)
    
    return d_PN_mag, gx, d_PN_vec