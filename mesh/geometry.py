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