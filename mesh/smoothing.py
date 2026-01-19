''' mesh/smoothing.py '''
import numpy as np
from .distance import project_points_to_specific_faces, project_points_to_boundary
from .containment import check_points_inside

def get_unique_edges(simplices):
    edges = np.vstack((simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)

def smooth_mesh(points, simplices, nodes, faces, n_sliding, sliding_face_indices, sizing_func, dt=0.2, niters=50):
    n_fixed = len(nodes)
    slide_start = n_fixed
    slide_end   = n_fixed + n_sliding
    
    for itr in range(niters):
        edges = get_unique_edges(simplices)
        idx1, idx2 = edges[:, 0], edges[:, 1]
        p1, p2 = points[idx1], points[idx2]
        
        # Spring Forces
        midpoints = (p1 + p2) / 2.0
        L_target = sizing_func(midpoints)
        
        diff = p1 - p2
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists = np.maximum(dists, 1e-10)
        
        F_mag = dists - L_target
        F_mag = np.where(F_mag > 0, F_mag, 0) # Only contract if too long
        
        v = diff / dists[:, None]
        force = v * F_mag[:, None]
        
        total_force = np.zeros_like(points)
        move = 0.2 * force
        np.add.at(total_force, idx2,  move)
        np.add.at(total_force, idx1, -move)
        
        points[n_fixed:] += total_force[n_fixed:] * dt
        
        # 1. LOCKED BOUNDARY
        if n_sliding > 0:
            sliders = points[slide_start : slide_end]
            snapped = project_points_to_specific_faces(sliders, sliding_face_indices, nodes, faces)
            points[slide_start : slide_end] = snapped
            
        # 2. LEAK CHECK (Inner nodes)
        inner_points = points[slide_end:]
        is_inside = check_points_inside(inner_points, nodes, faces)
        outside_mask = ~is_inside
        
        if np.any(outside_mask):
            # Only import/call this if you implemented the global search
            from .distance import project_points_to_boundary 
            leakers = inner_points[outside_mask]
            _, snapped = project_points_to_boundary(leakers, nodes, faces)
            points[slide_end:][outside_mask] = snapped

    return points