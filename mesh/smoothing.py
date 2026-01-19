''' mesh/smoothing.py '''
import numpy as np
from scipy.spatial import Delaunay
from .distance import project_points_to_specific_faces, project_points_to_boundary
from .containment import check_points_inside

def get_unique_edges(simplices):
    edges = np.vstack((simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)

def smooth_mesh(points, nodes, faces, n_sliding, sliding_face_indices, sizing_func, niters=80):
    """
    Now performs Dynamic Re-Triangulation!
    Note: We removed 'simplices' from arguments because we calculate them internally.
    """
    n_fixed = len(nodes)
    slide_start = n_fixed
    slide_end   = n_fixed + n_sliding
    
    # Parameters for stability
    dt = 0.2
    
    # We don't need to triangulate EVERY step (expensive). 
    # Doing it every 10 steps is usually enough to let points migrate.
    retriangulate_interval = 5 
    
    current_simplices = None
    
    for itr in range(niters):
        
        # --- 1. DYNAMIC RE-TRIANGULATION ---
        # If the points have moved, the old connections are stale. Rebuild them.
        if itr % retriangulate_interval == 0:
            tri = Delaunay(points)
            simplices = tri.simplices
            
            # We MUST filter out the triangles inside the concave notch/void
            # otherwise springs will pull nodes across the empty space.
            centroids = np.mean(points[simplices], axis=1)
            mask = check_points_inside(centroids, nodes, faces)
            current_simplices = simplices[mask]
            
        # --- 2. CALCULATE FORCES ---
        edges = get_unique_edges(current_simplices)
        idx1, idx2 = edges[:, 0], edges[:, 1]
        p1, p2 = points[idx1], points[idx2]
        
        # Desired Length (average of both ends)
        midpoints = (p1 + p2) / 2.0
        L_target = sizing_func(midpoints)
        
        # Current Length
        diff = p1 - p2
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists = np.maximum(dists, 1e-10)
        
        # Spring Force: F = L_current - L_target
        # (If L_current > L_target, pull them closer)
        F_mag = dists - L_target
        
        # We allow push AND pull now to let the lattice relax fully
        # (Previously we only allowed contraction, but expansion helps quality too)
        
        v = diff / dists[:, None]
        force = v * F_mag[:, None]
        
        total_force = np.zeros_like(points)
        move = 0.2 * force
        np.add.at(total_force, idx2,  move)
        np.add.at(total_force, idx1, -move)
        
        # --- 3. MOVE POINTS ---
        points[n_fixed:] += total_force[n_fixed:] * dt
        
        # --- 4. APPLY CONSTRAINTS ---
        
        # A. Locked Boundary Nodes
        if n_sliding > 0:
            sliders = points[slide_start : slide_end]
            snapped = project_points_to_specific_faces(sliders, sliding_face_indices, nodes, faces)
            points[slide_start : slide_end] = snapped
            
        # B. Inner Node Containment (The "Leaker" Check)
        # Only check this occasionally to save time, or every step if unstable
        if itr % 1 == 0:
            inner_points = points[slide_end:]
            is_inside = check_points_inside(inner_points, nodes, faces)
            outside_mask = ~is_inside
            
            if np.any(outside_mask):
                leakers = inner_points[outside_mask]
                _, snapped = project_points_to_boundary(leakers, nodes, faces)
                points[slide_end:][outside_mask] = snapped

    return points