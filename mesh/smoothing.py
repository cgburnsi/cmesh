''' mesh/smoothing.py '''
import numpy as np
from scipy.spatial import Delaunay
from .distance import project_points_to_specific_faces, project_points_to_boundary
from .containment import check_points_inside

def get_unique_edges(simplices):
    """ Extracts unique edges from a set of simplices. """
    edges = np.vstack((simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)

def spring_smoother(points, nodes, faces, n_sliding, sliding_face_indices, sizing_func, niters=100, constraints=None, hf_segments=None):
    """ Standard spring-based smoother with axisymmetry constraints. """
    n_fixed = len(nodes)
    slide_start, slide_end = n_fixed, n_fixed + n_sliding
    dt, retriangulate_interval = 0.2, 5 
    current_simplices = None

    for itr in range(niters):
        if itr % retriangulate_interval == 0:
            tri = Delaunay(points)
            mask = check_points_inside(np.mean(points[tri.simplices], axis=1), hf_segments)
            current_simplices = tri.simplices[mask]
            
        edges = get_unique_edges(current_simplices)
        idx1, idx2 = edges[:, 0], edges[:, 1]
        p1, p2 = points[idx1], points[idx2]
        
        diff = p1 - p2
        dists = np.maximum(np.sqrt(np.sum(diff**2, axis=1)), 1e-10)
        F_mag = dists - sizing_func((p1 + p2) / 2.0)
        
        force = (diff / dists[:, None]) * F_mag[:, None]
        total_force = np.zeros_like(points)
        np.add.at(total_force, idx2,  0.2 * force)
        np.add.at(total_force, idx1, -0.2 * force)
        
        raw_move = total_force * dt
        h_local = sizing_func(points)
        move_mags = np.linalg.norm(raw_move, axis=1)
        dist_to_boundary, _ = project_points_to_boundary(points, nodes, faces)
        limit = np.minimum(0.2 * h_local, 0.5 * dist_to_boundary)
        
        scale = np.where(move_mags > limit, limit / (move_mags + 1e-12), 1.0)
        points[n_fixed:] += raw_move[n_fixed:] * scale[n_fixed:][:, np.newaxis]
        
        # --- AXISYMMETRY CONSTRAINT ---
        # Prevent interior points from drifting across the symmetry axis
        points[n_fixed:, 1] = np.maximum(points[n_fixed:, 1], 1e-10)
        
        if n_sliding > 0:
            points[slide_start : slide_end] = project_points_to_specific_faces(
                points[slide_start : slide_end], sliding_face_indices, nodes, faces, constraints=constraints
            )
    return points

def distmesh_smoother(points, nodes, faces, n_sliding, sliding_face_indices, sizing_func, niters=100, constraints=None, hf_segments=None):
    """ DistMesh-style smoother with axisymmetry constraints. """
    n_fixed = len(nodes)
    slide_start, slide_end = n_fixed, n_fixed + n_sliding
    dt, retriangulate_interval = 0.2, 5
    current_simplices = None

    for itr in range(niters):
        if itr % retriangulate_interval == 0:
            tri = Delaunay(points)
            mask = check_points_inside(np.mean(points[tri.simplices], axis=1), hf_segments)
            current_simplices = tri.simplices[mask]

        edges = get_unique_edges(current_simplices)
        idx1, idx2 = edges[:, 0], edges[:, 1]
        p1, p2 = points[idx1], points[idx2]

        diff = p1 - p2
        dists = np.maximum(np.sqrt(np.sum(diff**2, axis=1)), 1e-10)
        L_target = sizing_func((p1 + p2) / 2.0)

        F_mag = np.maximum(L_target - dists, 0)
        spring_vec = (diff / dists[:, None]) * F_mag[:, None]
        
        total_force = np.zeros_like(points)
        np.add.at(total_force, idx1,  spring_vec)
        np.add.at(total_force, idx2, -spring_vec)

        raw_move = (total_force * 0.7) * dt
        h_local = sizing_func(points)
        move_mags = np.linalg.norm(raw_move, axis=1)
        dist_to_boundary, _ = project_points_to_boundary(points, nodes, faces)
        limit = np.minimum(0.1 * h_local, 0.5 * dist_to_boundary)
        
        scale = np.where(move_mags > limit, limit / (move_mags + 1e-12), 1.0)
        points[n_fixed:] += raw_move[n_fixed:] * scale[n_fixed:][:, np.newaxis]

        # --- AXISYMMETRY CONSTRAINT ---
        # Ensure interior points stay above the centerline
        points[n_fixed:, 1] = np.maximum(points[n_fixed:, 1], 1e-10)

        if n_sliding > 0:
            points[slide_start : slide_end] = project_points_to_specific_faces(
                points[slide_start : slide_end], sliding_face_indices, nodes, faces, constraints=constraints
            )
    return points