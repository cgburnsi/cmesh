''' mesh/initialization.py '''
import numpy as np
from .containment import check_points_inside

def resample_boundary_points(nodes, faces, sizing_field):
    """
    Distributes points along edges based on the 'segments' count in faces,
    weighted by the sizing_field density.
    """
    sliding_points = []
    sliding_face_ids = []
    
    for i, face in enumerate(faces):
        idx1 = face['n1'] - 1
        idx2 = face['n2'] - 1
        n_segments = face['segments']
        
        if n_segments <= 1: continue 
        
        p1 = np.array([nodes['x'][idx1], nodes['y'][idx1]])
        p2 = np.array([nodes['x'][idx2], nodes['y'][idx2]])
        
        # 1. Integrate the Sizing Field along the line
        n_samples = 100
        t_samples = np.linspace(0, 1, n_samples)
        sample_pts = p1 + (p2 - p1) * t_samples[:, np.newaxis]
        
        h_vals = sizing_field(sample_pts)
        density = 1.0 / np.maximum(h_vals, 1e-6)
        
        cumulative = np.cumsum(density)
        cumulative -= cumulative[0]
        if cumulative[-1] > 0:
            cumulative /= cumulative[-1]
        else:
            cumulative = t_samples 
        
        # 2. Pick target 't' values
        target_cdf = np.linspace(0, 1, n_segments + 1)[1:-1]
        t_final = np.interp(target_cdf, cumulative, t_samples)
        
        for t in t_final:
            pt = p1 + t * (p2 - p1)
            sliding_points.append(pt)
            sliding_face_ids.append(i)
            
    if len(sliding_points) > 0:
        return np.array(sliding_points), np.array(sliding_face_ids)
    else:
        return np.empty((0, 2)), np.array([], dtype=int)

def generate_inner_points(nodes, faces, sizing_func):
    """
    Generates internal points using Rejection Sampling.
    """
    min_x, max_x = np.min(nodes['x']), np.max(nodes['x'])
    min_y, max_y = np.min(nodes['y']), np.max(nodes['y'])
    
    # Check for empty nodes to avoid the ValueError
    if min_x == max_x and min_y == max_y:
        return np.empty((0, 2))

    # Determine h_min
    test_pts = np.random.rand(100, 2) * [max_x-min_x, max_y-min_y] + [min_x, min_y]
    h_vals = sizing_func(test_pts)
    h_min = np.min(h_vals)
    
    # Estimate count
    area = (max_x - min_x) * (max_y - min_y)
    n_estimated = int(area / (h_min**2 * np.sqrt(3)/2) * 2.0)
    
    candidates = np.random.rand(n_estimated, 2)
    candidates[:,0] = candidates[:,0] * (max_x - min_x) + min_x
    candidates[:,1] = candidates[:,1] * (max_y - min_y) + min_y
    
    h_local = sizing_func(candidates)
    probs = (h_min / h_local)**2
    
    dice = np.random.rand(len(candidates))
    points = candidates[dice < probs]
    
    mask = check_points_inside(points, nodes, faces)
    return points[mask]