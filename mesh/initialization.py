''' mesh/initialization.py '''
import numpy as np
from .containment import check_points_inside

def resample_boundary_points(nodes, faces, sizing_field, constraints=None):
    """
    Distributes points along edges based on the 'segments' count,
    now aware of geometric constraints (arcs) via the 'ctag' system.
    """
    sliding_points = []
    sliding_face_ids = []
    
    # 1. Build a lookup for constraints based on their ID (C_Tag)
    constraint_map = {}
    if constraints is not None:
        for c in constraints:
            # We now use the constraint's ID as the key for tag-based lookups
            if c['id'] > 0:
                constraint_map[c['id']] = c
    
    for i, face in enumerate(faces):
        idx1, idx2 = face['n1'] - 1, face['n2'] - 1
        n_segments = face['segments']
        if n_segments <= 1: continue 
        
        p1 = np.array([nodes['x'][idx1], nodes['y'][idx1]])
        p2 = np.array([nodes['x'][idx2], nodes['y'][idx2]])
        
        # 2. Look up the constraint using the face's 'ctag'
        const = constraint_map.get(face['ctag'])
        is_arc = (const is not None and const['type'] == 2)
        
        n_samples = 100
        t_samples = np.linspace(0, 1, n_samples)
        
        if is_arc:
            # Arc logic: Interpolate angles around the center defined in params p1, p2, p3
            cx, cy, R = const['p1'], const['p2'], const['p3']
            theta1 = np.arctan2(p1[1] - cy, p1[0] - cx)
            theta2 = np.arctan2(p2[1] - cy, p2[0] - cx)
            
            # Shortest path logic to ensure the arc doesn't wrap the long way around
            d_theta = theta2 - theta1
            if d_theta > np.pi: d_theta -= 2.0 * np.pi
            if d_theta < -np.pi: d_theta += 2.0 * np.pi
            
            theta_samples = theta1 + t_samples * d_theta
            sample_pts = np.column_stack((
                cx + R * np.cos(theta_samples),
                cy + R * np.sin(theta_samples)
            ))
        else:
            # Linear logic (C_Tag 0 or Line constraint)
            sample_pts = p1 + (p2 - p1) * t_samples[:, np.newaxis]
        
        # 3. Re-map 't' based on sizing field density for adaptive boundary spacing
        h_vals = sizing_field(sample_pts)
        density = 1.0 / np.maximum(h_vals, 1e-6)
        cumulative = np.cumsum(density)
        cumulative -= cumulative[0]
        cumulative /= cumulative[-1]
        
        target_cdf = np.linspace(0, 1, n_segments + 1)[1:-1]
        t_final = np.interp(target_cdf, cumulative, t_samples)
        
        for t in t_final:
            if is_arc:
                theta = theta1 + t * d_theta
                sliding_points.append([cx + R * np.cos(theta), cy + R * np.sin(theta)])
            else:
                sliding_points.append(p1 + t * (p2 - p1))
            sliding_face_ids.append(i)
            
    return np.array(sliding_points), np.array(sliding_face_ids)



def generate_inner_points(nodes, hf_segments, sizing_func):
    """
    Generates internal points using Rejection Sampling.
    Uses hf_segments for accurate containment near curved boundaries.
    """
    min_x, max_x = np.min(nodes['x']), np.max(nodes['x'])
    min_y, max_y = np.min(nodes['y']), np.max(nodes['y'])
    
    if min_x == max_x and min_y == max_y:
        return np.empty((0, 2))

    # Determine h_min for sampling density
    test_pts = np.random.rand(100, 2) * [max_x-min_x, max_y-min_y] + [min_x, min_y]
    h_vals = sizing_func(test_pts)
    h_min = np.min(h_vals)
    
    # Estimate candidate count based on area
    area = (max_x - min_x) * (max_y - min_y)
    n_estimated = int(area / (h_min**2 * np.sqrt(3)/2) * 2.0)
    
    candidates = np.random.rand(n_estimated, 2)
    candidates[:,0] = candidates[:,0] * (max_x - min_x) + min_x
    candidates[:,1] = candidates[:,1] * (max_y - min_y) + min_y
    
    h_local = sizing_func(candidates)
    probs = (h_min / h_local)**2
    
    dice = np.random.rand(len(candidates))
    points = candidates[dice < probs]
    
    # FIX: Use the 2-argument call for the new high-fidelity check
    mask = check_points_inside(points, hf_segments)
    return points[mask]