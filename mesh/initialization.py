''' mesh/initialization.py '''
import numpy as np
from .containment import check_points_inside

def resample_boundary_points(nodes, faces, sizing_field, constraints=None):
    """
    Distributes points along edges based on the local SizingField. 
    """
    sliding_points = []
    sliding_face_ids = []
    
    constraint_map = {c['id']: c for c in constraints} if constraints is not None else {}
    
    for i, face in enumerate(faces):
        idx1, idx2 = face['n1'] - 1, face['n2'] - 1
        p1 = np.array([nodes['x'][idx1], nodes['y'][idx1]])
        p2 = np.array([nodes['x'][idx2], nodes['y'][idx2]])
        
        n_segments = face['segments']
        if n_segments <= 0:
            length = np.linalg.norm(p2 - p1)
            h_target = sizing_field((p1 + p2) / 2.0) 
            n_segments = int(np.ceil(length / h_target))
            n_segments = max(n_segments, 1)

        if n_segments <= 1: 
            continue 
        
        const = constraint_map.get(face['ctag'])
        is_arc = (const is not None and const['type'] == 2)
        t_samples = np.linspace(0, 1, 100)
        
        if is_arc:
            cx, cy, R = const['p1'], const['p2'], const['p3']
            theta1 = np.arctan2(p1[1] - cy, p1[0] - cx)
            theta2 = np.arctan2(p2[1] - cy, p2[0] - cx)
            d_theta = theta2 - theta1
            if d_theta > np.pi: d_theta -= 2.0 * np.pi
            if d_theta < -np.pi: d_theta += 2.0 * np.pi
            
            theta_samples = theta1 + t_samples * d_theta
            sample_pts = np.column_stack((cx + R * np.cos(theta_samples), cy + R * np.sin(theta_samples)))
        else:
            sample_pts = p1 + (p2 - p1) * t_samples[:, np.newaxis]
        
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

def generate_frontal_points(sliding_pts, sliding_face_ids, nodes, faces, sizing_field, hf_segments):
    """
    Generates points offset from boundaries. 
    UPDATED: Returns associated face IDs to maintain tag consistency across layers.
    """
    if len(sliding_pts) == 0:
        return np.empty((0, 2)), np.array([], dtype=int)
        
    frontal_pts = []
    new_face_ids = []
    
    for i, p in enumerate(sliding_pts):
        f_idx = sliding_face_ids[i]
        face = faces[f_idx]
        
        # Only generate layers for Physical Walls (Tag 2)
        if face['tag'] != 2:
            continue
            
        n1, n2 = nodes[face['n1']-1], nodes[face['n2']-1]
        dx, dy = n2['x'] - n1['x'], n2['y'] - n1['y']
        mag = np.sqrt(dx**2 + dy**2)
        
        nx, ny = -dy/mag, dx/mag 
        h = sizing_field(p)
        
        probe = p + np.array([nx, ny]) * (h * 0.1)
        if not check_points_inside(np.atleast_2d(probe), hf_segments)[0]:
            nx, ny = -nx, -ny 
            
        frontal_pts.append(p + np.array([nx, ny]) * h)
        new_face_ids.append(f_idx)
        
    # Ensure return is always (N, 2) even if empty
    return np.array(frontal_pts).reshape(-1, 2), np.array(new_face_ids)