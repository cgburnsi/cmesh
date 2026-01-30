''' mesh/distance.py '''
import numpy as np

def get_closest_point_on_segment(px, py, x1, y1, x2, y2):
    """ Vectorized kernel to find the closest point on segment P1-P2 for point P. """
    dx, dy = px - x1, py - y1
    vx, vy = x2 - x1, y2 - y1
    len_sq = np.where((vx**2 + vy**2) == 0, 1.0, vx**2 + vy**2)
    t = np.clip((dx * vx + dy * vy) / len_sq, 0.0, 1.0)
    return x1 + t * vx, y1 + t * vy

def get_closest_point_on_arc(px, py, cx, cy, R, theta1, theta2):
    """ Finds the closest point on a circular arc defined by a center and radius. """
    dx, dy = px - cx, py - cy
    dist = np.maximum(np.sqrt(dx**2 + dy**2), 1e-12)
    
    # Project to the infinite circle
    nx, ny = cx + R * (dx / dist), cy + R * (dy / dist)
    
    # Angular clamping to the arc segment
    angle = np.arctan2(ny - cy, nx - cx)
    
    # Adjust for shortest angular path
    def normalize_angle(a, target):
        while a < target - np.pi: a += 2*np.pi
        while a > target + np.pi: a -= 2*np.pi
        return a

    theta2_adj = normalize_angle(theta2, theta1)
    angle_adj = normalize_angle(angle, theta1)
    
    # Clamp angle between theta1 and theta2_adj
    start, end = (theta1, theta2_adj) if theta1 < theta2_adj else (theta2_adj, theta1)
    clamped_angle = np.clip(angle_adj, start, end)
    
    return cx + R * np.cos(clamped_angle), cy + R * np.sin(clamped_angle)

def project_points_to_specific_faces(points, face_indices, nodes, faces, constraints=None):
    """ Projects points strictly to assigned faces, supporting arcs. """
    constraint_map = {c['target']: c for c in constraints if c['target'] > 0} if constraints is not None else {}
    new_pts = np.zeros_like(points)
    
    for i, f_idx in enumerate(face_indices):
        face = faces[f_idx]
        n1, n2 = nodes[face['n1']-1], nodes[face['n2']-1]
        x1, y1, x2, y2 = n1['x'], n1['y'], n2['x'], n2['y']
        
        const = constraint_map.get(face['id'])
        if const is not None and const['type'] == 2: # Circle/Arc
            cx, cy, R = const['p1'], const['p2'], const['p3']
            t1, t2 = np.arctan2(y1 - cy, x1 - cx), np.arctan2(y2 - cy, x2 - cx)
            new_pts[i] = get_closest_point_on_arc(points[i, 0], points[i, 1], cx, cy, R, t1, t2)
        else: # Default Line
            new_pts[i, 0], new_pts[i, 1] = get_closest_point_on_segment(points[i, 0], points[i, 1], x1, y1, x2, y2)
            
    return new_pts

def project_points_to_boundary(points, nodes, faces):
    """ Finds the closest point on ANY boundary face for a list of points. """
    px, py = points[:, 0][:, np.newaxis], points[:, 1][:, np.newaxis]
    idx_n1, idx_n2 = faces['n1'] - 1, faces['n2'] - 1
    x1, y1 = nodes['x'][idx_n1][np.newaxis, :], nodes['y'][idx_n1][np.newaxis, :]
    x2, y2 = nodes['x'][idx_n2][np.newaxis, :], nodes['y'][idx_n2][np.newaxis, :]
    
    cx_all, cy_all = get_closest_point_on_segment(px, py, x1, y1, x2, y2)
    dists_sq = (px - cx_all)**2 + (py - cy_all)**2
    min_indices = np.argmin(dists_sq, axis=1)
    
    rows = np.arange(len(points))
    best_x, best_y = cx_all[rows, min_indices], cy_all[rows, min_indices]
    return np.sqrt(dists_sq[rows, min_indices]), np.column_stack((best_x, best_y))