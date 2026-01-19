''' mesh/distance.py '''
import numpy as np

def get_closest_point_on_segment(px, py, x1, y1, x2, y2):
    """
    Vectorized kernel to find the closest point on segment P1-P2 for point P.
    
    Args:
        px, py: Coordinates of point P (N, 1) or (N, M)
        x1, y1: Coordinates of start node P1 (1, M)
        x2, y2: Coordinates of end node P2 (1, M)
        
    Returns:
        cx, cy: Coordinates of the closest point on the segment
    """
    # Vector from StartNode to Point (P - P1)
    dx = px - x1
    dy = py - y1
    
    # Vector of the Segment (P2 - P1)
    vx = x2 - x1
    vy = y2 - y1
    
    # Squared length of the segment
    len_sq = vx**2 + vy**2
    
    # Avoid division by zero for zero-length segments
    len_sq = np.where(len_sq == 0, 1.0, len_sq)
    
    # Project vector 'd' onto vector 'v'
    # t = (d . v) / (v . v)
    t = (dx * vx + dy * vy) / len_sq
    
    # Clamp t to the segment [0, 1]
    t = np.clip(t, 0.0, 1.0)
    
    # Calculate closest point
    cx = x1 + t * vx
    cy = y1 + t * vy
    
    return cx, cy

def project_points_to_specific_faces(points, face_indices, nodes, faces):
    """
    Projects points[i] strictly to faces[face_indices[i]].
    This prevents corner-rounding and drifting by locking nodes to specific topology.
    """
    # 1. Gather Face Data for each point
    relevant_faces = faces[face_indices]
    
    idx_n1 = relevant_faces['n1'] - 1
    idx_n2 = relevant_faces['n2'] - 1
    
    x1 = nodes['x'][idx_n1]
    y1 = nodes['y'][idx_n1]
    x2 = nodes['x'][idx_n2]
    y2 = nodes['y'][idx_n2]
    
    # 2. Vectorized Projection
    px, py = points[:, 0], points[:, 1]
    
    # Vector from StartNode to Point
    dx = px - x1
    dy = py - y1
    
    # Vector of the Wall Segment
    vx = x2 - x1
    vy = y2 - y1
    
    # Project: t = (v . d) / (v . v)
    v_sq = vx**2 + vy**2
    v_dot_d = vx*dx + vy*dy
    
    # Avoid div/0
    v_sq = np.maximum(v_sq, 1e-12)
    
    t = v_dot_d / v_sq
    
    # 3. Clamp t to [0, 1] - Hard Stop at corners
    t = np.clip(t, 0.0, 1.0)
    
    # 4. Calculate new position
    nearest_x = x1 + t * vx
    nearest_y = y1 + t * vy
    
    return np.column_stack((nearest_x, nearest_y))

def project_points_to_boundary(points, nodes, faces):
    """
    Finds the closest point on ANY boundary face for a list of points.
    Used for "leaker" checks (points that escaped the domain).
    """
    # 1. Prepare Data
    # Points: Shape (N, 1) for broadcasting
    px = points[:, 0][:, np.newaxis]
    py = points[:, 1][:, np.newaxis]
    
    # Faces: Shape (1, M) for broadcasting
    idx_n1 = faces['n1'] - 1
    idx_n2 = faces['n2'] - 1
    
    x1 = nodes['x'][idx_n1][np.newaxis, :]
    y1 = nodes['y'][idx_n1][np.newaxis, :]
    x2 = nodes['x'][idx_n2][np.newaxis, :]
    y2 = nodes['y'][idx_n2][np.newaxis, :]
    
    # 2. Run Kernel (N points vs M faces) -> Result (N, M)
    #    This gives us the closest point on *every* face for *every* point.
    cx_all, cy_all = get_closest_point_on_segment(px, py, x1, y1, x2, y2)
    
    # 3. Calculate Distances Squared (N, M)
    dx = px - cx_all
    dy = py - cy_all
    dists_sq = dx**2 + dy**2
    
    # 4. Find the Minimum Distance Index for each point
    min_indices = np.argmin(dists_sq, axis=1)
    
    # 5. Extract the coordinates corresponding to the minimum distance
    #    We use numpy's advanced indexing: range(N) selects the row, min_indices selects the col
    row_indices = np.arange(len(points))
    
    best_x = cx_all[row_indices, min_indices]
    best_y = cy_all[row_indices, min_indices]
    
    closest_points = np.column_stack((best_x, best_y))
    min_dists = np.sqrt(dists_sq[row_indices, min_indices])
    
    return min_dists, closest_points