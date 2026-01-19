''' mesh/distance.py '''
import numpy as np

def project_points_to_specific_faces(points, face_indices, nodes, faces):
    """
    Projects points[i] strictly to faces[face_indices[i]].
    """
    relevant_faces = faces[face_indices]
    idx_n1 = relevant_faces['n1'] - 1
    idx_n2 = relevant_faces['n2'] - 1
    
    x1, y1 = nodes['x'][idx_n1], nodes['y'][idx_n1]
    x2, y2 = nodes['x'][idx_n2], nodes['y'][idx_n2]
    
    px, py = points[:, 0], points[:, 1]
    
    # Vector P1 -> P
    dx, dy = px - x1, py - y1
    # Vector P1 -> P2 (Wall)
    vx, vy = x2 - x1, y2 - y1
    
    v_sq = vx**2 + vy**2
    v_dot_d = vx*dx + vy*dy
    
    t = v_dot_d / np.maximum(v_sq, 1e-12)
    t = np.clip(t, 0.0, 1.0) # HARD STOP at corners
    
    nearest_x = x1 + t * vx
    nearest_y = y1 + t * vy
    
    return np.column_stack((nearest_x, nearest_y))



def project_points_to_boundary(points, nodes, faces):
    """
    Finds the nearest boundary point for a list of query points.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        nodes:  Structured array of NODE_DTYPE
        faces:  Structured array of FACE_DTYPE
        
    Returns:
        distances: (N,) array of distances to the boundary
        nearest_points: (N, 2) array of (x,y) on the boundary
    """
    # 1. Prepare Data for Broadcasting
    #    Query Points: Shape (N, 1)
    px = points[:, 0][:, np.newaxis]
    py = points[:, 1][:, np.newaxis]
    
    #    Faces: Map to Node Coords -> Shape (1, M)
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
    dist_sq_all = dx**2 + dy**2
    
    # 4. Find Minimum Distance (The "True" Closest Face)
    #    We find the index of the face with min distance for each point
    min_indices = np.argmin(dist_sq_all, axis=1)
    
    # 5. Extract the winning coordinates
    #    We use fancy indexing to pull the specific (cx, cy) that won.
    row_indices = np.arange(len(points))
    
    nearest_x = cx_all[row_indices, min_indices]
    nearest_y = cy_all[row_indices, min_indices]
    
    distances = np.sqrt(dist_sq_all[row_indices, min_indices])
    
    return distances, np.column_stack((nearest_x, nearest_y))