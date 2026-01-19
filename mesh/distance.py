''' mesh/distance.py '''
import numpy as np

def get_closest_point_on_segment(px, py, x1, y1, x2, y2):
    """
    Core Kernel: Finds the closest point on a line segment (x1,y1)->(x2,y2) 
    for a query point (px,py).
    
    This function supports NumPy broadcasting:
    px, py: shape (N, 1)  (Query Points)
    x1, y1: shape (1, M)  (Segment Start)
    x2, y2: shape (1, M)  (Segment End)
    
    Returns:
        cx, cy: shape (N, M) - The closest point on EACH segment for EACH point.
    """
    # 1. Vector from Segment Start to Point (P - A)
    dpx = px - x1
    dpy = py - y1
    
    # 2. Segment Vector (B - A)
    sx = x2 - x1
    sy = y2 - y1
    
    # 3. Project P onto the line (dot product)
    #    t = (P-A) . (B-A) / |B-A|^2
    seg_len_sq = sx**2 + sy**2
    
    # Avoid division by zero
    seg_len_sq = np.where(seg_len_sq == 0, 1.0, seg_len_sq)
    
    t = (dpx * sx + dpy * sy) / seg_len_sq
    
    # 4. Clamp 't' to segment [0, 1] to handle endpoints
    t = np.clip(t, 0.0, 1.0)
    
    # 5. Calculate closest point
    cx = x1 + t * sx
    cy = y1 + t * sy
    
    return cx, cy

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