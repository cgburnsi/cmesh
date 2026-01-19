''' mesh/containment.py '''
import numpy as np

def check_points_inside(points, nodes, faces):
    """
    Determines which points are inside the boundary using Ray Casting.
    
    Args:
        points: (N, 2) array of query points.
        nodes:  Structured array of NODE_DTYPE.
        faces:  Structured array of FACE_DTYPE.
        
    Returns:
        is_inside: (N,) boolean array (True if inside).
    """
    # 1. Prepare Data
    px = points[:, 0]
    py = points[:, 1]
    
    # Get all face coordinates
    idx_n1 = faces['n1'] - 1
    idx_n2 = faces['n2'] - 1
    
    x1 = nodes['x'][idx_n1]
    y1 = nodes['y'][idx_n1]
    x2 = nodes['x'][idx_n2]
    y2 = nodes['y'][idx_n2]
    
    # 2. Ray Casting Algorithm (Vectorized Loop over Faces)
    #    We check every point against every face.
    #    To save memory, we accumulate the result.
    inside_mask = np.zeros(len(points), dtype=bool)
    
    # We loop over faces because N_points * M_faces can be huge in memory.
    # Checking 100k points against 1 face is very fast.
    for i in range(len(faces)):
        face_x1, face_y1 = x1[i], y1[i]
        face_x2, face_y2 = x2[i], y2[i]
        
        # Condition 1: The point's Y must be between the face's Ys
        # (face_y1 > py) != (face_y2 > py)
        y_cond = (face_y1 > py) != (face_y2 > py)
        
        # Condition 2: The point must be to the Left of the intersection
        # x < (intersection_x)
        # intersection_x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        # We only calculate this if Y_cond is true to save time, but
        # numpy does it all at once usually.
        
        # Safe division logic not strictly needed if we are careful, 
        # but let's just calculate the condition:
        # Px < (x2-x1)*(Py-y1)/(y2-y1) + x1
        
        # Determine intersection X
        slope = (face_x2 - face_x1) / (face_y2 - face_y1 + 1e-12) # Avoid div/0
        intersect_x = slope * (py - face_y1) + face_x1
        
        x_cond = px < intersect_x
        
        # If both overlap Y range AND are to the left, it's a crossing
        crossing = y_cond & x_cond
        
        # Toggle the state (Odd/Even check)
        inside_mask ^= crossing
        
    return inside_mask






if __name__ == '__main__':
    pass


