''' mesh/containment.py '''
import numpy as np

def check_points_inside(points, segments):
    """
    Determines which points are inside the boundary using Ray Casting.
    Works with a list of high-fidelity segments to support curved boundaries.
    
    Args:
        points: (N, 2) array of query points (centroids).
        segments: (M, 4) array where each row is [x1, y1, x2, y2].
        
    Returns:
        is_inside: (N,) boolean array.
    """
    px, py = points[:, 0], points[:, 1]
    inside_mask = np.zeros(len(points), dtype=bool)
    
    # Vectorized loop over segments
    for i in range(len(segments)):
        x1, y1, x2, y2 = segments[i]
        
        # Ray casting logic
        y_cond = (y1 > py) != (y2 > py)
        
        # Determine intersection X with avoid-division-by-zero
        intersect_x = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
        x_cond = px < intersect_x
        
        crossing = y_cond & x_cond
        inside_mask ^= crossing
        
    return inside_mask