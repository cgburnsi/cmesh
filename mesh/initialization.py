''' mesh/initialization.py '''
import numpy as np
from .containment import check_points_inside



def resample_boundary_points(nodes, faces, h0):
    """
    Returns points strictly *between* the endpoints of the faces.
    These are the 'Sliding' nodes.
    """
    sliding_points = []
    
    for face in faces:
        idx1 = face['n1'] - 1
        idx2 = face['n2'] - 1
        
        p1 = np.array([nodes['x'][idx1], nodes['y'][idx1]])
        p2 = np.array([nodes['x'][idx2], nodes['y'][idx2]])
        
        dist = np.linalg.norm(p2 - p1)
        
        # Calculate how many segments fit
        n_segments = int(np.ceil(dist / h0))
        
        # Generate points, but EXCLUDE start(0) and end(1)
        # because those are the Fixed Corners.
        if n_segments > 1:
            # [1:-1] slices off the first (0.0) and last (1.0) points
            t_vals = np.linspace(0, 1, n_segments + 1)[1:-1]
            for t in t_vals:
                point = p1 + t * (p2 - p1)
                sliding_points.append(point)
            
    if len(sliding_points) > 0:
        return np.array(sliding_points)
    else:
        # Return empty array if no points fit (coarse mesh)
        return np.empty((0, 2))

def generate_inner_points(nodes, faces, h0):
    """
    Generates ONLY the internal point cloud (excluding boundary).
    """
    # 1. Bounding Box
    min_x = np.min(nodes['x'])
    max_x = np.max(nodes['x'])
    min_y = np.min(nodes['y'])
    max_y = np.max(nodes['y'])
    
    # 2. Hexagonal Grid
    x_range = np.arange(min_x, max_x, h0)
    y_range = np.arange(min_y, max_y, h0 * np.sqrt(3)/2) 
    xx, yy = np.meshgrid(x_range, y_range)
    xx[1::2] += h0 / 2.0
    
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # 3. Filter: Keep only points INSIDE
    mask = check_points_inside(points, nodes, faces)
    return points[mask]

def resample_boundary(nodes, faces, h0):
    """
    Walks along every face and places new nodes at spacing h0.
    """
    boundary_points = []
    
    # Loop over every face (edge)
    for face in faces:
        # Get coordinates of start (P1) and end (P2)
        idx1 = face['n1'] - 1
        idx2 = face['n2'] - 1
        
        p1 = np.array([nodes['x'][idx1], nodes['y'][idx1]])
        p2 = np.array([nodes['x'][idx2], nodes['y'][idx2]])
        
        # Calculate length
        dist = np.linalg.norm(p2 - p1)
        
        # Calculate how many segments fit
        n_segments = int(np.ceil(dist / h0))
        
        # Generate points along the line
        # linspace(0, 1, N+1) gives us points including endpoints
        t_vals = np.linspace(0, 1, n_segments + 1)
        
        for t in t_vals:
            point = p1 + t * (p2 - p1)
            boundary_points.append(point)
            
    # Remove duplicates (because corners will be generated twice)
    boundary_points = np.array(boundary_points)
    unique_boundary = np.unique(boundary_points, axis=0)
    
    return unique_boundary

def generate_initial_points(nodes, faces, h0):
    """
    Updated to include boundary resampling.
    """
    # ... (Keep your existing Bounding Box & Meshgrid code) ...
    min_x = np.min(nodes['x'])
    max_x = np.max(nodes['x'])
    min_y = np.min(nodes['y'])
    max_y = np.max(nodes['y'])
    
    x_range = np.arange(min_x, max_x, h0)
    y_range = np.arange(min_y, max_y, h0 * np.sqrt(3)/2) 
    xx, yy = np.meshgrid(x_range, y_range)
    xx[1::2] += h0 / 2.0
    
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Filter INSIDE
    mask = check_points_inside(points, nodes, faces)
    inner_points = points[mask]
    
    # --- NEW STEP: Resample Boundary ---
    bnd_points = resample_boundary(nodes, faces, h0)
    
    # Combine: [Boundary Points] + [Inner Points]
    # We stack them so the boundary is definitely included
    all_points = np.vstack((bnd_points, inner_points))
    
    return all_points






if __name__ == '__main__':
    pass


