''' mesh/triangulation.py '''
import numpy as np
from scipy.spatial import Delaunay
from .containment import check_points_inside

def triangulate_and_filter(fixed_nodes, cloud_points, faces):
    """
    1. Stacks fixed nodes and cloud points.
    2. Triangulates (Delaunay).
    3. Removes triangles whose centroids are outside the boundary.
    
    Returns:
        points: (N, 2) array of all coordinates
        simplices: (M, 3) array of valid triangle indices
    """
    # 1. Stack Data: [Fixed] + [Cloud]
    fixed_xy = np.column_stack((fixed_nodes['x'], fixed_nodes['y']))
    
    # Handle case where cloud is empty (e.g. very coarse mesh)
    if len(cloud_points) > 0:
        all_points = np.vstack((fixed_xy, cloud_points))
    else:
        all_points = fixed_xy
        
    # 2. Raw Delaunay
    tri = Delaunay(all_points)
    simplices = tri.simplices
    
    # 3. Compute Centroids of every triangle
    #    Shape: (M_tris, 3_nodes, 2_coords)
    tri_coords = all_points[simplices]
    
    #    Average along axis 1 (the 3 nodes) to get centroid (M, 2)
    centroids = np.mean(tri_coords, axis=1)
    
    # 4. Filter: Keep only triangles inside the domain
    #    Re-use our vectorized ray-caster!
    inside_mask = check_points_inside(centroids, fixed_nodes, faces)
    
    valid_simplices = simplices[inside_mask]
    
    return all_points, valid_simplices


if __name__ == '__main__':
    pass


