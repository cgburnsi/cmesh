''' mesh/generator.py '''
import numpy as np
from scipy.spatial import Delaunay

# Local imports from within the mesh package
from .containment import check_points_inside
from .initialization import resample_boundary_points, generate_inner_points
from .sizing import SizingField
from .geometry import compute_triangle_quality 
from .smoothing import spring_smoother  # Import the default option

class MeshGenerator:
    def __init__(self, data, smoother=None):
        self.nodes_in = data['nodes']
        self.faces_in = data['faces']
        self.constraints_in = data['constraints']
        self.fields_in = data['fields']
        
        self.field = SizingField(self.fields_in)

        # Inject the smoothing strategy
        self.smoother = smoother if smoother is not None else spring_smoother

        
    def generate(self, niters=1000):
        # 1. Seeding (Fixed Points, 'Sliding' Points on Constraints, and Interior Points)
        fixed_pts = np.column_stack((self.nodes_in['x'], self.nodes_in['y']))
        sliding_pts, sliding_face_indices = resample_boundary_points(self.nodes_in, self.faces_in, self.field)
        
        interior_pts = generate_inner_points(self.nodes_in, self.faces_in, self.field)
        
        points = np.vstack((fixed_pts, sliding_pts, interior_pts))
        
        # 2. Initial Triangulation
        tri = Delaunay(points)
        simplices = tri.simplices
        centroids = np.mean(points[simplices], axis=1)
        mask = check_points_inside(centroids, self.nodes_in, self.faces_in)
        # initial_cells = simplices[mask] # Placeholder for debugging if needed
        
        n_fixed, n_sliding = len(fixed_pts), len(sliding_pts)
        
        # 3. Smoothing
        smoothed_points = self.smoother(
            points.copy(), self.nodes_in, self.faces_in, n_sliding, 
            sliding_face_indices, self.field, niters=niters
        )
        
        # 4. Final Re-Triangulation
        tri_final = Delaunay(smoothed_points)
        final_simplices = tri_final.simplices
        c_final = np.mean(smoothed_points[final_simplices], axis=1)
        mask_final = check_points_inside(c_final, self.nodes_in, self.faces_in)
        final_cells = final_simplices[mask_final]
        
        return smoothed_points, final_cells
    
    def get_quality(self, points, cells):
        """ Returns the quality metrics for the final generated mesh. """
        q_values = compute_triangle_quality(points, cells)
        
        stats = {
            'min': np.min(q_values),
            'avg': np.mean(q_values),
            'worst_indices': np.where(q_values < 0.2)[0] # Identify bad slivers
        }
        return q_values, stats
    
    
    
    
    
    
    
