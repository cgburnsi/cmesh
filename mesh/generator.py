''' mesh/generator.py '''
import numpy as np
from scipy.spatial import Delaunay

# Local imports from within the mesh package
from .containment import check_points_inside
from .initialization import resample_boundary_points, generate_inner_points
from .sizing import SizingField
from .geometry import compute_triangle_quality 
from .smoothing import spring_smoother

class MeshGenerator:
    def __init__(self, data, smoother=None):
        self.nodes_in = data['nodes']
        self.faces_in = data['faces']
        self.constraints_in = data['constraints']
        self.fields_in = data['fields']
        
        # Sizing field initialized with boundary context
        self.field = SizingField(self.fields_in)

        # Inject the smoothing strategy
        self.smoother = smoother if smoother is not None else spring_smoother

    def generate(self, niters=1000, refine=True):
        # 1. Seeding (Fixed Points, 'Sliding' Points, and Interior Points)
        fixed_pts = np.column_stack((self.nodes_in['x'], self.nodes_in['y']))
        sliding_pts, sliding_face_indices = resample_boundary_points(
            self.nodes_in, self.faces_in, self.field
        )
        interior_pts = generate_inner_points(self.nodes_in, self.faces_in, self.field)
        
        points = np.vstack((fixed_pts, sliding_pts, interior_pts))
        n_fixed, n_sliding = len(fixed_pts), len(sliding_pts)
        
        # 2. Initial Smoothing Pass
        # Gets the seeded points into a decent initial state
        smoothed_points = self.smoother(
            points.copy(), self.nodes_in, self.faces_in, n_sliding, 
            sliding_face_indices, self.field, niters=niters
        )
        
        # 3. POINT INSERTION (Refinement)
        if refine:
            tri = Delaunay(smoothed_points)
            centroids = np.mean(smoothed_points[tri.simplices], axis=1)
            mask = check_points_inside(centroids, self.nodes_in, self.faces_in)
            active_cells = tri.simplices[mask]
            
            # Compute quality and identify bad triangles (Q < 0.3)
            q_values = compute_triangle_quality(smoothed_points, active_cells)
            bad_mask = q_values < 0.3
            
            if np.any(bad_mask):
                # Inject centroids of bad triangles into the cloud
                bad_centroids = np.mean(smoothed_points[active_cells[bad_mask]], axis=1)
                smoothed_points = np.vstack((smoothed_points, bad_centroids))
                
                # Stabilizing Smoothing Pass
                smoothed_points = self.smoother(
                    smoothed_points, self.nodes_in, self.faces_in, n_sliding, 
                    sliding_face_indices, self.field, niters=200
                )
        
        # 4. POINT COALESCENCE (Deletion)
        # Check for points that have clumped too close together
        tri_check = Delaunay(smoothed_points)
        edges = np.vstack((tri_check.simplices[:, [0, 1]], 
                           tri_check.simplices[:, [1, 2]], 
                           tri_check.simplices[:, [2, 0]]))
        
        p1, p2 = smoothed_points[edges[:,0]], smoothed_points[edges[:,1]]
        edge_lens = np.linalg.norm(p1 - p2, axis=1)
        
        # FIX: Calculate the midpoint for EACH edge individually
        midpoints = (p1 + p2) / 2.0
        
        # This will now return an array of h values, one for each edge
        h_target = self.field(midpoints)
        
        # Now both arrays have the same shape, allowing comparison
        too_close = (edge_lens < 0.4 * h_target)
        
        if np.any(too_close):
            kill_list = edges[too_close, 1]
            # Ensure we only delete interior nodes, never fixed or sliding boundary nodes
            kill_list = kill_list[kill_list >= (n_fixed + n_sliding)]
            
            if len(kill_list) > 0:
                smoothed_points = np.delete(smoothed_points, np.unique(kill_list), axis=0)
                # Final stabilization smoothing after cleanup
                smoothed_points = self.smoother(
                    smoothed_points, self.nodes_in, self.faces_in, n_sliding, 
                    sliding_face_indices, self.field, niters=100
                )

        # 5. Final Re-Triangulation
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
            'worst_indices': np.where(q_values < 0.2)[0]
        }
        return q_values, stats