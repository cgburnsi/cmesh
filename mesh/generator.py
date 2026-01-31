''' mesh/generator.py '''
import numpy as np
from scipy.spatial import Delaunay
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
        self.field = SizingField(self.fields_in)
        self.smoother = smoother if smoother is not None else spring_smoother

    def generate(self, niters=1000, refine=True):
        # 1. Seeding (Fixed, Sliding, and Interior) [cite: 13]
        fixed_pts = np.column_stack((self.nodes_in['x'], self.nodes_in['y']))
        sliding_pts, sliding_face_indices = resample_boundary_points(
            self.nodes_in, self.faces_in, self.field, constraints=self.constraints_in
        )
        interior_pts = generate_inner_points(self.nodes_in, self.faces_in, self.field)
        
        points = np.vstack((fixed_pts, sliding_pts, interior_pts))
        n_fixed, n_sliding = len(fixed_pts), len(sliding_pts)
        
        # Define high-fidelity boundary for containment checks 
        # This prevents the 'straight-line' clipping of circular arcs.
        boundary_poly = np.vstack((fixed_pts, sliding_pts)) 
        
        # 2. Initial Smoothing Pass [cite: 13]
        smoothed_points = self.smoother(
            points.copy(), self.nodes_in, self.faces_in, n_sliding, 
            sliding_face_indices, self.field, niters=niters, constraints=self.constraints_in
        )
        
        # 3. Refinement [cite: 13]
        if refine:
            tri = Delaunay(smoothed_points)
            centroids = np.mean(smoothed_points[tri.simplices], axis=1)
            active_cells = tri.simplices[check_points_inside(centroids, boundary_poly)]
            q_values = compute_triangle_quality(smoothed_points, active_cells)
            bad_mask = q_values < 0.3
            
            if np.any(bad_mask):
                bad_centroids = np.mean(smoothed_points[active_cells[bad_mask]], axis=1)
                smoothed_points = np.vstack((smoothed_points, bad_centroids))
                smoothed_points = self.smoother(
                    smoothed_points, self.nodes_in, self.faces_in, n_sliding, 
                    sliding_face_indices, self.field, niters=200, constraints=self.constraints_in
                )
        
        # 4. Final Triangulation [cite: 13]
        tri_final = Delaunay(smoothed_points)
        c_final = np.mean(smoothed_points[tri_final.simplices], axis=1)
        final_cells = tri_final.simplices[check_points_inside(c_final, boundary_poly)]
        
        return smoothed_points, final_cells
    
    def get_quality(self, points, cells):
        q_values = compute_triangle_quality(points, cells)
        return q_values, {'min': np.min(q_values), 'avg': np.mean(q_values), 'worst_indices': np.where(q_values < 0.2)[0]}