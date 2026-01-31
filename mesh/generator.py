''' mesh/generator.py '''
import numpy as np
from scipy.spatial import Delaunay
from .containment import check_points_inside
from .initialization import resample_boundary_points, generate_frontal_points
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

    def generate(self, niters=1000, n_layers=3):
        # 1. Boundary Seeding
        fixed_pts = np.column_stack((self.nodes_in['x'], self.nodes_in['y']))
        sliding_pts, sliding_face_ids = resample_boundary_points(
            self.nodes_in, self.faces_in, self.field, constraints=self.constraints_in
        )
        
        hf_segments = []
        for i, face in enumerate(self.faces_in):
            p_start, p_end = fixed_pts[face['n1']-1], fixed_pts[face['n2']-1]
            face_pts = sliding_pts[sliding_face_ids == i]
            path = np.vstack((p_start, face_pts, p_end))
            for j in range(len(path) - 1):
                hf_segments.append([path[j,0], path[j,1], path[j+1,0], path[j+1,1]])
        hf_segments = np.array(hf_segments)

        # 2. Multi-Layer Frontal Inflation
        # FIX: Track Face IDs through the layers to prevent Tag Mismatch
        current_layer_pts = sliding_pts
        current_face_ids = sliding_face_ids
        all_frontal_pts = []
        
        for _ in range(n_layers):
            new_layer, new_face_ids = generate_frontal_points(
                current_layer_pts, current_face_ids, self.nodes_in, self.faces_in, self.field, hf_segments
            )
            all_frontal_pts.append(new_layer)
            current_layer_pts = new_layer
            current_face_ids = new_face_ids
            
        points = np.vstack([fixed_pts, sliding_pts] + all_frontal_pts)
        n_fixed, n_sliding = len(fixed_pts), len(sliding_pts)
        
        # 3. Iterative Refinement
        for ref_iter in range(8):
            tri = Delaunay(points)
            centroids = np.mean(points[tri.simplices], axis=1)
            
            inside_mask = check_points_inside(centroids, hf_segments)
            active_simplices = tri.simplices[inside_mask]
            active_centroids = centroids[inside_mask]
            
            p1, p2, p3 = points[active_simplices[:,0]], points[active_simplices[:,1]], points[active_simplices[:,2]]
            area = 0.5 * np.abs(p1[:,0]*(p2[:,1]-p3[:,1]) + p2[:,0]*(p3[:,1]-p1[:,1]) + p3[:,0]*(p1[:,1]-p2[:,1]))
            
            h_target = self.field(active_centroids)
            refine_mask = area > (h_target**2 * 0.6) 
            
            if not np.any(refine_mask):
                break
                
            points = np.vstack((points, active_centroids[refine_mask]))
            
            points = self.smoother(
                points, self.nodes_in, self.faces_in, n_sliding, sliding_face_ids, 
                self.field, niters=20, constraints=self.constraints_in, hf_segments=hf_segments
            )

        # 4. Final smoothing pass
        final_points = self.smoother(
            points, self.nodes_in, self.faces_in, n_sliding, sliding_face_ids, 
            self.field, niters=niters, constraints=self.constraints_in, hf_segments=hf_segments
        )
        
        tri_final = Delaunay(final_points)
        c_final = np.mean(final_points[tri_final.simplices], axis=1)
        final_cells = tri_final.simplices[check_points_inside(c_final, hf_segments)]
        
        return final_points, final_cells

    def get_quality(self, points, cells):
        q_values = compute_triangle_quality(points, cells)
        return q_values, {
            'min': np.min(q_values), 
            'avg': np.mean(q_values), 
            'worst_indices': np.where(q_values < 0.2)[0]
        }