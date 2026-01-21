''' main.py '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from core import input_reader
from mesh.containment import check_points_inside
from mesh.initialization import resample_boundary_points, generate_inner_points
from mesh.smoothing import smooth_mesh
from mesh.sizing import SizingField


class MeshGenerator:
    def __init__(self, data):
        self.nodes_in = data['nodes']
        self.faces_in = data['faces']
        self.constraints_in = data['constraints']
        self.fields_in = data['fields']
        
        self.field = SizingField(self.fields_in)
        
        # Connectivity Arrays
        
    def generate(self):
        
        # 1 Seeding (Fixed Points, 'Sliding' Points on Constraints, and Interior Points)
        fixed_pts = np.column_stack((self.nodes_in['x'], self.nodes_in['y']))
        sliding_pts, sliding_face_indicies = resample_boundary_points(self.nodes_in, self.faces_in, self.field)
        interior_pts = generate_inner_points(self.nodes_in, self.faces_in, self.field)
        
        points = np.vstack((fixed_pts, sliding_pts, interior_pts))
        
        
        # 2 Initial Triangulation
        tri = Delaunay(points)
        simplices = tri.simplices
        centroids = np.mean(points[simplices], axis=1)
        mask = check_points_inside(centroids, self.nodes_in, self.faces_in)
        initial_cells = simplices[mask]
        n_fixed, n_sliding = len(fixed_pts), len(sliding_pts)

        
        # 3 Smoothing
        smoothed_points = smooth_mesh(points.copy(), self.nodes_in, self.faces_in, n_sliding, 
                                      sliding_face_indicies, self.field, niters=1000)
        
        
        # 4. Final Re-Triangulation
        tri_final = Delaunay(smoothed_points)
        final_simplices = tri_final.simplices
        c_final = np.mean(smoothed_points[final_simplices], axis=1)
        mask_final = check_points_inside(c_final, data['nodes'], data['faces'])
        final_cells = final_simplices[mask_final]
        
        return smoothed_points, final_cells
        

if __name__ == '__main__':
    
    data = input_reader('geom1.inp')
    
    mesh = MeshGenerator(data)
    smoothed_points, final_cells = mesh.generate()
    
    # 6. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Source-Driven Locked Mesh")
    ax.triplot(smoothed_points[:,0], smoothed_points[:,1], final_cells, 'k-', lw=0.5)
    ax.scatter(smoothed_points[:,0], smoothed_points[:,1], s=2, c='k')
    ax.set_aspect('equal')
    plt.savefig('snapmesh_sources.png')
    
    
    
    
    
    