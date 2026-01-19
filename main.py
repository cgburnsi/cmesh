''' main.py '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from core import input_reader
from mesh.containment import check_points_inside
from mesh.initialization import resample_boundary_points, generate_inner_points
from mesh.smoothing import smooth_mesh
from mesh.sizing import SizingField

if __name__ == '__main__':
    filename = 'geom1.inp'
    data = input_reader(filename)
    
    # 1. Initialize Sizing Field
    sizing_field = SizingField(data['sources'])
    print(f"--- Sizing Field Initialized (Background h={sizing_field.h_background}) ---")

    # 2. Build Layers (Now using Locked Boundaries)
    fixed_points = np.column_stack((data['nodes']['x'], data['nodes']['y']))
    
    # Returns POINTS and their LOCKED INDICES
    sliding_points, sliding_face_indices = resample_boundary_points(data['nodes'], data['faces'], sizing_field)
    
    cloud_points = generate_inner_points(data['nodes'], data['faces'], sizing_field)
    
    points = np.vstack((fixed_points, sliding_points, cloud_points))
    n_fixed, n_sliding = len(fixed_points), len(sliding_points)
    
    print(f"Nodes: {n_fixed} Fixed, {n_sliding} Sliding, {len(cloud_points)} Inner")

    # 3. Initial Triangulation
    tri = Delaunay(points)
    simplices = tri.simplices
    centroids = np.mean(points[simplices], axis=1)
    mask = check_points_inside(centroids, data['nodes'], data['faces'])
    initial_cells = simplices[mask]

    # 4. Smooth (Solver)
    print("Smoothing mesh (with dynamic topology updates)...")
    
    # Note: We no longer pass 'initial_cells'
    smoothed_points = smooth_mesh(
        points.copy(), 
        data['nodes'], data['faces'], 
        n_sliding, sliding_face_indices, 
        sizing_field, 
        niters=1000
    )

    # 5. Final Re-Triangulation
    tri_final = Delaunay(smoothed_points)
    final_simplices = tri_final.simplices
    c_final = np.mean(smoothed_points[final_simplices], axis=1)
    mask_final = check_points_inside(c_final, data['nodes'], data['faces'])
    final_cells = final_simplices[mask_final]

    # 6. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Source-Driven Locked Mesh")
    ax.triplot(smoothed_points[:,0], smoothed_points[:,1], final_cells, 'k-', lw=0.5)
    ax.scatter(smoothed_points[:,0], smoothed_points[:,1], s=2, c='k')
    ax.set_aspect('equal')
    plt.savefig('snapmesh_sources.png')