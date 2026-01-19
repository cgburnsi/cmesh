''' main.py '''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay

# Core Imports
from core import input_reader
from mesh.containment import check_points_inside

# Initialization Imports
# We use the separate functions we defined to build the layers
from mesh.initialization import resample_boundary_points, generate_inner_points

# Smoothing Imports
from mesh.smoothing import smooth_mesh

if __name__ == '__main__':
    # 1. Load Geometry
    #    Now reads [nodes], [faces], [constraints]
    filename = 'geom1.inp'
    data = input_reader(filename)
    
    # Target Edge Length
    h0 = 0.4
    print(f"--- Processing {filename} with h0={h0} ---")

    # 2. Build the Node Layers
    
    # Layer 1: Fixed Corners (The raw nodes from the file)
    # These are immutable. They define the topology anchors.
    fixed_points = np.column_stack((data['nodes']['x'], data['nodes']['y']))
    
    # Layer 2: Sliding Boundary Nodes
    # These are generated *between* the fixed nodes. 
    # They are allowed to move, but constrained to the geometry.
    sliding_points = resample_boundary_points(data['nodes'], data['faces'], h0)
    
    # Layer 3: Inner Cloud
    # These fill the void and move freely in 2D space.
    cloud_points = generate_inner_points(data['nodes'], data['faces'], h0)
    
    # 3. Stack the System
    #    Order matters! [Fixed] -> [Sliding] -> [Cloud]
    points = np.vstack((fixed_points, sliding_points, cloud_points))
    
    n_fixed   = len(fixed_points)
    n_sliding = len(sliding_points)
    n_inner   = len(cloud_points)
    total_n   = len(points)
    
    print("Nodes Breakdown:")
    print(f"  - Fixed (Corners): {n_fixed}")
    print(f"  - Sliding (Wall):  {n_sliding}")
    print(f"  - Inner (Cloud):   {n_inner}")
    print(f"  - Total:           {total_n}")
    
    # 4. Triangulate (Connectivity)
    #    We use Delaunay on the full set to establish neighbor links.
    print("Triangulating...")
    tri = Delaunay(points)
    simplices = tri.simplices
    
    # Filter bad triangles (Concave/Holes)
    centroids = np.mean(points[simplices], axis=1)
    mask = check_points_inside(centroids, data['nodes'], data['faces'])
    cells = simplices[mask]
    
    print(f"Generated {len(cells)} valid triangular cells.")

    # 5. Smooth (The Solver)
    print("Smoothing mesh (relaxing forces)...")
    
    # We pass the constraint data so the snapper knows what to do.
    # We also pass 'n_sliding' so it knows WHICH nodes to snap.
    smoothed_points = smooth_mesh(
        points.copy(), 
        cells, 
        data['nodes'], 
        data['faces'], 
        n_sliding,  # <--- Vital: Tells solver points [n_fixed : n_fixed+n_sliding] are constrained
        h0, 
        niters=50
    )

    # 6. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Plot 1: Initial State ---
    ax1.set_title("Initial Placement (Before Smoothing)")
    ax1.triplot(points[:,0], points[:,1], cells, 'k-', linewidth=0.5, alpha=0.3)
    
    # Color code the layers
    ax1.scatter(fixed_points[:,0], fixed_points[:,1], c='red', s=50, label='Fixed', zorder=10)
    ax1.scatter(sliding_points[:,0], sliding_points[:,1], c='lime', s=20, label='Sliding', zorder=9)
    ax1.scatter(cloud_points[:,0], cloud_points[:,1], c='blue', s=10, label='Inner', zorder=8)
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')

    # --- Plot 2: Smoothed State ---
    ax2.set_title("Relaxed Mesh (After Smoothing)")
    ax2.triplot(smoothed_points[:,0], smoothed_points[:,1], cells, 'k-', linewidth=0.5, alpha=0.5)
    
    # Extract layers from the smoothed array for plotting
    s_fixed = smoothed_points[:n_fixed]
    s_slide = smoothed_points[n_fixed : n_fixed + n_sliding]
    s_inner = smoothed_points[n_fixed + n_sliding :]
    
    ax2.scatter(s_fixed[:,0], s_fixed[:,1], c='red', s=50, zorder=10)
    ax2.scatter(s_slide[:,0], s_slide[:,1], c='lime', s=20, zorder=9)
    ax2.scatter(s_inner[:,0], s_inner[:,1], c='blue', s=10, zorder=8)
    ax2.set_aspect('equal')
    
    plt.savefig('snapmesh_final_result.png')
    print("Result saved to 'snapmesh_final_result.png'")