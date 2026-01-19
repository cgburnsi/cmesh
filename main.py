''' main.py '''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri  # <--- This line defines 'mtri'
from core import input_reader
from mesh import generate_initial_points
from mesh import generate_initial_points, triangulate_and_filter

from mesh.smoothing import smooth_mesh




#import matplotlib.pyplot as plt
#import numpy as np
#from core import input_reader
#from mesh import compute_face_metrics, project_points_to_boundary

def plot_results(nodes, faces, query_points, snapped_points):
    """
    Visualizes the boundary, query points, and their snapped locations.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # --- 1. Plot the Boundary Faces ---
    # Create a quick lookup for node coordinates: ID -> (x, y)
    node_map = {n['id']: (n['x'], n['y']) for n in nodes}
    
    # Loop over faces and plot segments
    for i, face in enumerate(faces):
        n1_id, n2_id = face['n1'], face['n2']
        p1 = node_map[n1_id]
        p2 = node_map[n2_id]
        
        # Plot line segment
        # Only label the first one to avoid messing up the legend
        label = 'Boundary' if i == 0 else ""
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2, label=label)

    # --- 2. Plot Query Points (Red) ---
    ax.scatter(query_points[:, 0], query_points[:, 1], 
               c='red', s=80, label='Query Points', zorder=5)

    # --- 3. Plot Snapped Points (Blue) ---
    ax.scatter(snapped_points[:, 0], snapped_points[:, 1], 
               c='blue', s=80, label='Snapped Points', zorder=5)

    # --- 4. Draw Connecting Arrows ---
    for i in range(len(query_points)):
        qx, qy = query_points[i]
        sx, sy = snapped_points[i]
        
        # Draw dashed arrow from Query -> Snapped
        ax.arrow(qx, qy, sx-qx, sy-qy, 
                 head_width=0.03, length_includes_head=True, 
                 fc='gray', ec='gray', linestyle='--', alpha=0.6)

    # --- Styling ---
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_title("Snapmesh: Distance Kernel Projection")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Auto-scale with some padding
    all_x = np.concatenate((query_points[:,0], snapped_points[:,0]))
    all_y = np.concatenate((query_points[:,1], snapped_points[:,1]))
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)





from mesh.initialization import resample_boundary # Make sure to import or update generate_initial_points

if __name__ == '__main__':
    data = input_reader('geom1.inp')
    h0 = 0.1
    
    print(f"Generating points with boundary resampling (h0={h0})...")
    
    # 1. Resample the boundary explicitly
    bnd_points = resample_boundary(data['nodes'], data['faces'], h0)
    print(f"Created {len(bnd_points)} boundary nodes.")
    
    # 2. Generate the inner cloud (using your existing function, but modified to not duplicate boundary)
    #    Actually, let's just use the updated generate_initial_points if you pasted the code above.
    #    OR, here is how to do it manually in main.py to test:
    
    cloud = generate_initial_points(data['nodes'], data['faces'], h0) 
    # Note: If you updated generate_initial_points as above, 'cloud' now contains EVERYTHING.
    
    # 3. Triangulate
    #    We pass an empty list for 'fixed_nodes' because 'cloud' now has them all!
    #    We rely on the cloud containing the boundary.
    empty_fixed = np.array([], dtype=data['nodes'].dtype)
    
    points, cells = triangulate_and_filter(data['nodes'], cloud, data['faces'])
    
    # ... Run smoothing ...
    smooth_points = smooth_mesh(points.copy(), cells, data['nodes'], data['faces'], h0, niters=50)

    # ... Plotting code ...

    # 5. Visualize Both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    
    # Plot Initial
    ax1.triplot(points[:,0], points[:,1], cells, 'k-', alpha=0.3)
    ax1.set_title("Initial")
    
    # Plot Smoothed
    # Note: We should re-triangulate after moving points for perfect results, 
    # but for small movements, the topology is fine.
    ax2.triplot(smooth_points[:,0], smooth_points[:,1], cells, 'k-', alpha=0.5)
    ax2.scatter(smooth_points[:,0], smooth_points[:,1], s=5, c='blue')
    ax2.set_title("Smoothed (Relaxed)")


    
    
    
    
    
    