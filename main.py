''' main.py '''
from core import input_reader
from mesh import MeshGenerator
from viz import MeshPlotter
from mesh.smoothing import spring_smoother, distmesh_smoother  # Import the one you want to test

if __name__ == '__main__':
    # 1. Load Data
    data = input_reader('geom1.inp')
    
    # 2. Generate Mesh
    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    #mesh = MeshGenerator(data, smoother=spring_smoother)
    smoothed_points, final_cells = mesh.generate(niters=1000)    
    
    q_values, stats = mesh.get_quality(smoothed_points, final_cells)
    print(f"--- MESH QUALITY REPORT ---")
    print(f"Minimum Quality: {stats['min']:.4f}")
    print(f"Average Quality: {stats['avg']:.4f}")
    print(f"Sliver Count (Q < 0.2): {len(stats['worst_indices'])}")
    
    # 3. Package for Plotter
    plot_data = {
        'xv':  smoothed_points[:, 0],
        'yv':  smoothed_points[:, 1],
        'lcv': final_cells
    }
    
    # 4. Visualization
    view = MeshPlotter(arrays=plot_data)
    view.plot_edges()
    view.plot_nodes()
    view.save('snapmesh_sources.png')
    view.show()
    