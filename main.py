''' main.py '''
import numpy as np
from core import input_reader
from mesh import MeshGenerator
from viz import MeshPlotter
from mesh.smoothing import spring_smoother, distmesh_smoother  
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics


if __name__ == '__main__':
    # 1. Load Data
    data = input_reader('geom1.inp')
    
    # 2. Generate Mesh
    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    #mesh = MeshGenerator(data, smoother=spring_smoother)
    smoothed_points, final_cells = mesh.generate(niters=100)    
    

    
    # NEW: Build Connectivity
    face_nodes, face_cells, cell_faces = build_fvm_connectivity(final_cells)
    
    n_internal = np.sum(face_cells[:, 1] != -1)
    n_boundary = np.sum(face_cells[:, 1] == -1)
    
    print(f"\n--- CONNECTIVITY REPORT ---")
    print(f"Total Cells: {len(final_cells)}")
    print(f"Total Faces: {len(face_nodes)}")
    print(f"  - Internal Faces: {n_internal}")
    print(f"  - Boundary Faces: {n_boundary}")
    
    
    cell_areas, cell_centroids = compute_cell_metrics(smoothed_points, final_cells)
    face_lengths, face_normals, face_midpoints = compute_fvm_face_metrics(
        smoothed_points, face_nodes, face_cells, cell_centroids
    )
    
    total_area = np.sum(cell_areas)
    print(f"\n--- FVM GEOMETRY REPORT ---")
    print(f"Total Mesh Area: {total_area:.4f}")
    print(f"Total Cell Count: {len(final_cells)}")
    print(f"Mean Face Length: {np.mean(face_lengths):.4f}")
    
    
    
    q_values, stats = mesh.get_quality(smoothed_points, final_cells)
    print(f"\n--- MESH QUALITY REPORT ---")
    print(f"Minimum Quality: {stats['min']:.4f}")
    print(f"Average Quality: {stats['avg']:.4f}")
    print(f"Sliver Count (Q < 0.2): {len(stats['worst_indices'])}")
    
    if stats['min'] < 0.01:
        bad_indices = np.where(q_values < 0.01)[0]
        print(f"Degenerate Cell Indices: {bad_indices}")
        # The vertices of the first bad cell
        print(f"Vertices of Cell {bad_indices[0]}: \n{smoothed_points[final_cells[bad_indices[0]]]}")
    
    
    bc_groups = map_boundary_faces(
        smoothed_points, 
        face_nodes, 
        face_cells, 
        data['nodes'], 
        data['faces']
    )    
    print(f"\n--- BOUNDARY CONDITION REPORT ---")
    for tag, indices in bc_groups.items():
        print(f"  - Tag {tag}: {len(indices)} Faces")
        
        
    
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
    