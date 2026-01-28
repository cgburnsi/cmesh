''' main.py '''
import numpy as np
from core import input_reader
from mesh import MeshGenerator
from viz import MeshPlotter
from mesh.smoothing import distmesh_smoother  
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics, compute_fvm_weights
from solver.steady_state import solve_diffusion

if __name__ == '__main__':
    
    # 1. Load Input Data
    data = input_reader('geom1.inp')
    calc_mode = data['settings'].get('mode', 'axisymmetric')
    
    print(f"--- DIAGNOSTIC: System Initialization ---")
    print(f"Calculation Mode: {calc_mode}")
    print(f"Nodes Loaded: {len(data['nodes'])}")
    print(f"Faces Loaded: {len(data['faces'])}")

    # 2. Extract BC values into a dictionary for the solver: {tag: value}
    bc_values = {row['id']: row['v'] for row in data['boundaries']}
    print(f"Boundary Conditions: {bc_values}")

    # 3. Mesh Generation
    print(f"\n--- DIAGNOSTIC: Mesh Generation ---")
    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    smoothed_points, final_cells = mesh.generate(niters=1000)
    
    q_vals, q_stats = mesh.get_quality(smoothed_points, final_cells)
    print(f"Mesh Quality - Avg: {q_stats['avg']:.4f}, Min: {q_stats['min']:.4f}")

    # 4. FVM Connectivity & Metrics
    print(f"\n--- DIAGNOSTIC: FVM Topology & Metrics ---")
    face_nodes, face_cells, cell_faces = build_fvm_connectivity(final_cells)
    
    # Pass 'calc_mode' to apply 2*pi*y scaling if axisymmetric
    cell_volumes, cell_centroids = compute_cell_metrics(smoothed_points, final_cells, mode=calc_mode)
    face_areas, face_normals, face_midpoints = compute_fvm_face_metrics(
        smoothed_points, face_nodes, face_cells, cell_centroids, mode=calc_mode
    )
    
    print(f"Total Domain Volume: {np.sum(cell_volumes):.6f}")
    print(f"Internal Faces: {np.sum(face_cells[:, 1] != -1)}")
    print(f"Boundary Faces: {np.sum(face_cells[:, 1] == -1)}")

    # 5. Boundary Mapping & Solver
    bc_groups = map_boundary_faces(smoothed_points, face_nodes, face_cells, data['nodes'], data['faces'])    
    d_PN, gx, vec_PN = compute_fvm_weights(face_cells, cell_centroids, face_midpoints)
    
    # Solve Diffusion
    T = solve_diffusion(
        face_cells, face_areas, d_PN, bc_groups, cell_volumes, k=1.0, bc_values=bc_values
    )
    
    # 6. Final Reports
    print(f"\n--- FVM GRADIENT REPORT ---")
    print(f"Temperature Range: {np.min(T):.2f}K to {np.max(T):.2f}K")
    
    # 7. Visualization
    plot_data = {'xv': smoothed_points[:, 0], 'yv': smoothed_points[:, 1], 'lcv': final_cells}
    
    # Mesh Structure
    mesh_view = MeshPlotter(arrays=plot_data)
    mesh_view.plot_edges()
    mesh_view.plot_nodes()
    mesh_view.save('snapmesh_sources.png')
    
    # Thermal Gradient
    thermal_view = MeshPlotter(arrays=plot_data)
    thermal_view.plot_scalar(smoothed_points, final_cells, T)
    thermal_view.save('thermal_gradient.png')
    
    thermal_view.show()