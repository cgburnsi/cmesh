''' main.py '''
import numpy as np
from core import input_reader
from mesh import MeshGenerator
from viz import MeshPlotter
from mesh.smoothing import spring_smoother, distmesh_smoother  
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics, compute_fvm_weights
from solver.steady_state import solve_diffusion

if __name__ == '__main__':
    data = input_reader('geom1.inp')
    
    data = input_reader('geom1.inp')

    # Extract BC values into a dictionary for the solver: {tag: value}
    bc_values = {row['id']: row['v'] for row in data['boundaries']}

    calc_mode = data['settings'].get('mode', 'axisymmetric')
    print(f"Calculation Mode: {calc_mode}")

    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    smoothed_points, final_cells = mesh.generate(niters=1000)    
    
    face_nodes, face_cells, cell_faces = build_fvm_connectivity(final_cells)
    cell_areas, cell_centroids = compute_cell_metrics(smoothed_points, final_cells)
    face_lengths, face_normals, face_midpoints = compute_fvm_face_metrics(
        smoothed_points, face_nodes, face_cells, cell_centroids
    )
    
    bc_groups = map_boundary_faces(smoothed_points, face_nodes, face_cells, data['nodes'], data['faces'])    
    d_PN, gx, vec_PN = compute_fvm_weights(face_cells, cell_centroids, face_midpoints)
    
    # 1. Define Temperature Gradient (800K Inlet, 300K Walls/Outlet)
    boundary_conditions = {1: 800.0, 2: 300.0, 3: 100.0}
    T = solve_diffusion(face_cells, face_lengths, d_PN, bc_groups, k=1.0, bc_values=boundary_conditions)
    
    # 2. Print Reports
    print(f"\n--- FVM GRADIENT REPORT ---")
    print(f"  - Temperature Range: {np.min(T):.2f}K to {np.max(T):.2f}K")
    
    # 3. Visualization - Mesh Structure
    plot_data = {'xv': smoothed_points[:, 0], 'yv': smoothed_points[:, 1], 'lcv': final_cells}
    mesh_view = MeshPlotter(arrays=plot_data)
    mesh_view.plot_edges()
    mesh_view.plot_nodes()
    mesh_view.save('snapmesh_sources.png')
    
    # 4. Visualization - Thermal Gradient
    thermal_view = MeshPlotter(arrays=plot_data)
    thermal_view.plot_scalar(smoothed_points, final_cells, T)
    thermal_view.save('thermal_gradient.png')
    
    # Final step: Show all open figures
    thermal_view.show()