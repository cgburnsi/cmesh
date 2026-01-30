''' main.py '''
import numpy as np
from core import input_reader
from report.reporting import FVMReporter
from mesh import MeshGenerator
from viz import MeshPlotter
from mesh.smoothing import distmesh_smoother  
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics, compute_fvm_weights
from solver.steady_state import solve_advection_diffusion

if __name__ == '__main__':
    
    # 1. Initialization
    data = input_reader('geom1.inp')
    calc_mode = data['settings'].get('mode', 'axisymmetric')
    FVMReporter.sys_init(calc_mode, data)
   
    # 1. Access Fluid Properties
    # Assuming 'air' is your active fluid for this run
    fluid = data['fluids'].get('air')
    r_gas = fluid['R'] if fluid else 287.05
    
    # 2. Extract BC values specifically from the 'T' field
    bc_values = {row['id']: row['T'] for row in data['boundaries']}
    
    # 3. Calculate Inlet Density for Diagnostic (Optional)
    p_inlet = data['boundaries']['p'][0]
    t_inlet = data['boundaries']['T'][0]
    rho_inlet = p_inlet / (r_gas * t_inlet)
    print(f"Calculated Inlet Density: {rho_inlet:.4f} kg/m^3")
    

    # 2. Mesh Generation
    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    points, cells = mesh.generate(niters=1000)
    _, q_stats = mesh.get_quality(points, cells)

    # 3. FVM Topology & Metrics
    face_nodes, face_cells, cell_faces = build_fvm_connectivity(cells)
    cell_vols, cell_centroids = compute_cell_metrics(points, cells, mode=calc_mode)
    face_areas, face_normals, face_mids = compute_fvm_face_metrics(
        points, face_nodes, face_cells, cell_centroids, mode=calc_mode
    )
    FVMReporter.mesh_stats(q_stats, cell_vols)
    FVMReporter.topology(face_cells)

    # 4. Phase 2 Setup: Dynamic Velocity Field
    # Pull global constant velocity from the first boundary entry (Inlet)
    u_init = data['boundaries']['u'][0]
    v_init = data['boundaries']['v'][0]
    
    u_field = np.zeros((len(face_areas), 2))
    u_field[:, 0] = u_init 
    u_field[:, 1] = v_init
    
    print(f"Applying boundary-defined flow: u={u_init}, v={v_init}") 
    
    # 5. Mapping and Solver
    bc_values = {row['id']: row['v'] for row in data['boundaries']}
    bc_groups = map_boundary_faces(points, face_nodes, face_cells, data['nodes'], data['faces'])    
    d_PN, _, _ = compute_fvm_weights(face_cells, cell_centroids, face_mids)
    
    T = solve_advection_diffusion(
        face_cells, face_areas, face_normals, d_PN, bc_groups, 
        cell_vols, u_field, k=1.0, bc_values=bc_values
    )
    FVMReporter.scalar_report(T)
    
    # 6. Visualization
    plot_data = {'xv': points[:, 0], 'yv': points[:, 1], 'lcv': cells}
    thermal_view = MeshPlotter(arrays=plot_data)
    thermal_view.plot_scalar(points, cells, T, title=f"Advection-Diffusion (Axisymmetric)")
    thermal_view.show()