''' main.py '''
import numpy as np
from core import input_reader
from report.reporting import FVMReporter
from mesh import MeshGenerator
from viz import MeshPlotter
from mesh.smoothing import distmesh_smoother  
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics, compute_fvm_weights

if __name__ == '__main__':
    
    # 1. Initialization
    data = input_reader('geom1.inp')
    calc_mode = data['settings'].get('mode', 'axisymmetric')
    FVMReporter.sys_init(calc_mode, data)
   
    fluid = data['fluids'].get('air')
    r_gas = fluid['R'] if fluid else 287.05
    gamma = fluid['gamma'] if fluid else 1.4
    bc_values = {row['id']: row['T'] for row in data['boundaries']}
    
    # 2. Mesh Generation
    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    points, cells = mesh.generate(niters=1000)
    _, q_stats = mesh.get_quality(points, cells)

    # --- SEPARATE MESH & GEOMETRY VERIFICATION PLOT ---
    print("\n--- MESH VERIFICATION: Close plot window to start FVM solver ---")
    mv_data = {'xv': points[:, 0], 'yv': points[:, 1], 'lcv': cells}
    mesh_view = MeshPlotter(arrays=mv_data)
    mesh_view.plot_edges(color='blue', linewidth=0.3)
    mesh_view.plot_constraints(data['nodes'], data['faces'], data['constraints'])
    mesh_view.plot_geometry(data['nodes'], label_nodes=True)
    mesh_view.ax.set_title("Generated Mesh & Geometry Verification")
    mesh_view.show()

    # 3. FVM Topology & Metrics
    face_nodes, face_cells, cell_faces = build_fvm_connectivity(cells)
    cell_vols, cell_centroids = compute_cell_metrics(points, cells, mode=calc_mode)
    face_areas, face_normals, face_mids = compute_fvm_face_metrics(
        points, face_nodes, face_cells, cell_centroids, mode=calc_mode
    )
    FVMReporter.mesh_stats(q_stats, cell_vols)
    FVMReporter.topology(face_cells)

    # 4. Solver Initialization
    u_init, v_init = data['boundaries']['u'][0], data['boundaries']['v'][0]
    u_field = np.zeros((len(face_areas), 2))
    u_field[:, 0], u_field[:, 1] = u_init, v_init
    
    bc_groups = map_boundary_faces(points, face_nodes, face_cells, data['nodes'], data['faces'])    
    d_PN, _, _ = compute_fvm_weights(face_cells, cell_centroids, face_mids)
    
    T = np.full(len(cells), 300.0)
    k, cfl_target = 1.0, 0.5  # Reduced CFL for initial stability
    
    if calc_mode == 'axisymmetric':
        planar_areas = cell_vols / (2.0 * np.pi * cell_centroids[:, 1])
    else:
        planar_areas = cell_vols
    dx = np.sqrt(planar_areas)

    print(f"\n--- STARTING FVM SOLVER LOOP ---")
    
    # 5. Iterative Time-Marching Loop
    for step in range(15001):
        # --- A. STABILITY FIX: Dynamic Time-Step (CFL + Diffusion) ---
        # Add temperature floor to prevent invalid sqrt in sos
        T_safe = np.maximum(T, 1e-6)
        sos = np.sqrt(gamma * r_gas * T_safe)
        vel_mag_local = np.sqrt(u_init**2 + v_init**2) 
        
        # Convective Limit
        dt_conv = np.min(cfl_target * dx / (vel_mag_local + sos))
        
        # Diffusive Limit: dt < 0.5 * dx^2 / k
        dt_diff = np.min(0.4 * (dx**2) / k) 
        
        dt = min(dt_conv, dt_diff)
        
        residuals = np.zeros_like(T)
        for f_idx in range(len(face_areas)):
            P, N = face_cells[f_idx]
            area, dist = face_areas[f_idx], d_PN[f_idx]
            
            # Diffusion Flux
            if N != -1:
                flux_diff = k * area * (T[N] - T[P]) / dist
            else:
                tag = -1
                for b_tag, f_list in bc_groups.items():
                    if f_idx in f_list: tag = b_tag; break
                T_bc = bc_values.get(tag, T[P])
                flux_diff = k * area * (T_bc - T[P]) / dist

            # --- B. LOGIC FIX: Upwind Advection with BC Support ---
            mass_flux = np.dot(u_field[f_idx], face_normals[f_idx]) * area
            if mass_flux > 0:
                # Flow leaving cell P
                flux_adv = mass_flux * T[P]
            else:
                # Flow entering cell P
                if N != -1:
                    flux_adv = mass_flux * T[N]
                else:
                    # Use specified boundary temperature for inflows
                    flux_adv = mass_flux * T_bc

            residuals[P] += (flux_diff - flux_adv)
            if N != -1: residuals[N] -= (flux_diff - flux_adv)

        # Update and monitor convergence
        T += (dt / cell_vols) * residuals
        res_norm = FVMReporter.residual_report(step, residuals, dt, np.max(vel_mag_local/sos))
        
        if res_norm < 1e-8:
            print(f"Converged at step {step}")
            break
        if np.isnan(res_norm):
            print(f"Solver diverged at step {step}")
            break

    # 6. Final Reporting & Visualization
    FVMReporter.scalar_report(T)
    result_view = MeshPlotter(points=points, cells=cells)
    result_view.plot_scalar(points, cells, T, title="Final Temperature Distribution")
    result_view.show()