''' main.py '''
import numpy as np
from core import input_reader
from report.reporting import FVMReporter
from mesh import MeshGenerator
from mesh.smoothing import distmesh_smoother  
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics, compute_fvm_weights

if __name__ == '__main__':
    
    # 1. Setup & Data Loading
    data = input_reader('geom1.inp')
    calc_mode = data['settings'].get('mode', 'axisymmetric')
    FVMReporter.sys_init(calc_mode, data)
   
    fluid = data['fluids'].get('air')
    r_gas = fluid['R'] if fluid else 287.05
    bc_values = {row['id']: row['T'] for row in data['boundaries']}

    # 2. Mesh & Geometry
    mesh = MeshGenerator(data, smoother=distmesh_smoother)
    points, cells = mesh.generate(niters=100)
    _, q_stats = mesh.get_quality(points, cells)

    face_nodes, face_cells, cell_faces = build_fvm_connectivity(cells)
    cell_vols, cell_centroids = compute_cell_metrics(points, cells, mode=calc_mode)
    face_areas, face_normals, face_mids = compute_fvm_face_metrics(
        points, face_nodes, face_cells, cell_centroids, mode=calc_mode
    )
    FVMReporter.mesh_stats(q_stats, cell_vols)
    FVMReporter.topology(face_cells)

    # 3. Explicit Solver Initialization
    u_init = data['boundaries']['u'][0]
    v_init = data['boundaries']['v'][0]
    u_field = np.zeros((len(face_areas), 2))
    u_field[:, 0], u_field[:, 1] = u_init, v_init
    
    bc_groups = map_boundary_faces(points, face_nodes, face_cells, data['nodes'], data['faces'])    
    d_PN, _, _ = compute_fvm_weights(face_cells, cell_centroids, face_mids)
    
    # Initialize Field (Ambient 300K)
    T = np.full(len(cells), 300.0)
    k = 1.0        # Thermal conductivity
    cfl = 0.5      # Stability constant
    dt = 1e-4      # Initial time step (will be dynamic in Phase 3)
    
    print(f"\n--- STARTING EXPLICIT TIME-MARCHING ---")
    
    # 4. Explicit Time-Marching Loop
    for step in range(10001):
        residuals = np.zeros_like(T)
        
        # Calculate Fluxes for all Internal and Boundary Faces
        for f_idx in range(len(face_areas)):
            P, N = face_cells[f_idx]
            area = face_areas[f_idx]
            dist = d_PN[f_idx]
            
            # --- Diffusion Flux ---
            if N != -1:
                # Internal: k * Area * (T_N - T_P) / dist
                flux_diff = k * area * (T[N] - T[P]) / dist
            else:
                # Boundary: Find which tag this face belongs to
                tag = -1
                for b_tag, faces in bc_groups.items():
                    if f_idx in faces:
                        tag = b_tag
                        break
                
                T_bc = bc_values.get(tag, T[P])
                # Boundary flux (Dirichlet assumption for now)
                flux_diff = k * area * (T_bc - T[P]) / dist

            # --- Advection Flux (Upwind) ---
            mass_flux = np.dot(u_field[f_idx], face_normals[f_idx]) * area
            if mass_flux > 0: # Flow from P to N (or out of domain)
                flux_adv = mass_flux * T[P]
            else: # Flow from N to P (or into domain)
                if N != -1:
                    flux_adv = mass_flux * T[N]
                else:
                    flux_adv = mass_flux * bc_values.get(tag, T[P])

            # Accumulate Net Flux (Residuals)
            total_flux = flux_diff - flux_adv
            residuals[P] += total_flux
            if N != -1:
                residuals[N] -= total_flux

        # 5. Update the Field
        T += (dt / cell_vols) * residuals
        
        # 6. Monitor Convergence
        res_norm = FVMReporter.residual_report(step, residuals)
        if res_norm < 1e-9:
            print(f"Converged at step {step}")
            break

    FVMReporter.scalar_report(T)
    
    # 7. Final Visualization
    from viz import MeshPlotter
    plot_data = {'xv': points[:, 0], 'yv': points[:, 1], 'lcv': cells}
    thermal_view = MeshPlotter(arrays=plot_data)
    thermal_view.plot_scalar(points, cells, T, title="Explicit Advection-Diffusion")
    thermal_view.show()