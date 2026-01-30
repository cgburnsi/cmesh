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
    
    # 1. Initialization & Thermodynamics
    data = input_reader('geom1.inp')
    calc_mode = data['settings'].get('mode', 'axisymmetric')
    FVMReporter.sys_init(calc_mode, data)
   
    # Fetch properties for the active fluid (e.g., 'air')
    fluid = data['fluids'].get('air')
    r_gas = fluid['R'] if fluid else 287.05
    gamma = fluid['gamma'] if fluid else 1.4
    
    # Extract Temperature BCs
    bc_values = {row['id']: row['T'] for row in data['boundaries']}
    
    # Diagnostic Inlet Density Calculation
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

    # 4. Solver Initialization
    # Pull global constant velocity from the boundary data
    u_init = data['boundaries']['u'][0]
    v_init = data['boundaries']['v'][0]
    
    u_field = np.zeros((len(face_areas), 2))
    u_field[:, 0] = u_init 
    u_field[:, 1] = v_init
    
    bc_groups = map_boundary_faces(points, face_nodes, face_cells, data['nodes'], data['faces'])    
    d_PN, _, _ = compute_fvm_weights(face_cells, cell_centroids, face_mids)
    
    # Initial Temperature Field
    T = np.full(len(cells), 300.0)
    k = 1.0            # Thermal conductivity
    cfl_target = 0.8   # Stability factor
    
    # Calculate characteristic length (dx) for CFL
    # For Axisymmetric, we derive planar area to get a 2D geometric constraint
    if calc_mode == 'axisymmetric':
        planar_areas = cell_vols / (2.0 * np.pi * cell_centroids[:, 1])
    else:
        planar_areas = cell_vols
    dx = np.sqrt(planar_areas)

    print(f"\n--- STARTING EXPLICIT LOOP (CFL CONTROL) ---")
    print(f"Applying boundary-defined flow: u={u_init}, v={v_init}") 
    
    # 5. Iterative Time-Marching Loop
    for step in range(15001):
        # --- A. Dynamic Time-Stepping (CFL) ---
        # a = sqrt(gamma * R * T)
        sos = np.sqrt(gamma * r_gas * T)
        vel_mag = np.sqrt(u_init**2 + v_init**2) 
        
        # dt = CFL * dx / (u + a)
        dt_local = cfl_target * dx / (vel_mag + sos)
        dt = np.min(dt_local)
        
        # Monitor Mach Number
        max_mach = np.max(vel_mag / sos)
        
        # --- B. Flux Calculation ---
        residuals = np.zeros_like(T)
        
        for f_idx in range(len(face_areas)):
            P, N = face_cells[f_idx]
            area = face_areas[f_idx]
            dist = d_PN[f_idx]
            
            # Diffusion Flux
            if N != -1:
                flux_diff = k * area * (T[N] - T[P]) / dist
            else:
                # Boundary mapping
                tag = -1
                for b_tag, faces in bc_groups.items():
                    if f_idx in faces:
                        tag = b_tag
                        break
                T_bc = bc_values.get(tag, T[P])
                flux_diff = k * area * (T_bc - T[P]) / dist

            # Advection Flux (Upwind)
            mass_flux = np.dot(u_field[f_idx], face_normals[f_idx]) * area
            if mass_flux > 0: # Out of P
                flux_adv = mass_flux * T[P]
            else: # Into P
                if N != -1:
                    flux_adv = mass_flux * T[N]
                else:
                    flux_adv = mass_flux * bc_values.get(tag, T[P])

            total_flux = flux_diff - flux_adv
            residuals[P] += total_flux
            if N != -1:
                residuals[N] -= total_flux

        # --- C. Update & Convergence ---
        T += (dt / cell_vols) * residuals
        
        # Use the updated reporter for monitoring convergence and stability
        res_norm = FVMReporter.residual_report(step, residuals, dt, max_mach)
        
        if res_norm < 1e-8:
            print(f"Converged at step {step}")
            break

    # 6. Final Reporting & Visualization
    FVMReporter.scalar_report(T)
    
    plot_data = {'xv': points[:, 0], 'yv': points[:, 1], 'lcv': cells}
    thermal_view = MeshPlotter(arrays=plot_data)
    thermal_view.plot_scalar(points, cells, T, title="CFL-Controlled Advection-Diffusion")
    thermal_view.show()