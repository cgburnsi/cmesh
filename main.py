''' main.py '''
import numpy as np
from core import input_reader
from report.reporting import FVMReporter
from mesh import MeshGenerator
from solver.euler import EulerSolver

if __name__ == '__main__':
    
    # 1. Initialization & Data Loading
    data = input_reader('geom1.inp')
    calc_mode = data['settings'].get('mode', 'axisymmetric')
    FVMReporter.sys_init(calc_mode, data)
   
    fluid = data['fluids'].get('air')
    r_gas = fluid['R']
    gamma = fluid['gamma']
    
    # 2. Mesh Generation
    mesh = MeshGenerator(data)
    points, cells = mesh.generate(niters=300, n_layers=3)
    _, q_stats = mesh.get_quality(points, cells)

    # 3. Solver Setup & Topology Diagnostics
    solver = EulerSolver(points, cells, data, mode=calc_mode)
    
    # Restore Diagnostic Printing
    FVMReporter.mesh_stats(q_stats, solver.cell_vols)
    FVMReporter.topology(solver.face_cells)

    # 4. Automated Initial Conditions
    inlet_bc = data['boundaries'][data['boundaries']['id'] == 1][0]
    u_init, v_init, p_init, t_init = inlet_bc['u'], inlet_bc['v'], inlet_bc['p'], inlet_bc['T']
    rho_init = p_init / (r_gas * t_init)
    
    solver.initialize_field(rho=rho_init, u=u_init, v=v_init, p=p_init)
    
    print(f"\n--- STARTING INVISCID EULER SOLVER ---")

    # 5. Execution Loop
    for step in range(20001):
        dt = solver.compute_time_step(cfl=0.5)
        residual = solver.step(dt)
        
        # Periodic Progress Report
        if step % 500 == 0:
            FVMReporter.residual_report(step, residual, dt, solver.max_mach)
            
        if residual < 1e-8:
            print(f"Converged at step {step}")
            break
        if np.isnan(residual):
            print(f"Solver diverged at step {step}")
            break

    # 6. Final Reporting & Visualization
    FVMReporter.scalar_report(solver.get_primitive('T'), label="Temperature")
    solver.plot_results(variable="p", title="Static Pressure Distribution")