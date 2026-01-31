''' solver/euler.py '''
import numpy as np
from .physics import prim_to_cons, cons_to_prim
from .fluxes import compute_rusanov_flux
from mesh.connectivity import build_fvm_connectivity, map_boundary_faces
from mesh.geometry import compute_cell_metrics, compute_fvm_face_metrics
from viz.plotter import MeshPlotter

class EulerSolver:
    def __init__(self, points, cells, data, mode='axisymmetric'):
        self.points = points
        self.cells = cells
        self.mode = mode
        
        fluid = data['fluids'].get('air')
        self.R = fluid['R']
        self.gamma = fluid['gamma']
        
        # 1. Geometry & Metrics
        self.face_nodes, self.face_cells, self.cell_faces = build_fvm_connectivity(cells)
        # Capture the planar_areas return
        self.cell_vols, self.cell_centroids, self.planar_areas = compute_cell_metrics(points, cells, mode=mode)
        self.face_areas, self.face_normals, self.face_mids = compute_fvm_face_metrics(
            points, self.face_nodes, self.face_cells, self.cell_centroids, mode=mode
        )
        
        self.bc_groups = map_boundary_faces(points, self.face_nodes, self.face_cells, data['nodes'], data['faces'])
        self.bc_data = {row['id']: row for row in data['boundaries']}
        
        self._face_to_tag = np.full(len(self.face_areas), -1, dtype=int)
        for tag, f_list in self.bc_groups.items():
            self._face_to_tag[f_list] = tag
        
        self.Q = np.zeros((len(cells), 4))
        self.max_mach = 0.0

    def initialize_field(self, rho, u, v, p):
        """ Initializes the field with consistent total energy. """
        T_init = p / (rho * self.R)
        rho_c, rhou_c, rhov_c, rhoE_c = prim_to_cons(rho, u, v, p, T_init, self.gamma, self.R)
        self.Q[:, 0] = rho_c
        self.Q[:, 1] = rhou_c
        self.Q[:, 2] = rhov_c
        self.Q[:, 3] = rhoE_c

    def compute_time_step(self, cfl=0.45):
        """ Robust CFL calculation using cell height (Volume/Area). """
        rho, rhou, rhov, rhoE = self.Q[:,0], self.Q[:,1], self.Q[:,2], self.Q[:,3]
        _, u, v, p, T = cons_to_prim(rho, rhou, rhov, rhoE, self.gamma, self.R)
        
        # Stability protection for acoustic speed
        a = np.sqrt(np.maximum(self.gamma * self.R * T, 1.0))
        vel = np.sqrt(u**2 + v**2)
        
        # Find minimum characteristic length (cell altitude)
        cell_dx = np.full(len(self.cells), 1e10)
        for f_in_c in range(3):
            f_idx = self.cell_faces[:, f_in_c]
            cell_dx = np.minimum(cell_dx, self.cell_vols / (self.face_areas[f_idx] + 1e-12))
        
        self.max_mach = np.max(vel / a)
        return np.min(cfl * cell_dx / (vel + a + 1e-12))

    def step(self, dt):
        """ Advances the solution using Rusanov fluxes and mirror BCs. """
        residuals = np.zeros_like(self.Q)
        
        for f_idx in range(len(self.face_areas)):
            P, N = self.face_cells[f_idx]
            area, normal = self.face_areas[f_idx], self.face_normals[f_idx]
            
            if N != -1:
                flux = compute_rusanov_flux(self.Q[P], self.Q[N], normal, area, self.gamma, self.R)
                residuals[P] -= flux
                residuals[N] += flux
            else:
                tag = self._face_to_tag[f_idx]
                if tag == -1: continue 

                rho, rhou, rhov, rhoE = self.Q[P]
                
                if tag == 1: # STAGNATION INLET
                    bc = self.bc_data[tag]
                    rho_bc, rhou_bc, rhov_bc, rhoE_bc = prim_to_cons(
                        bc['p']/(self.R*bc['T']), bc['u'], bc['v'], bc['p'], bc['T'], self.gamma, self.R
                    )
                    Q_ghost = np.array([rho_bc, rhou_bc, rhov_bc, rhoE_bc])
                    flux = compute_rusanov_flux(self.Q[P], Q_ghost, normal, area, self.gamma, self.R)
                
                elif tag == 2 or tag == 4: # REFLECTION / SYMMETRY
                    u, v = rhou/rho, rhov/rho
                    vn = u * normal[0] + v * normal[1]
                    u_g, v_g = u - 2.0 * vn * normal[0], v - 2.0 * vn * normal[1]
                    Q_ghost = np.array([rho, rho * u_g, rho * v_g, rhoE])
                    flux = compute_rusanov_flux(self.Q[P], Q_ghost, normal, area, self.gamma, self.R)
                
                elif tag == 3: # EXTRAPOLATION
                    flux = compute_rusanov_flux(self.Q[P], self.Q[P], normal, area, self.gamma, self.R)
                
                residuals[P] -= flux

        if self.mode == 'axisymmetric':
            # Source Term: p * 2 * pi * Planar_Area
            rho, rhou, rhov, rhoE = self.Q[:,0], self.Q[:,1], self.Q[:,2], self.Q[:,3]
            _, _, _, p, _ = cons_to_prim(rho, rhou, rhov, rhoE, self.gamma, self.R)
            residuals[:, 2] += p * (2.0 * np.pi * self.planar_areas)

        self.Q += (dt / self.cell_vols[:, np.newaxis]) * residuals
        return np.sqrt(np.mean(residuals**2))


    def get_primitive(self, var='T'):
        """ Returns requested field for visualization. """
        rho, rhou, rhov, rhoE = self.Q[:,0], self.Q[:,1], self.Q[:,2], self.Q[:,3]
        rho, u, v, p, T = cons_to_prim(rho, rhou, rhov, rhoE, self.gamma, self.R)
        mapping = {'rho': rho, 'u': u, 'v': v, 'p': p, 'T': T}
        return mapping.get(var, T)

    def plot_results(self, variable='p', title="Results"):
        """ Visualization using the MeshPlotter. """
        field = self.get_primitive(variable)
        plotter = MeshPlotter(points=self.points, cells=self.cells)
        plotter.plot_scalar(self.points, self.cells, field, title=title)
        plotter.show()