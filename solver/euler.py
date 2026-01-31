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
        
        # Fluid properties
        fluid = data['fluids'].get('air')
        self.R = fluid['R']
        self.gamma = fluid['gamma']
        
        # 1. Geometry & Metrics
        self.face_nodes, self.face_cells, self.cell_faces = build_fvm_connectivity(cells)
        self.cell_vols, self.cell_centroids = compute_cell_metrics(points, cells, mode=mode)
        self.face_areas, self.face_normals, self.face_mids = compute_fvm_face_metrics(
            points, self.face_nodes, self.face_cells, self.cell_centroids, mode=mode
        )
        
        # Boundary mapping
        self.bc_groups = map_boundary_faces(points, self.face_nodes, self.face_cells, data['nodes'], data['faces'])
        self.bc_data = {row['id']: row for row in data['boundaries']}
        
        # 2. State Initialization (Conserved Variables: rho, rhou, rhov, rhoE)
        self.Q = np.zeros((len(cells), 4))
        self.max_mach = 0.0

    def initialize_field(self, rho, u, v, p):
        """ Sets a uniform initial state across the domain. """
        rho, rhou, rhov, rhoE = prim_to_cons(rho, u, v, p, p/(rho*self.R), self.gamma, self.R)
        self.Q[:, 0] = rho
        self.Q[:, 1] = rhou
        self.Q[:, 2] = rhov
        self.Q[:, 3] = rhoE

    def compute_time_step(self, cfl=0.5):
        """ Calculates global dt based on acoustic wave speeds. """
        rho, u, v, p, T = cons_to_prim(self.Q[:,0], self.Q[:,1], self.Q[:,2], self.Q[:,3], self.gamma, self.R)
        a = np.sqrt(self.gamma * self.R * T)
        vel = np.sqrt(u**2 + v**2)
        
        # Calculate local dx based on planar area (sqrt(Area))
        planar_area = self.cell_vols / (2.0 * np.pi * self.cell_centroids[:, 1] if self.mode == 'axisymmetric' else 1.0)
        dx = np.sqrt(planar_area)
        
        self.max_mach = np.max(vel / a)
        return np.min(cfl * dx / (vel + a))

    def step(self, dt):
        """ Performs one time-step using Rusanov fluxes and source terms. """
        residuals = np.zeros_like(self.Q)
        
        # A. Internal Face Fluxes
        for f_idx in range(len(self.face_areas)):
            P, N = self.face_cells[f_idx]
            area = self.face_areas[f_idx]
            normal = self.face_normals[f_idx]
            
            if N != -1:
                # Standard Internal Flux
                flux = compute_rusanov_flux(self.Q[P], self.Q[N], normal, area, self.gamma, self.R)
                residuals[P] -= flux
                residuals[N] += flux
            else:
                # B. Boundary Fluxes
                tag = -1
                for b_tag, f_list in self.bc_groups.items():
                    if f_idx in f_list: tag = b_tag; break
                
                bc = self.bc_data[tag]
                if bc['type'] == 1: # Dirichlet (Stagnation Inlet or Wall)
                    rho_bc, rhou_bc, rhov_bc, rhoE_bc = prim_to_cons(
                        bc['p']/(self.R*bc['T']), bc['u'], bc['v'], bc['p'], bc['T'], self.gamma, self.R
                    )
                    Q_bc = np.array([rho_bc, rhou_bc, rhov_bc, rhoE_bc])
                    flux = compute_rusanov_flux(self.Q[P], Q_bc, normal, area, self.gamma, self.R)
                else: # Neumann (Outlet - Extrapolation)
                    flux = compute_rusanov_flux(self.Q[P], self.Q[P], normal, area, self.gamma, self.R)
                
                residuals[P] -= flux

        # C. Axisymmetric Source Term
        if self.mode == 'axisymmetric':
            # Radial momentum source: p/r integrated over volume
            _, _, _, p, _ = cons_to_prim(self.Q[:,0], self.Q[:,1], self.Q[:,2], self.Q[:,3], self.gamma, self.R)
            # Source = [0, 0, P * Planar_Area * 2*pi, 0]
            planar_area = self.cell_vols / (2.0 * np.pi * self.cell_centroids[:, 1])
            residuals[:, 2] += p * (2.0 * np.pi * planar_area)

        # D. Update State
        self.Q += (dt / self.cell_vols[:, np.newaxis]) * residuals
        return np.sqrt(np.mean(residuals**2))

    def get_primitive(self, var='T'):
        """ Returns requested primitive field for reporting or plotting. """
        rho, u, v, p, T = cons_to_prim(self.Q[:,0], self.Q[:,1], self.Q[:,2], self.Q[:,3], self.gamma, self.R)
        mapping = {'rho': rho, 'u': u, 'v': v, 'p': p, 'T': T}
        return mapping.get(var, T)

    def plot_results(self, variable='p', title="Results"):
        """ Direct visualization from the solver state. """
        field = self.get_primitive(variable)
        plotter = MeshPlotter(points=self.points, cells=self.cells)
        plotter.plot_scalar(self.points, self.cells, field, title=title)
        plotter.show()