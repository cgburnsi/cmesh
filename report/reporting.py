''' report/reporting.py '''
import numpy as np

class FVMReporter:
    @staticmethod
    def sys_init(calc_mode, data):
        """ Reports the initial state and loaded configuration. """
        print(f"--- DIAGNOSTIC: System Initialization ---")
        print(f"  - Calculation Mode: {calc_mode}")
        print(f"  - Nodes Loaded:     {len(data['nodes'])}")
        print(f"  - Faces Loaded:     {len(data['faces'])}")
        print(f"  - Boundary Tags:    {list(np.unique(data['boundaries']['id']))}")

    @staticmethod
    def mesh_stats(q_stats, cell_vols):
        """ Reports on the geometric integrity of the generated mesh. """
        print(f"\n--- DIAGNOSTIC: Mesh Quality & Geometry ---")
        print(f"  - Quality (Avg):    {q_stats['avg']:.4f}")
        print(f"  - Quality (Min):    {q_stats['min']:.4f}")
        print(f"  - Total Volume:     {np.sum(cell_vols):.6f}")

    @staticmethod
    def topology(face_cells):
        """ Reports the connectivity counts for the Finite Volume Method. """
        internal = np.sum(face_cells[:, 1] != -1)
        boundary = np.sum(face_cells[:, 1] == -1)
        print(f"\n--- DIAGNOSTIC: FVM Topology ---")
        print(f"  - Internal Faces:   {internal}")
        print(f"  - Boundary Faces:   {boundary}")

    @staticmethod
    def scalar_report(scalar_field, label="Temperature"):
        """ Reports the final range of the solved physical field. """
        print(f"\n--- FVM GRADIENT REPORT: {label} ---")
        print(f"  - Field Range:      {np.min(scalar_field):.2f}K to {np.max(scalar_field):.2f}K")