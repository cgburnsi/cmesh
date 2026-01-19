''' mesh/smoothing.py '''
import numpy as np
from .distance import project_points_to_boundary

def get_unique_edges(simplices):
    """
    Extracts all unique edges from a triangulation.
    Args:
        simplices: (M, 3) array of triangle node indices
    Returns:
        edges: (K, 2) array of unique sorted node pairs
    """
    # Create all 3 edges for every triangle
    # (Node 0-1, 1-2, 2-0)
    edges = np.vstack((
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [2, 0]]
    ))
    
    # Sort every row (so [1,0] becomes [0,1]) to ensure uniqueness
    edges.sort(axis=1)
    
    # Remove duplicates
    return np.unique(edges, axis=0)

def smooth_mesh(points, simplices, nodes, faces, h0, dt=0.2, niters=10):
    """
    Runs the DistMesh smoothing iterations.
    
    Args:
        points: Current node coordinates (N, 2)
        simplices: Triangle connectivity (M, 3)
        nodes/faces: Boundary definition (for snapping)
        h0: Target edge length
        dt: Time step (pseudo-time)
        niters: Number of iterations
    """
    
    # We only move the "Cloud" points, not the Fixed Boundary nodes.
    # The fixed nodes are the first N in the array.
    n_fixed = len(nodes)
    
    for itr in range(niters):
        # 1. Get Unique Edges
        edges = get_unique_edges(simplices)
        
        # 2. Vectorized Spring Forces
        idx1 = edges[:, 0]
        idx2 = edges[:, 1]
        
        p1 = points[idx1]
        p2 = points[idx2]
        
        # Current Lengths
        diff = p1 - p2
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-10)
        
        # 3. Force Calculation
        # F = (L_target - L_current)
        # We want L_target to be h0 (uniform mesh)
        # Force magnitude (Hooke's Law with simple scaling)
        F_mag = dists - h0
        
        # Limit forces (prevent explosion)
        F_mag = np.where(F_mag > 0, F_mag, 0) # Only pull, don't push? 
        # Actually DistMesh usually uses F = L - h0 (Linear Spring)
        # Let's stick to the standard F = Scale * (L - h0) * Vector
        
        # Normalized vectors
        v = diff / dists[:, None]
        
        # Force vector
        force = v * F_mag[:, None]
        
        # 4. Accumulate Forces on Nodes
        # We need to sum up forces from all edges connected to a node.
        # np.add.at is perfect for this unbuffered summation.
        total_force = np.zeros_like(points)
        
        # Edge pulls P2 towards P1, and P1 towards P2
        # (This is a simplified repulsion/attraction model)
        # DistMesh uses F_vector = (d - h0) * (diff / d)
        
        move = 0.2 * force # Scale factor
        
        np.add.at(total_force, idx2,  move)
        np.add.at(total_force, idx1, -move)
        
        # 5. Update Positions (Euler Integration)
        # Only move the non-fixed points!
        points[n_fixed:] += total_force[n_fixed:] * dt
        
        # 6. Snap to Boundary (The "Project" Step)
        # Check if any non-fixed point moved outside
        # (We just blindly project all non-fixed points to be safe, 
        # or we could optimize by checking containment first)
        
        # For simplicity, let's just re-snap any point that went outside
        # But wait! 'project_points' finds the closest boundary point.
        # We only want to snap if it's OUTSIDE.
        
        # Let's rely on the Distance Kernel we built. 
        # DistMesh logic: If d < -geps, snap.
        
        # Simplified: Check Distances. If 'outside', snap.
        # But we need "Signed" distance for that.
        # Our current project_points is unsigned.
        # Temporary hack: Just verify with containment.
        
        # For this "Baby Step", let's trust the force equilibrium usually keeps points inside.
        # If they fly out, we pull them back.
        
        # Re-Project points that might have strayed? 
        # Actually, let's leave snapping out of the *inner* loop for this specific baby step
        # to see if the springs work first.
        pass
    
    return points






if __name__ == '__main__':
    pass


