''' viz/plotter.py '''
import matplotlib.pyplot as plt
import numpy as np

class MeshPlotter:
    def __init__(self, points=None, cells=None, arrays=None):
        if arrays is not None:
            self.xv = arrays['xv']
            self.yv = arrays['yv']
            self.lcv = arrays['lcv']
        else:
            self.xv = points[:, 0]
            self.yv = points[:, 1]
            self.lcv = cells
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect('equal')
        self.ax.set_title("Generated Mesh")

    def plot_edges(self, color='k', linewidth=0.5):
        """ Plots the wireframe of the generated mesh. """
        self.ax.triplot(self.xv, self.yv, self.lcv, color=color, lw=linewidth)

    def plot_nodes(self, color='k', size=2):
        """ Plots the vertices of the generated mesh. """
        self.ax.scatter(self.xv, self.yv, s=size, c=color)

    def plot_geometry(self, nodes, color='red', marker='P', size=60, label_nodes=True):
        """ 
        Plots the original input geometry points.
        'nodes' should be the structured array from data['nodes'].
        """
        self.ax.scatter(nodes['x'], nodes['y'], s=size, c=color, marker=marker, 
                        label='Geometry Nodes', zorder=10)
        
        if label_nodes:
            for node in nodes:
                self.ax.text(node['x'], node['y'], f" {node['id']}", 
                             color=color, fontsize=9, fontweight='bold',
                             va='bottom', ha='left', zorder=11)
        self.ax.legend()

    def show(self):
        plt.show()

    def save(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        
    def plot_scalar(self, points, cells, scalar, title="Temperature Field", cmap='viridis'):
        tpc = self.ax.tripcolor(points[:, 0], points[:, 1], cells, 
                                facecolors=scalar, cmap=cmap, edgecolors='none', alpha=0.9)
        self.fig.colorbar(tpc, ax=self.ax, label='Value')
        self.ax.set_title(title)
        return tpc
    
    def plot_constraints(self, nodes, faces, constraints, color='green', linewidth=2):
        """ Draws the analytical curves (arcs/lines) for verification. """
        constraint_map = {c['id']: c for c in constraints}
        
        for face in faces:
            n1, n2 = nodes[face['n1']-1], nodes[face['n2']-1]
            p1 = np.array([n1['x'], n1['y']])
            p2 = np.array([n2['x'], n2['y']])
            
            const = constraint_map.get(face['ctag'])
            if const is not None and const['type'] == 2: # Circle/Arc
                cx, cy, R = const['p1'], const['p2'], const['p3']
                t1, t2 = np.arctan2(p1[1]-cy, p1[0]-cx), np.arctan2(p2[1]-cy, p2[0]-cx)
                
                # Handle shortest path
                dt = t2 - t1
                if dt > np.pi: dt -= 2*np.pi
                if dt < -np.pi: dt += 2*np.pi
                
                t_fine = np.linspace(t1, t1 + dt, 50)
                self.ax.plot(cx + R*np.cos(t_fine), cy + R*np.sin(t_fine), 
                             color=color, lw=linewidth, label='Arc Constraint' if face['id']==2 else "")
            else:
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', ls='--', lw=1)