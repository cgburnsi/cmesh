''' viz/plotter.py '''
import matplotlib.pyplot as plt
import numpy as np

class MeshPlotter:
    def __init__(self, points=None, cells=None, arrays=None):
        # Support both the old dictionary style and direct array passing
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
        """ Plots the wireframe. """
        self.ax.triplot(self.xv, self.yv, self.lcv, color=color, lw=linewidth)

    def plot_nodes(self, color='k', size=2):
        """ Plots the vertices. """
        self.ax.scatter(self.xv, self.yv, s=size, c=color)

    def show(self):
        plt.show()

    def save(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")