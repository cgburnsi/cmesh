''' mesh/sizing.py '''
import numpy as np

class SizingField:
    def __init__(self, sources_array):
        self.sources = sources_array
        
        # 1. Background H
        globals = self.sources[self.sources['type'] == 0]
        if len(globals) > 0:
            self.h_background = globals[0]['h']
        else:
            self.h_background = 1.0

    def __call__(self, points):
        """ Evaluates h at points (N, 2). Returns (N,) """
        n_points = len(points)
        
        # Start with the global background size everywhere
        h_values = np.full(n_points, self.h_background)
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Growth Rate (Slope)
        # 0.2 means size increases by 0.2 units for every 1.0 unit of distance.
        # This prevents the "Cliff Edge" effect.
        GROWTH_RATE = 0.05
        
        # 2. Apply Box Sources with Decay
        boxes = self.sources[self.sources['type'] == 1]
        for box in boxes:
            x1, y1, x2, y2, h_target = box['x1'], box['y1'], box['x2'], box['y2'], box['h']
            
            xmin, xmax = min(x1,x2), max(x1,x2)
            ymin, ymax = min(y1,y2), max(y1,y2)
            
            # --- Vectorized Distance to Box ---
            # If inside, dx and dy are 0.
            # If outside, they represent dist to nearest edge.
            dx = np.maximum(0, np.maximum(xmin - x, x - xmax))
            dy = np.maximum(0, np.maximum(ymin - y, y - ymax))
            
            dist = np.sqrt(dx**2 + dy**2)
            
            # --- Field Function ---
            # h(d) = Target + (Slope * Distance)
            h_box = h_target + (GROWTH_RATE * dist)
            
            # --- Union of Fields ---
            # We want the mesh to be fine if *any* source demands it.
            # So we take the Minimum of the current value and this new constraint.
            h_values = np.minimum(h_values, h_box)
            
        return h_values