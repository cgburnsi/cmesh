''' mesh/sizing.py '''
import numpy as np

class SizingField:
    def __init__(self, sources_array):
        self.sources = sources_array
        
        # Find global background h (Type 0)
        globals = self.sources[self.sources['type'] == 0]
        if len(globals) > 0:
            self.h_background = globals[0]['h']
        else:
            self.h_background = 1.0 # Fallback

    def __call__(self, points):
        """
        Evaluates h at a list of points (N, 2).
        Returns (N,) array of sizes.
        """
        n_points = len(points)
        h_values = np.full(n_points, self.h_background)
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Check Box Sources (Type 1)
        boxes = self.sources[self.sources['type'] == 1]
        for box in boxes:
            x1, y1, x2, y2, h_target = box['x1'], box['y1'], box['x2'], box['y2'], box['h']
            
            # Create mask for points inside this box
            # Allow for box defined in any order
            xmin, xmax = min(x1,x2), max(x1,x2)
            ymin, ymax = min(y1,y2), max(y1,y2)
            
            mask_x = (x >= xmin) & (x <= xmax)
            mask_y = (y >= ymin) & (y <= ymax)
            mask = mask_x & mask_y
            
            # Apply the source size. Take MINIMUM if multiple sources overlap.
            h_values[mask] = np.minimum(h_values[mask], h_target)
            
        return h_values