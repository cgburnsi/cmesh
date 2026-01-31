''' mesh/sizing.py '''
import numpy as np

class SizingField:
    def __init__(self, fields_array, nodes=None, faces=None):
        self.fields = fields_array
        
        # 1. Background Value (Type 0)
        globals = self.fields[self.fields['type'] == 0]
        self.h_background = globals[0]['v'] if len(globals) > 0 else 1.0

    def __call__(self, points):
        # FIX: Ensure points are treated as a 2D array (N, 2) to avoid indexing errors
        pts = np.atleast_2d(points)
        n_points = len(pts)
        h_values = np.full(n_points, self.h_background)
        
        # 2. Apply Box Fields (Type 1)
        boxes = self.fields[self.fields['type'] == 1]
        for box in boxes:
            x1, y1, x2, y2, h_target = box['x1'], box['y1'], box['x2'], box['y2'], box['v']
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            
            # Calculate distance from the box using the guaranteed 2D 'pts'
            dx = np.maximum(0, np.maximum(xmin - pts[:, 0], pts[:, 0] - xmax))
            dy = np.maximum(0, np.maximum(ymin - pts[:, 1], pts[:, 1] - ymax))
            dist = np.sqrt(dx**2 + dy**2)
            
            # Smooth transition zone: increase target h as distance from box increases
            h_box = h_target + (0.15 * dist)
            h_values = np.minimum(h_values, h_box)
            
        # Return a scalar if a single point was passed, otherwise return the array
        return h_values[0] if np.ndim(points) == 1 else h_values