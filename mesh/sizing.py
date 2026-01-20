''' mesh/sizing.py '''
import numpy as np

class SizingField:
    def __init__(self, fields_array):
        # We now accept 'fields' instead of 'sources'
        self.fields = fields_array
        
        # 1. Background Value (Type 0)
        # Note: We access ['v'] instead of ['h']
        globals = self.fields[self.fields['type'] == 0]
        if len(globals) > 0:
            self.h_background = globals[0]['v']  # <--- CHANGED
        else:
            self.h_background = 1.0

    def __call__(self, points):
        n_points = len(points)
        h_values = np.full(n_points, self.h_background)
        
        x = points[:, 0]
        y = points[:, 1]
        
        GROWTH_RATE = 0.05
        
        # 2. Apply Box Fields (Type 1)
        boxes = self.fields[self.fields['type'] == 1]
        for box in boxes:
            # Note: We unpack 'v' instead of 'h'
            x1, y1, x2, y2, h_target = box['x1'], box['y1'], box['x2'], box['y2'], box['v'] # <--- CHANGED
            
            xmin, xmax = min(x1,x2), max(x1,x2)
            ymin, ymax = min(y1,y2), max(y1,y2)
            
            dx = np.maximum(0, np.maximum(xmin - x, x - xmax))
            dy = np.maximum(0, np.maximum(ymin - y, y - ymax))
            dist = np.sqrt(dx**2 + dy**2)
            
            h_box = h_target + (GROWTH_RATE * dist)
            h_values = np.minimum(h_values, h_box)
            
        return h_values