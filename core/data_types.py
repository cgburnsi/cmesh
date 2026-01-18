''' core/data_types.py
    ------------------
    Central definition of memory layout
'''
import numpy as np

NODE_DTYPE = np.dtype([('id', 'i4'), ('x',  'f8'), ('y',  'f8')])
FACE_DTYPE = np.dtype([('id', 'i4'), ('n1',  'i4'), ('n2',  'i4'), ('tag', 'i4')])
CELL_DTYPE = np.dtype([('id', 'i4'), ('f1',  'i4'), ('f2',  'i4'), ('f3',  'i4')])

# Example: ID, Type (1=Line, 2=Arc, 3=Circle), Param1, Param2, Param3
CONSTRAINT_DTYPE = np.dtype([
    ('id', 'i4'),
    ('type', 'i4'),     # e.g., 0=FixedNode, 1=LinearBoundary, 2=ArcBoundary
    ('target', 'i4'),   # The ID of the Node or Edge being constrained
    ('p1', 'f8'),       # Generic parameter 1 (e.g., Radius, or X center)
    ('p2', 'f8'),       # Generic parameter 2 (e.g., Y center)
    ('p3', 'f8')        # Generic parameter 3
])





if __name__ == '__main__':
    pass


