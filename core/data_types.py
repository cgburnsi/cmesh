''' core/data_types.py '''
import numpy as np

NODE_DTYPE = np.dtype([('id', 'i4'), ('x',  'f8'), ('y',  'f8')])

# Updated: Added 'segments' (int) to the end
FACE_DTYPE = np.dtype([('id', 'i4'), ('n1',  'i4'), ('n2',  'i4'), ('tag', 'i4'), ('segments', 'i4')])

CELL_DTYPE = np.dtype([('id', 'i4'), ('f1',  'i4'), ('f2',  'i4'), ('f3',  'i4')])

CONSTRAINT_DTYPE = np.dtype([
    ('id', 'i4'),
    ('type', 'i4'),
    ('target', 'i4'),
    ('p1', 'f8'), ('p2', 'f8'), ('p3', 'f8')
])

# NEW: Definition for Sources
SOURCE_DTYPE = np.dtype([
    ('id', 'i4'),
    ('type', 'i4'),     # 0=Global, 1=Box
    ('x1', 'f8'), ('y1', 'f8'),
    ('x2', 'f8'), ('y2', 'f8'),
    ('h', 'f8')         # Target Size
])