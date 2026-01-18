''' core/data_types.py
    ------------------
    Central definition of memory layout
'''
import numpy as np


# NODE: ID, X, Y
NODE_DTYPE = np.dtype([('id', 'i4'), ('x',  'f8'), ('y',  'f8')])
EDGE_DTYPE = np.dtype([('id', 'i4'), ('n1',  'i4'), ('n2',  'i4')])
CELL_DTYPE = np.dtype([('id', 'i4'), ('e1',  'i4'), ('e2',  'i4'), ('e3',  'i4')])





if __name__ == '__main__':
    pass


