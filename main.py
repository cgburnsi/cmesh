import numpy as np

from core import NODE_DTYPE, FACE_DTYPE, CELL_DTYPE
from core import input_reader


    
if __name__ == '__main__':

    data = input_reader('geom1.inp')
    
    print("--- Geometry Loaded ---")
    print(f"Nodes: {len(data['nodes'])}")
    print(f"Faces: {len(data['faces'])}")
    print(f"Cells: {len(data['cells'])}")
    print(f"Constraints: {len(data['constraints'])}")
    
    print("\n--- Faces Data (ID, N1, N2, Tag) ---")
    print(data['faces'])




