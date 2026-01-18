import numpy as np



from core import NODE_DTYPE, EDGE_DTYPE, CELL_DTYPE
from core import input_reader


    
if __name__ == '__main__':

    data = input_reader('geom1.inp')

    
    # Demonstration of the new power of structured arrays:
    print("--- Structured Data ---")
    print("Nodes Array:\n", data["nodes"])
    
    print("\n--- Easy Access ---")
    # Now you can access columns by name!
    print("All X coordinates:", data["nodes"]['x'])
    print("Edge Start Nodes:", data["edges"]['n1'])
    print("Cell ID 1 Edges:", data["cells"][0]) # Returns (1, 1, 2, 3)




