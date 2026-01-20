''' core/__init__.py
    ----------------
    The 'Core' Package: Data Structures & Input Handling
    
    This package serves as the foundation for the entire SnapMesh application.
    It contains the immutable definitions (Data Types) and the primary 
    Interface (Input Reader) that all other packages rely on.
    
    Modules Exported:
    -----------------
    1. Data Types (via .data_types):
       - NODE_DTYPE, FACE_DTYPE, CELL_DTYPE: Topology definitions.
       - CONSTRAINT_DTYPE: Geometry definitions.
       - FIELD_DTYPE: Sizing and Physics field definitions.
       
    2. Input/Output (via .input_reader):
       - input_reader: The main function to parse .inp files into NumPy arrays.
    
    Usage:
    ------
    >>> from core import input_reader, NODE_DTYPE
    >>> data = input_reader('geom.inp')
'''

# -----------------------------------------------------------------------------
# 1. Expose Data Structures
# -----------------------------------------------------------------------------
# We import these here so external scripts can access them as:
#   core.NODE_DTYPE
# instead of:
#   core.data_types.NODE_DTYPE
from .data_types import (
    NODE_DTYPE, 
    FACE_DTYPE, 
    CELL_DTYPE, 
    CONSTRAINT_DTYPE, 
    FIELD_DTYPE
)

# -----------------------------------------------------------------------------
# 2. Expose Input Tools
# -----------------------------------------------------------------------------
from .input_reader import input_reader

# -----------------------------------------------------------------------------
# 3. Package Metadata
# -----------------------------------------------------------------------------
__version__ = "0.2.0"