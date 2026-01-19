''' core/__init__.py '''
# Import data types so they are available as core.NODE_DTYPE, etc.
from .data_types import (
    NODE_DTYPE, 
    FACE_DTYPE, 
    CELL_DTYPE, 
    CONSTRAINT_DTYPE, 
    SOURCE_DTYPE
)

# Import the reader
from .input_reader import input_reader

__version__ = "0.2.0"