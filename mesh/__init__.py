''' mesh/__init__.py
    ----------------------
    Mesh Generation Utilities
'''

''' mesh/__init__.py '''
from .geometry import compute_face_metrics
from .distance import project_points_to_boundary
from .initialization import generate_initial_points
from .triangulation import triangulate_and_filter # <--- NEW


__version__ = "0.1.0"