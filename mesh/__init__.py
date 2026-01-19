''' mesh/__init__.py '''
# Geometry tools
from .geometry import compute_face_metrics

# Initialization tools
from .initialization import (
    resample_boundary_points, 
    generate_inner_points
)

# Solver tools
from .distance import (
    project_points_to_boundary,
    project_points_to_specific_faces
)
from .smoothing import smooth_mesh
from .containment import check_points_inside
from .sizing import SizingField

__version__ = "0.2.0"