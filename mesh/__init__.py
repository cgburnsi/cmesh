''' mesh/__init__.py '''
# Geometry tools
from .geometry import compute_face_metrics, compute_triangle_quality

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
from .smoothing import spring_smoother, distmesh_smoother
from .containment import check_points_inside
from .sizing import SizingField

from .generator import MeshGenerator # Add this line
from .connectivity import build_fvm_connectivity


__version__ = "0.2.0"