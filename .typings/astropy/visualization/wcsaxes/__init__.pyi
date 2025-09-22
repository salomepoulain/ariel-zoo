from .core import *
from .helpers import *
from .patches import *
from .coordinate_helpers import CoordinateHelper as CoordinateHelper
from .coordinates_map import CoordinatesMap as CoordinatesMap
from .wcsapi import custom_ucd_coord_meta_mapping as custom_ucd_coord_meta_mapping
from _typeshed import Incomplete
from astropy import config as _config

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.visualization.wcsaxes`.
    """
    coordinate_range_samples: Incomplete
    frame_boundary_samples: Incomplete
    grid_samples: Incomplete
    contour_grid_samples: Incomplete

conf: Incomplete
