from .fk5 import FK5 as FK5
from .icrs import ICRS as ICRS
from .utils import EQUINOX_J2000 as EQUINOX_J2000
from _typeshed import Incomplete
from astropy.coordinates.baseframe import frame_transform_graph as frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose as matrix_transpose, rotation_matrix as rotation_matrix
from astropy.coordinates.transformations import DynamicMatrixTransform as DynamicMatrixTransform

def _icrs_to_fk5_matrix():
    """
    B-matrix from USNO circular 179.  Used by the ICRS->FK5 transformation
    functions.
    """

_ICRS_TO_FK5_J2000_MAT: Incomplete

def icrs_to_fk5(icrscoord, fk5frame): ...
def fk5_to_icrs(fk5coord, icrsframe): ...
