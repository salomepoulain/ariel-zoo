from .fk4 import FK4NoETerms as FK4NoETerms
from .fk5 import FK5 as FK5
from .utils import EQUINOX_B1950 as EQUINOX_B1950, EQUINOX_J2000 as EQUINOX_J2000
from _typeshed import Incomplete
from astropy.coordinates.baseframe import frame_transform_graph as frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose as matrix_transpose
from astropy.coordinates.transformations import DynamicMatrixTransform as DynamicMatrixTransform

_B1950_TO_J2000_M: Incomplete
_FK4_CORR: Incomplete

def _fk4_B_matrix(obstime):
    """
    This is a correction term in the FK4 transformations because FK4 is a
    rotating system - see Murray 89 eqn 29.
    """
def fk4_no_e_to_fk5(fk4noecoord, fk5frame): ...
def fk5_to_fk4_no_e(fk5coord, fk4noeframe): ...
