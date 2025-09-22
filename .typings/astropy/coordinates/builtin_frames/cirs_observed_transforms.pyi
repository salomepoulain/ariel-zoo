from .altaz import AltAz as AltAz
from .cirs import CIRS as CIRS
from .hadec import HADec as HADec
from .utils import PIOVER2 as PIOVER2
from astropy.coordinates.baseframe import frame_transform_graph as frame_transform_graph
from astropy.coordinates.erfa_astrom import erfa_astrom as erfa_astrom
from astropy.coordinates.representation import SphericalRepresentation as SphericalRepresentation, UnitSphericalRepresentation as UnitSphericalRepresentation
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference as FunctionTransformWithFiniteDifference
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED

def cirs_to_observed(cirs_coo, observed_frame): ...
def observed_to_cirs(observed_coo, cirs_frame): ...
