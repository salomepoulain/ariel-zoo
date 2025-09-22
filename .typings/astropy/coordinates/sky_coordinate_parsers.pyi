from .baseframe import BaseCoordinateFrame as BaseCoordinateFrame, _get_diff_cls as _get_diff_cls, _get_repr_cls as _get_repr_cls, frame_transform_graph as frame_transform_graph
from .representation import BaseRepresentation as BaseRepresentation, SphericalRepresentation as SphericalRepresentation, UnitSphericalRepresentation as UnitSphericalRepresentation
from _typeshed import Incomplete
from astropy.units import IrreducibleUnit as IrreducibleUnit, Unit as Unit
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED

PLUS_MINUS_RE: Incomplete
J_PREFIXED_RA_DEC_RE: Incomplete

def _get_frame_class(frame):
    """
    Get a frame class from the input `frame`, which could be a frame name
    string, or frame class.
    """

_conflict_err_msg: str

def _get_frame_without_data(args, kwargs):
    """
    Determines the coordinate frame from input SkyCoord args and kwargs.

    This function extracts (removes) all frame attributes from the kwargs and
    determines the frame class either using the kwargs, or using the first
    element in the args (if a single frame object is passed in, for example).
    This function allows a frame to be specified as a string like 'icrs' or a
    frame class like ICRS, or an instance ICRS(), as long as the instance frame
    attributes don't conflict with kwargs passed in (which could require a
    three-way merge with the coordinate data possibly specified via the args).
    """
def _parse_coordinate_data(frame, args, kwargs):
    """
    Extract coordinate data from the args and kwargs passed to SkyCoord.

    By this point, we assume that all of the frame attributes have been
    extracted from kwargs (see _get_frame_without_data()), so all that are left
    are (1) extra SkyCoord attributes, and (2) the coordinate data, specified in
    any of the valid ways.
    """
def _get_representation_component_units(args, kwargs):
    """
    Get the unit from kwargs for the *representation* components (not the
    differentials).
    """
def _parse_coordinate_arg(coords, frame, units): ...
def _get_representation_attrs(frame, units, kwargs):
    '''
    Find instances of the "representation attributes" for specifying data
    for this frame.  Pop them off of kwargs, run through the appropriate class
    constructor (to validate and apply unit), and put into the output
    valid_kwargs.  "Representation attributes" are the frame-specific aliases
    for the underlying data values in the representation, e.g. "ra" for "lon"
    for many equatorial spherical representations, or "w" for "x" in the
    cartesian representation of Galactic.

    This also gets any *differential* kwargs, because they go into the same
    frame initializer later on.
    '''
def _parse_one_coord_str(coord_str: str, *, is_radec: bool = True) -> tuple[str, str]:
    """Parse longitude-like and latitude-like values from a string.

    Currently the following formats are always supported:

     * space separated 2-value or 6-value format

    If the input can be assumed to represent an RA and Dec then the
    following are additionally supported:

     * space separated <6-value format, this requires a plus or minus sign
       separation between RA and Dec
     * sign separated format
     * JHHMMSS.ss+DDMMSS.ss format, with up to two optional decimal digits
     * JDDDMMSS.ss+DDMMSS.ss format, with up to two optional decimal digits

    Parameters
    ----------
    coord_str : str
        Coordinate string to parse.
    is_radec : bool, keyword-only
        Whether the coordinates represent an RA and Dec.

    Returns
    -------
    longitude-like, latitude-like : str
        Parsed coordinate values. If ``is_radec`` is `True` then they are
        RA and Dec.
    """
