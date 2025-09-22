from _typeshed import Incomplete
from astropy.coordinates import SkyCoord as SkyCoord

RA_REGEX: str
DEC_REGEX: str
JCOORD_REGEX: Incomplete
JPARSER: Incomplete

def _sexagesimal(g): ...
def search(name, raise_: bool = False):
    """Regex match for coordinates in name."""
def to_ra_dec_angles(name):
    """get RA in hourangle and DEC in degrees by parsing name."""
def to_skycoord(name, frame: str = 'icrs'):
    """Convert to `name` to `SkyCoords` object."""
def shorten(name):
    """Produce a shortened version of the full object name.

    The shortened name is built from the prefix (usually the survey name) and RA (hour,
    minute), DEC (deg, arcmin) parts.
    e.g.: '2MASS J06495091-0737408' --> '2MASS J0649-0737'

    Parameters
    ----------
    name : str
        Full object name with J-coords embedded.

    Returns
    -------
    shortName: str
    """
