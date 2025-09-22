from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['Supergalactic']

class Supergalactic(BaseCoordinateFrame):
    """
    Supergalactic Coordinates
    (see Lahav et al. 2000, <https://ui.adsabs.harvard.edu/abs/2000MNRAS.312..166L>,
    and references therein).
    """
    frame_specific_representation_info: Incomplete
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
    _nsgp_gal: Incomplete
