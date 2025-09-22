from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['BaseRADecFrame']

class BaseRADecFrame(BaseCoordinateFrame):
    '''
    A base class that defines default representation info for frames that
    represent longitude and latitude as Right Ascension and Declination
    following typical "equatorial" conventions.
    '''
    frame_specific_representation_info: Incomplete
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
