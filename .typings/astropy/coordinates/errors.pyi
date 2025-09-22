from _typeshed import Incomplete
from astropy.coordinates import BaseCoordinateFrame
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['ConvertError', 'NonRotationTransformationError', 'NonRotationTransformationWarning', 'UnknownSiteException']

class UnitsError(ValueError):
    """
    Raised if units are missing or invalid.
    """
class ConvertError(Exception):
    """
    Raised if a coordinate system cannot be converted to another.
    """

class NonRotationTransformationError(ValueError):
    """
    Raised for transformations that are not simple rotations. Such
    transformations can change the angular separation between coordinates
    depending on its direction.
    """
    frame_to: Incomplete
    frame_from: Incomplete
    def __init__(self, frame_to: BaseCoordinateFrame, frame_from: BaseCoordinateFrame) -> None: ...
    def __str__(self) -> str: ...

class UnknownSiteException(KeyError):
    site: Incomplete
    attribute: Incomplete
    close_names: Incomplete
    def __init__(self, site, attribute, close_names: Incomplete | None = None) -> None: ...
    def __str__(self) -> str: ...

class NonRotationTransformationWarning(AstropyUserWarning):
    """
    Emitted for transformations that are not simple rotations. Such
    transformations can change the angular separation between coordinates
    depending on its direction.
    """
    frame_to: Incomplete
    frame_from: Incomplete
    def __init__(self, frame_to: BaseCoordinateFrame, frame_from: BaseCoordinateFrame) -> None: ...
    def __str__(self) -> str: ...
