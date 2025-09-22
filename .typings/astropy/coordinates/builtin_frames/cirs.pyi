from .baseradec import BaseRADecFrame
from _typeshed import Incomplete

__all__ = ['CIRS']

class CIRS(BaseRADecFrame):
    """
    A coordinate or frame in the Celestial Intermediate Reference System (CIRS).

    The frame attributes are listed under **Other Parameters**.
    """
    obstime: Incomplete
    location: Incomplete
