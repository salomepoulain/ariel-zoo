from _typeshed import Incomplete
from astropy.utils.exceptions import AstropyWarning

__all__ = ['RangeError', 'BoundsError', 'IllegalHourError', 'IllegalMinuteError', 'IllegalSecondError', 'IllegalHourWarning', 'IllegalMinuteWarning', 'IllegalSecondWarning']

class RangeError(ValueError):
    """
    Raised when some part of an angle is out of its valid range.
    """
class BoundsError(RangeError):
    """
    Raised when an angle is outside of its user-specified bounds.
    """

class IllegalHourError(RangeError):
    """
    Raised when an hour value is not in the range [0,24).

    Parameters
    ----------
    hour : int, float

    Examples
    --------
    .. code-block:: python

        if not 0 <= hr < 24:
           raise IllegalHourError(hour)
    """
    hour: Incomplete
    def __init__(self, hour) -> None: ...
    def __str__(self) -> str: ...

class IllegalHourWarning(AstropyWarning):
    """
    Raised when an hour value is 24.

    Parameters
    ----------
    hour : int, float
    """
    hour: Incomplete
    alternativeactionstr: Incomplete
    def __init__(self, hour, alternativeactionstr: Incomplete | None = None) -> None: ...
    def __str__(self) -> str: ...

class IllegalMinuteError(RangeError):
    """
    Raised when an minute value is not in the range [0,60].

    Parameters
    ----------
    minute : int, float

    Examples
    --------
    .. code-block:: python

        if not 0 <= min < 60:
            raise IllegalMinuteError(minute)

    """
    minute: Incomplete
    def __init__(self, minute) -> None: ...
    def __str__(self) -> str: ...

class IllegalMinuteWarning(AstropyWarning):
    """
    Raised when a minute value is 60.

    Parameters
    ----------
    minute : int, float
    """
    minute: Incomplete
    alternativeactionstr: Incomplete
    def __init__(self, minute, alternativeactionstr: Incomplete | None = None) -> None: ...
    def __str__(self) -> str: ...

class IllegalSecondError(RangeError):
    """
    Raised when an second value (time) is not in the range [0,60].

    Parameters
    ----------
    second : int, float

    Examples
    --------
    .. code-block:: python

        if not 0 <= sec < 60:
            raise IllegalSecondError(second)
    """
    second: Incomplete
    def __init__(self, second) -> None: ...
    def __str__(self) -> str: ...

class IllegalSecondWarning(AstropyWarning):
    """
    Raised when a second value is 60.

    Parameters
    ----------
    second : int, float
    """
    second: Incomplete
    alternativeactionstr: Incomplete
    def __init__(self, second, alternativeactionstr: Incomplete | None = None) -> None: ...
    def __str__(self) -> str: ...
