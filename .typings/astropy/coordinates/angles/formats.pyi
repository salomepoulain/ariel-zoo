from .errors import IllegalHourError as IllegalHourError, IllegalHourWarning as IllegalHourWarning, IllegalMinuteError as IllegalMinuteError, IllegalMinuteWarning as IllegalMinuteWarning, IllegalSecondError as IllegalSecondError, IllegalSecondWarning as IllegalSecondWarning
from _typeshed import Incomplete
from astropy.utils import parsing as parsing

class _AngleParser:
    """
    Parses the various angle formats including:

       * 01:02:30.43 degrees
       * 1 2 0 hours
       * 1°2′3″
       * 1d2m3s
       * -1h2m3s
       * 1°2′3″N

    This class should not be used directly.  Use `parse_angle`
    instead.
    """
    _thread_local: Incomplete
    def __init__(self) -> None: ...
    @classmethod
    def _get_simple_unit_names(cls): ...
    @classmethod
    def _make_parser(cls): ...
    def parse(self, angle, unit, debug: bool = False): ...

def _check_hour_range(hrs: float) -> None:
    """
    Checks that the given value is in the range [-24,24].  If the value
    is equal to -24 or 24, then a warning is raised.
    """
def _check_minute_range(m: float) -> None:
    """
    Checks that the given value is in the range [0,60].  If the value
    is equal to 60, then a warning is raised.
    """
def _check_second_range(sec: float) -> None:
    """
    Checks that the given value is in the range [0,60].  If the value
    is equal to 60, then a warning is raised.
    """
def parse_angle(angle, unit: Incomplete | None = None, debug: bool = False):
    """
    Parses an input string value into an angle value.

    Parameters
    ----------
    angle : str
        A string representing the angle.  May be in one of the following forms:

            * 01:02:30.43 degrees
            * 1 2 0 hours
            * 1°2′3″
            * 1d2m3s
            * -1h2m3s

    unit : `~astropy.units.UnitBase` instance, optional
        The unit used to interpret the string.  If ``unit`` is not
        provided, the unit must be explicitly represented in the
        string, either at the end or as number separators.

    debug : bool, optional
        If `True`, print debugging information from the parser.

    Returns
    -------
    value, unit : tuple
        ``value`` is the value as a floating point number or three-part
        tuple, and ``unit`` is a `Unit` instance which is either the
        unit passed in or the one explicitly mentioned in the input
        string.
    """
def _decimal_to_sexagesimal(a, /):
    """
    Convert a floating-point input to a 3 tuple
    - if input is in degrees, the result is (degree, arcminute, arcsecond)
    - if input is in hourangle, the result is (hour, minute, second)
    """
def _decimal_to_sexagesimal_string(angle, precision: Incomplete | None = None, pad: bool = False, sep=(':',), fields: int = 3):
    """
    Given a floating point angle, convert it to string
    """
