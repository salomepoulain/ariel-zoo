from .funcs import get_sun as get_sun
from _typeshed import Incomplete
from astropy.utils.console import _color_text as _color_text, color_print as color_print

__all__: Incomplete

class HumanError(ValueError): ...
class CelestialError(ValueError): ...

def get_sign(dt):
    """ """

_VALID_SIGNS: Incomplete
_CONST_TO_SIGNS: Incomplete
_ZODIAC: Incomplete

def _get_zodiac(yr): ...
def horoscope(birthday, corrected: bool = True, chinese: bool = False) -> None:
    '''
    Enter your birthday as an `astropy.time.Time` object and
    receive a mystical horoscope about things to come.

    Parameters
    ----------
    birthday : `astropy.time.Time` or str
        Your birthday as a `datetime.datetime` or `astropy.time.Time` object
        or "YYYY-MM-DD"string.
    corrected : bool
        Whether to account for the precession of the Earth instead of using the
        ancient Greek dates for the signs.  After all, you do want your *real*
        horoscope, not a cheap inaccurate approximation, right?

    chinese : bool
        Chinese annual zodiac wisdom instead of Western one.

    Returns
    -------
    Infinite wisdom, condensed into astrologically precise prose.

    Notes
    -----
    This function was implemented on April 1.  Take note of that date.
    '''
def inject_horoscope() -> None: ...
