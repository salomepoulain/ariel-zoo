from . import core as core, generic as generic, utils as utils
from astropy.units import UnitBase as UnitBase
from astropy.units.errors import UnitScaleError as UnitScaleError
from astropy.utils import classproperty as classproperty
from typing import Literal

class FITS(generic.Generic):
    """
    The FITS standard unit format.

    This supports the format defined in the Units section of the `FITS
    Standard <https://fits.gsfc.nasa.gov/fits_standard.html>`_.
    """
    def _units(cls) -> dict[str, UnitBase]: ...
    @classmethod
    def _parse_unit(cls, unit: str, detailed_exception: bool = True) -> UnitBase: ...
    @classmethod
    def to_string(cls, unit: UnitBase, fraction: bool | Literal['inline'] = False) -> str: ...
    @classmethod
    def parse(cls, s: str, debug: bool = False) -> UnitBase: ...
