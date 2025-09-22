from . import core as core, utils as utils
from .fits import FITS as FITS
from astropy.extern.ply.lex import Lexer as Lexer
from astropy.units import UnitBase as UnitBase
from astropy.units.errors import UnitParserWarning as UnitParserWarning, UnitsWarning as UnitsWarning
from astropy.utils import classproperty as classproperty, parsing as parsing
from astropy.utils.parsing import ThreadSafeParser as ThreadSafeParser
from typing import ClassVar, Literal

class OGIP(FITS):
    """
    Support the units in `Office of Guest Investigator Programs (OGIP)
    FITS files
    <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_93_001/>`__.
    """
    _tokens: ClassVar[tuple[str, ...]]
    _deprecated_units: ClassVar[frozenset[str]]
    def _units(cls) -> dict[str, UnitBase]: ...
    def _lexer(cls) -> Lexer: ...
    def _parser(cls) -> ThreadSafeParser:
        """
        The grammar here is based on the description in the
        `Specification of Physical Units within OGIP FITS files
        <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_93_001/>`__,
        which is not terribly precise.  The exact grammar is here is
        based on the YACC grammar in the `unity library
        <https://bitbucket.org/nxg/unity/>`_.
        """
    @classmethod
    def parse(cls, s: str, debug: bool = False) -> UnitBase: ...
    @classmethod
    def _format_superscript(cls, number: str) -> str: ...
    @classmethod
    def to_string(cls, unit: UnitBase, fraction: bool | Literal['inline'] = 'inline') -> str: ...
