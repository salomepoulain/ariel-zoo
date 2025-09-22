import numpy as np
from . import core as core
from .base import Base as Base
from astropy.extern.ply.lex import LexToken as LexToken, Lexer as Lexer
from astropy.units import CompositeUnit as CompositeUnit, NamedUnit as NamedUnit, UnitBase as UnitBase
from astropy.units.errors import UnitsWarning as UnitsWarning
from astropy.utils import classproperty as classproperty, parsing as parsing
from astropy.utils.misc import did_you_mean as did_you_mean
from astropy.utils.parsing import ThreadSafeParser as ThreadSafeParser
from re import Match, Pattern
from typing import ClassVar, Final

class Generic(Base):
    '''
    A "generic" format.

    The syntax of the format is based directly on the FITS standard,
    but instead of only supporting the units that FITS knows about, it
    supports any unit available in the `astropy.units` namespace.
    '''
    _tokens: ClassVar[tuple[str, ...]]
    _deprecated_units: ClassVar[frozenset[str]]
    def _lexer(cls) -> Lexer: ...
    def _parser(cls) -> ThreadSafeParser:
        '''
        The grammar here is based on the description in the `FITS
        standard
        <http://fits.gsfc.nasa.gov/standard30/fits_standard30aa.pdf>`_,
        Section 4.3, which is not terribly precise.  The exact grammar
        is here is based on the YACC grammar in the `unity library
        <https://bitbucket.org/nxg/unity/>`_.

        This same grammar is used by the `"fits"` and `"vounit"`
        formats, the only difference being the set of available unit
        strings.
        '''
    @classmethod
    def _get_unit(cls, t: LexToken) -> UnitBase: ...
    @classmethod
    def _parse_unit(cls, s: str, detailed_exception: bool = True) -> UnitBase: ...
    _unit_symbols: ClassVar[dict[str, str]]
    _prefixable_unit_symbols: ClassVar[dict[str, str]]
    _unit_suffix_symbols: ClassVar[dict[str, str]]
    _translations: ClassVar[dict[int, str]]
    _superscripts: Final[str]
    _superscript_translations: ClassVar[dict[int, int]]
    _regex_superscript: ClassVar[Pattern[str]]
    @classmethod
    def _convert_superscript(cls, m: Match[str]) -> str: ...
    @classmethod
    def parse(cls, s: str, debug: bool = False) -> UnitBase: ...
    @classmethod
    def _do_parse(cls, s: str, debug: bool = False) -> UnitBase: ...
    @classmethod
    def _get_unit_name(cls, unit: NamedUnit) -> str: ...
    @classmethod
    def _validate_unit(cls, unit: str, detailed_exception: bool = True) -> None: ...
    @classmethod
    def _did_you_mean_units(cls, unit: str) -> str:
        """
        A wrapper around `astropy.utils.misc.did_you_mean` that deals with
        the display of deprecated units.

        Parameters
        ----------
        unit : str
            The invalid unit string

        Returns
        -------
        msg : str
            A message with alternatives, or the empty string.
        """
    @classmethod
    def _fix_deprecated(cls, x: str) -> list[str]: ...
    @classmethod
    def _try_decomposed(cls, unit: UnitBase) -> str | None: ...
    @classmethod
    def _decompose_to_known_units(cls, unit: CompositeUnit | NamedUnit) -> UnitBase:
        '''
        Partially decomposes a unit so it is only composed of units that
        are "known" to a given format.
        '''
    @classmethod
    def format_exponential_notation(cls, val: float | np.number, format_spec: str = 'g') -> str: ...
