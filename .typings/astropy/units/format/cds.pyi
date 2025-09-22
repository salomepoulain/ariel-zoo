import numpy as np
from .fits import FITS as FITS
from .generic import Generic as Generic
from astropy.extern.ply.lex import Lexer as Lexer
from astropy.units import UnitBase as UnitBase
from astropy.units.utils import is_effectively_unity as is_effectively_unity
from astropy.utils import classproperty as classproperty, parsing as parsing
from astropy.utils.misc import did_you_mean as did_you_mean
from astropy.utils.parsing import ThreadSafeParser as ThreadSafeParser
from typing import ClassVar, Literal

class CDS(FITS):
    """
    Support the `Centre de Données astronomiques de Strasbourg
    <https://cds.unistra.fr/>`_ `Standards for Astronomical
    Catalogues 2.0 <https://vizier.unistra.fr/vizier/doc/catstd-3.2.htx>`_
    format, and the `complete set of supported units
    <https://vizier.unistra.fr/viz-bin/Unit>`_.  This format is used
    by VOTable up to version 1.2.
    """
    _space: ClassVar[str]
    _times: ClassVar[str]
    _scale_unit_separator: ClassVar[str]
    _tokens: ClassVar[tuple[str, ...]]
    def _units(cls) -> dict[str, UnitBase]: ...
    def _lexer(cls) -> Lexer: ...
    def _parser(cls) -> ThreadSafeParser:
        """
        The grammar here is based on the description in the `Standards
        for Astronomical Catalogues 2.0
        <https://vizier.unistra.fr/vizier/doc/catstd-3.2.htx>`_, which is not
        terribly precise.  The exact grammar is here is based on the
        YACC grammar in the `unity library <https://purl.org/nxg/dist/unity/>`_.
        """
    @classmethod
    def _parse_unit(cls, unit: str, detailed_exception: bool = True) -> UnitBase: ...
    @classmethod
    def parse(cls, s: str, debug: bool = False) -> UnitBase: ...
    @classmethod
    def _format_mantissa(cls, m: str) -> str: ...
    @classmethod
    def _format_superscript(cls, number: str) -> str: ...
    @classmethod
    def format_exponential_notation(cls, val: float | np.number, format_spec: str = '.8g') -> str: ...
    @classmethod
    def to_string(cls, unit: UnitBase, fraction: bool | Literal['inline'] = False) -> str: ...
