from . import base as base
from astropy.units import UnitBase as UnitBase
from astropy.utils import classproperty as classproperty
from collections.abc import Callable as Callable
from typing import ClassVar, Literal

class Console(base.Base):
    """
    Output-only format for to display pretty formatting at the
    console.

    For example::

      >>> import astropy.units as u
      >>> print(u.Ry.decompose().to_string('console'))  # doctest: +FLOAT_CMP
      2.1798721*10^-18 m^2 kg s^-2
      >>> print(u.Ry.decompose().to_string('console', fraction='multiline'))  # doctest: +FLOAT_CMP
                       m^2 kg
      2.1798721*10^-18 ------
                        s^2
      >>> print(u.Ry.decompose().to_string('console', fraction='inline'))  # doctest: +FLOAT_CMP
      2.1798721*10^-18 m^2 kg / s^2
    """
    _line: ClassVar[str]
    _space: ClassVar[str]
    def _fraction_formatters(cls) -> dict[bool | str, Callable[[str, str, str], str]]: ...
    @classmethod
    def _format_superscript(cls, number: str) -> str: ...
    @classmethod
    def _format_multiline_fraction(cls, scale: str, numerator: str, denominator: str) -> str: ...
    @classmethod
    def to_string(cls, unit: UnitBase, fraction: bool | Literal['inline', 'multiline'] = False) -> str: ...
