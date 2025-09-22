from . import console as console
from astropy.units import NamedUnit as NamedUnit, UnitBase as UnitBase
from astropy.units.typing import Real as Real
from typing import ClassVar, Literal

class Latex(console.Console):
    """
    Output LaTeX to display the unit based on IAU style guidelines.

    Attempts to follow the `IAU Style Manual
    <https://www.iau.org/static/publications/stylemanual1989.pdf>`_.
    """
    _space: ClassVar[str]
    _scale_unit_separator: ClassVar[str]
    _times: ClassVar[str]
    @classmethod
    def _format_mantissa(cls, m: str) -> str: ...
    @classmethod
    def _format_superscript(cls, number: str) -> str: ...
    @classmethod
    def _format_unit_power(cls, unit: NamedUnit, power: Real = 1) -> str: ...
    @classmethod
    def _format_multiline_fraction(cls, scale: str, numerator: str, denominator: str) -> str: ...
    @classmethod
    def to_string(cls, unit: UnitBase, fraction: bool | Literal['inline', 'multiline'] = 'multiline') -> str: ...

class LatexInline(Latex):
    """
    Output LaTeX to display the unit based on IAU style guidelines with negative
    powers.

    Attempts to follow the `IAU Style Manual
    <https://www.iau.org/static/publications/stylemanual1989.pdf>`_ and the
    `ApJ and AJ style guide
    <https://journals.aas.org/manuscript-preparation/>`_.
    """
    name: ClassVar[str]
    @classmethod
    def to_string(cls, unit: UnitBase, fraction: bool | Literal['inline', 'multiline'] = False) -> str: ...
