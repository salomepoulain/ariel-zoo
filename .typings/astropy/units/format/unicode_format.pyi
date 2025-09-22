from . import console as console
from astropy.units import NamedUnit as NamedUnit
from astropy.units.typing import Real as Real
from typing import ClassVar

class Unicode(console.Console):
    """
    Output-only format to display pretty formatting at the console
    using Unicode characters.

    For example::

      >>> import astropy.units as u
      >>> print(u.bar.decompose().to_string('unicode'))
      100000 kg m⁻¹ s⁻²
      >>> print(u.bar.decompose().to_string('unicode', fraction='multiline'))
              kg
      100000 ────
             m s²
      >>> print(u.bar.decompose().to_string('unicode', fraction='inline'))
      100000 kg / (m s²)
    """
    _times: ClassVar[str]
    _line: ClassVar[str]
    @classmethod
    def _format_mantissa(cls, m: str) -> str: ...
    @classmethod
    def _format_unit_power(cls, unit: NamedUnit, power: Real = 1) -> str: ...
    @classmethod
    def _format_superscript(cls, number: str) -> str: ...
