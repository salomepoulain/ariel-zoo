import numpy as np
from astropy.units import NamedUnit as NamedUnit, UnitBase as UnitBase
from astropy.units.typing import Real as Real, UnitPower as UnitPower
from astropy.units.utils import maybe_simple_fraction as maybe_simple_fraction
from astropy.utils import classproperty as classproperty
from collections.abc import Callable as Callable, Iterable
from typing import ClassVar, Literal

class Base:
    """
    The abstract base class of all unit formats.
    """
    registry: ClassVar[dict[str, type[Base]]]
    _space: ClassVar[str]
    _scale_unit_separator: ClassVar[str]
    _times: ClassVar[str]
    name: ClassVar[str]
    def __new__(cls, *args, **kwargs): ...
    def __init_subclass__(cls, **kwargs) -> None: ...
    def _fraction_formatters(cls) -> dict[bool | str, Callable[[str, str, str], str]]: ...
    @classmethod
    def format_exponential_notation(cls, val: float | np.number, format_spec: str = '.8g') -> str:
        """
        Formats a value in exponential notation.

        Parameters
        ----------
        val : number
            The value to be formatted

        format_spec : str, optional
            Format used to split up mantissa and exponent

        Returns
        -------
        str
            The value in exponential notation in a this class's format.
        """
    @classmethod
    def _format_mantissa(cls, m: str) -> str: ...
    @classmethod
    def _format_superscript(cls, number: str) -> str: ...
    @classmethod
    def _format_unit_power(cls, unit: NamedUnit, power: UnitPower = 1) -> str:
        """Format the unit for this format class raised to the given power.

        This is overridden in Latex where the name of the unit can depend on the power
        (e.g., for degrees).
        """
    @classmethod
    def _format_power(cls, power: UnitPower) -> str: ...
    @classmethod
    def _format_unit_list(cls, units: Iterable[tuple[NamedUnit, Real]]) -> str: ...
    @classmethod
    def _format_inline_fraction(cls, scale: str, numerator: str, denominator: str) -> str: ...
    @classmethod
    def to_string(cls, unit: UnitBase, *, fraction: bool | Literal['inline'] = True) -> str:
        """Convert a unit to its string representation.

        Implementation for `~astropy.units.UnitBase.to_string`.

        Parameters
        ----------
        unit : |Unit|
            The unit to convert.
        fraction : {False|True|'inline'|'multiline'}, optional
            Options are as follows:

            - `False` : display unit bases with negative powers as they are
              (e.g., ``km s-1``);
            - 'inline' or `True` : use a single-line fraction (e.g., ``km / s``);
            - 'multiline' : use a multiline fraction (available for the
              ``latex``, ``console`` and ``unicode`` formats only; e.g.,
              ``$\\mathrm{\\frac{km}{s}}$``).

        Raises
        ------
        ValueError
            If ``fraction`` is not recognized.
        """
    @classmethod
    def _to_string(cls, unit: UnitBase, *, fraction: bool | str) -> str: ...
    @classmethod
    def parse(cls, s: str) -> UnitBase:
        """
        Convert a string to a unit object.
        """
