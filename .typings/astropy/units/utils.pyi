import numpy as np
from .core import UnitBase as UnitBase
from .errors import UnitScaleError as UnitScaleError
from .quantity import Quantity as Quantity
from .typing import Complex as Complex, Real as Real, UnitPower as UnitPower, UnitScale as UnitScale
from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from numpy.typing import NDArray as NDArray
from typing import Literal, SupportsFloat, TypeVar, overload

DType = TypeVar('DType', bound=np.generic)
FloatLike = TypeVar('FloatLike', bound=SupportsFloat)
_float_finfo: Incomplete
_JUST_BELOW_UNITY: Incomplete
_JUST_ABOVE_UNITY: Incomplete

def _get_first_sentence(s: str) -> str:
    """
    Get the first sentence from a string and remove any carriage
    returns.
    """
def _iter_unit_summary(namespace: dict[str, object]) -> Generator[tuple[UnitBase, str, str, str, Literal['Yes', 'No']], None, None]:
    """
    Generates the ``(unit, doc, represents, aliases, prefixes)``
    tuple used to format the unit summary docs in `generate_unit_summary`.
    """
def generate_unit_summary(namespace: dict[str, object]) -> str:
    """
    Generates a summary of units from a given namespace.  This is used
    to generate the docstring for the modules that define the actual
    units.

    Parameters
    ----------
    namespace : dict
        A namespace containing units.

    Returns
    -------
    docstring : str
        A docstring containing a summary table of the units.
    """
def generate_prefixonly_unit_summary(namespace: dict[str, object]) -> str:
    """
    Generates table entries for units in a namespace that are just prefixes
    without the base unit.  Note that this is intended to be used *after*
    `generate_unit_summary` and therefore does not include the table header.

    Parameters
    ----------
    namespace : dict
        A namespace containing units that are prefixes but do *not* have the
        base unit in their namespace.

    Returns
    -------
    docstring : str
        A docstring containing a summary table of the units.
    """
def is_effectively_unity(value: Complex) -> bool: ...
def sanitize_scale_type(scale: Complex) -> UnitScale: ...
def sanitize_scale_value(scale: UnitScale) -> UnitScale: ...
def maybe_simple_fraction(p: Real, max_denominator: int = 100) -> UnitPower:
    """Fraction very close to x with denominator at most max_denominator.

    The fraction has to be such that fraction/x is unity to within 4 ulp.
    If such a fraction does not exist, returns the float number.

    The algorithm is that of `fractions.Fraction.limit_denominator`, but
    sped up by not creating a fraction to start with.

    If the input is zero, an integer or `fractions.Fraction`, just return it.
    """
def sanitize_power(p: Real) -> UnitPower:
    """Convert the power to a float, an integer, or a Fraction.

    If a fractional power can be represented exactly as a floating point
    number, convert it to a float, to make the math much faster; otherwise,
    retain it as a `fractions.Fraction` object to avoid losing precision.
    Conversely, if the value is indistinguishable from a rational number with a
    low-numbered denominator, convert to a Fraction object.
    If a power can be represented as an integer, use that.

    Parameters
    ----------
    p : float, int, Rational, Fraction
        Power to be converted.
    """
def resolve_fractions(a: Real, b: Real) -> tuple[Real, Real]:
    """
    If either input is a Fraction, convert the other to a Fraction
    (at least if it does not have a ridiculous denominator).
    This ensures that any operation involving a Fraction will use
    rational arithmetic and preserve precision.
    """
@overload
def quantity_asanyarray(a: Sequence[int]) -> NDArray[int]: ...
@overload
def quantity_asanyarray(a: Sequence[int], dtype: DType) -> NDArray[DType]: ...
@overload
def quantity_asanyarray(a: Sequence[Quantity]) -> Quantity: ...
