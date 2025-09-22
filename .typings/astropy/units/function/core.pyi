from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from astropy.units import Quantity
from astropy.units.typing import UnitPower
from functools import cached_property

__all__ = ['FunctionUnitBase', 'FunctionQuantity']

class FunctionUnitBase(metaclass=ABCMeta):
    """Abstract base class for function units.

    Function units are functions containing a physical unit, such as dB(mW).
    Most of the arithmetic operations on function units are defined in this
    base class.

    While instantiation is defined, this class should not be used directly.
    Rather, subclasses should be used that override the abstract properties
    `_default_function_unit` and `_quantity_class`, and the abstract methods
    `from_physical`, and `to_physical`.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, the same as the function unit set by the subclass.
    """
    @property
    @abstractmethod
    def _default_function_unit(self):
        """Default function unit corresponding to the function.

        This property should be overridden by subclasses, with, e.g.,
        `~astropy.unit.MagUnit` returning `~astropy.unit.mag`.
        """
    @property
    @abstractmethod
    def _quantity_class(self):
        """Function quantity class corresponding to this function unit.

        This property should be overridden by subclasses, with, e.g.,
        `~astropy.unit.MagUnit` returning `~astropy.unit.Magnitude`.
        """
    @abstractmethod
    def from_physical(self, x):
        """Transformation from value in physical to value in function units.

        This method should be overridden by subclasses.  It is used to
        provide automatic transformations using an equivalency.
        """
    @abstractmethod
    def to_physical(self, x):
        """Transformation from value in function to value in physical units.

        This method should be overridden by subclasses.  It is used to
        provide automatic transformations using an equivalency.
        """
    __array_priority__: int
    _physical_unit: Incomplete
    _function_unit: Incomplete
    def __init__(self, physical_unit: Incomplete | None = None, function_unit: Incomplete | None = None) -> None: ...
    def _copy(self, physical_unit: Incomplete | None = None):
        """Copy oneself, possibly with a different physical unit."""
    @property
    def physical_unit(self): ...
    @property
    def function_unit(self): ...
    @property
    def equivalencies(self):
        """List of equivalencies between function and physical units.

        Uses the `from_physical` and `to_physical` methods.
        """
    def decompose(self, bases=...):
        """Copy the current unit with the physical unit decomposed.

        For details, see `~astropy.units.UnitBase.decompose`.
        """
    @property
    def si(self):
        """Copy the current function unit with the physical unit in SI."""
    @property
    def cgs(self):
        """Copy the current function unit with the physical unit in CGS."""
    @cached_property
    def _physical_type_id(self) -> tuple[tuple[str, UnitPower], ...]:
        """Get physical type corresponding to physical unit."""
    @property
    def physical_type(self):
        """Return the physical type of the physical unit (e.g., 'length')."""
    def is_equivalent(self, other, equivalencies=[]):
        """
        Returns `True` if this unit is equivalent to ``other``.

        Parameters
        ----------
        other : `~astropy.units.Unit`, string, or tuple
            The unit to convert to. If a tuple of units is specified, this
            method returns true if the unit matches any of those in the tuple.

        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`astropy:unit_equivalencies`.
            This list is in addition to the built-in equivalencies between the
            function unit and the physical one, as well as possible global
            defaults set by, e.g., `~astropy.units.set_enabled_equivalencies`.
            Use `None` to turn off any global equivalencies.

        Returns
        -------
        bool
        """
    def to(self, other, value: float = 1.0, equivalencies=[]):
        """
        Return the converted values in the specified unit.

        Parameters
        ----------
        other : `~astropy.units.Unit`, `~astropy.units.FunctionUnitBase`, or str
            The unit to convert to.

        value : int, float, or scalar array-like, optional
            Value(s) in the current unit to be converted to the specified unit.
            If not provided, defaults to 1.0.

        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`astropy:unit_equivalencies`.
            This list is in meant to treat only equivalencies between different
            physical units; the built-in equivalency between the function
            unit and the physical one is automatically taken into account.

        Returns
        -------
        values : scalar or array
            Converted value(s). Input value sequences are returned as
            numpy arrays.

        Raises
        ------
        `~astropy.units.UnitsError`
            If units are inconsistent.
        """
    def is_unity(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __rlshift__(self, other):
        """Unit conversion operator ``<<``."""
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __pow__(self, power): ...
    def __pos__(self): ...
    def to_string(self, format: str = 'generic', **kwargs):
        '''
        Output the unit in the given format as a string.

        The physical unit is appended, within parentheses, to the function
        unit, as in "dB(mW)", with both units set using the given format

        Parameters
        ----------
        format : `astropy.units.format.Base` subclass or str
            The name of a format or a formatter class.  If not
            provided, defaults to the generic format.
        '''
    def __format__(self, format_spec) -> str:
        """Try to format units using a formatter."""
    def __str__(self) -> str:
        """Return string representation for unit."""
    def __repr__(self) -> str: ...
    def _repr_latex_(self):
        """
        Generate latex representation of unit name.  This is used by
        the IPython notebook to print a unit with a nice layout.

        Returns
        -------
        Latex string
        """
    def __hash__(self): ...

class FunctionQuantity(Quantity):
    """A representation of a (scaled) function of a number with a unit.

    Function quantities are quantities whose units are functions containing a
    physical unit, such as dB(mW).  Most of the arithmetic operations on
    function quantities are defined in this base class.

    While instantiation is also defined here, this class should not be
    instantiated directly.  Rather, subclasses should be made which have
    ``_unit_class`` pointing back to the corresponding function unit class.

    Parameters
    ----------
    value : number, quantity-like, or sequence thereof
        The numerical value of the function quantity. If a number or
        a `~astropy.units.Quantity` with a function unit, it will be converted
        to ``unit`` and the physical unit will be inferred from ``unit``.
        If a `~astropy.units.Quantity` with just a physical unit, it will
        converted to the function unit, after, if necessary, converting it to
        the physical unit inferred from ``unit``.

    unit : str, `~astropy.units.UnitBase`, or `~astropy.units.FunctionUnitBase`, optional
        For an `~astropy.units.FunctionUnitBase` instance, the
        physical unit will be taken from it; for other input, it will be
        inferred from ``value``. By default, ``unit`` is set by the subclass.

    dtype : `~numpy.dtype`, optional
        The dtype of the resulting Numpy array or scalar that will
        hold the value.  If not provided, it is determined from the input,
        except that any input that cannot represent float (integer and bool)
        is converted to float.

    copy : bool, optional
        If `True` (default), then the value is copied.  Otherwise, a copy will
        only be made if ``__array__`` returns a copy, if value is a nested
        sequence, or if a copy is needed to satisfy an explicitly given
        ``dtype``.  (The `False` option is intended mostly for internal use,
        to speed up initialization where a copy is known to have been made.
        Use with care.)

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  As in `~numpy.array`.  Ignored
        if the input does not need to be converted and ``copy=False``.

    subok : bool, optional
        If `False` (default), the returned array will be forced to be of the
        class used.  Otherwise, subclasses will be passed through.

    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array
        should have.  Ones will be prepended to the shape as needed to meet
        this requirement.  This parameter is ignored if the input is a
        `~astropy.units.Quantity` and ``copy=False``.

    Raises
    ------
    TypeError
        If the value provided is not a Python numeric type.
    TypeError
        If the unit provided is not a `~astropy.units.FunctionUnitBase`
        or `~astropy.units.Unit` object, or a parseable string unit.
    """
    _unit_class: Incomplete
    __array_priority__: int
    _supported_ufuncs = SUPPORTED_UFUNCS
    _supported_functions = SUPPORTED_FUNCTIONS
    def __new__(cls, value, unit: Incomplete | None = None, dtype=..., copy: bool = True, order: Incomplete | None = None, subok: bool = False, ndmin: int = 0): ...
    @property
    def physical(self):
        """The physical quantity corresponding the function one."""
    @property
    def _function_view(self):
        """View as Quantity with function unit, dropping the physical unit.

        Use `~astropy.units.quantity.Quantity.value` for just the value.
        """
    @property
    def si(self):
        """Return a copy with the physical unit in SI units."""
    @property
    def cgs(self):
        """Return a copy with the physical unit in CGS units."""
    def decompose(self, bases=[]):
        """Generate a new instance with the physical unit decomposed.

        For details, see `~astropy.units.Quantity.decompose`.
        """
    def __quantity_subclass__(self, unit): ...
    _unit: Incomplete
    def _set_unit(self, unit) -> None: ...
    def __array_ufunc__(self, function, method, *inputs, **kwargs): ...
    def _maybe_new_view(self, result):
        """View as function quantity if the unit is unchanged.

        Used for the case that self.unit.physical_unit is dimensionless,
        where multiplication and division is done using the Quantity
        equivalent, to transform them back to a FunctionQuantity if possible.
        """
    def __mul__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def _comparison(self, other, comparison_func):
        """Do a comparison between self and other, raising UnitsError when
        other cannot be converted to self because it has different physical
        unit, and returning NotImplemented when there are other errors.
        """
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __lshift__(self, other):
        """Unit conversion operator `<<`."""
    def _wrap_function(self, function, *args, **kwargs): ...
    def max(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False): ...
    def min(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False): ...
    def sum(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False): ...
    def cumsum(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None): ...
    def clip(self, a_min, a_max, out: Incomplete | None = None): ...
