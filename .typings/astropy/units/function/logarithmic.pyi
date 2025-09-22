from .core import FunctionQuantity, FunctionUnitBase
from _typeshed import Incomplete

__all__ = ['LogUnit', 'MagUnit', 'DexUnit', 'DecibelUnit', 'LogQuantity', 'Magnitude', 'Decibel', 'Dex']

class LogUnit(FunctionUnitBase):
    """Logarithmic unit containing a physical one.

    Usually, logarithmic units are instantiated via specific subclasses
    such `~astropy.units.MagUnit`, `~astropy.units.DecibelUnit`, and
    `~astropy.units.DexUnit`.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the logarithmic function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, the same as the logarithmic unit set by the subclass.

    """
    def _default_function_unit(self): ...
    @property
    def _quantity_class(self): ...
    def from_physical(self, x):
        """Transformation from value in physical to value in logarithmic units.
        Used in equivalency.
        """
    def to_physical(self, x):
        """Transformation from value in logarithmic to value in physical units.
        Used in equivalency.
        """
    def _add_and_adjust_physical_unit(self, other, sign_self, sign_other):
        """Add/subtract LogUnit to/from another unit, and adjust physical unit.

        self and other are multiplied by sign_self and sign_other, resp.

        We wish to do:   ±lu_1 + ±lu_2  -> lu_f          (lu=logarithmic unit)
                  and     pu_1^(±1) * pu_2^(±1) -> pu_f  (pu=physical unit)

        Raises
        ------
        UnitsError
            If function units are not equivalent.
        """
    def __neg__(self): ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...

class MagUnit(LogUnit):
    """Logarithmic physical units expressed in magnitudes.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the magnitude function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, this is ``mag``, but this allows one to use an equivalent
        unit such as ``2 mag``.
    """
    def _default_function_unit(self): ...
    @property
    def _quantity_class(self): ...

class DexUnit(LogUnit):
    """Logarithmic physical units expressed in magnitudes.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the magnitude function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, this is ``dex``, but this allows one to use an equivalent
        unit such as ``0.5 dex``.
    """
    def _default_function_unit(self): ...
    @property
    def _quantity_class(self): ...
    def to_string(self, format: str = 'generic'): ...

class DecibelUnit(LogUnit):
    """Logarithmic physical units expressed in dB.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the decibel function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, this is ``dB``, but this allows one to use an equivalent
        unit such as ``2 dB``.
    """
    def _default_function_unit(self): ...
    @property
    def _quantity_class(self): ...

class LogQuantity(FunctionQuantity):
    """A representation of a (scaled) logarithm of a number with a unit.

    Parameters
    ----------
    value : number, `~astropy.units.Quantity`, `~astropy.units.LogQuantity`, or sequence of quantity-like.
        The numerical value of the logarithmic quantity. If a number or
        a `~astropy.units.Quantity` with a logarithmic unit, it will be
        converted to ``unit`` and the physical unit will be inferred from
        ``unit``.  If a `~astropy.units.Quantity` with just a physical unit,
        it will converted to the logarithmic unit, after, if necessary,
        converting it to the physical unit inferred from ``unit``.

    unit : str, `~astropy.units.UnitBase`, or `~astropy.units.FunctionUnitBase`, optional
        For an `~astropy.units.FunctionUnitBase` instance, the
        physical unit will be taken from it; for other input, it will be
        inferred from ``value``. By default, ``unit`` is set by the subclass.

    dtype : `~numpy.dtype`, optional
        The ``dtype`` of the resulting Numpy array or scalar that will
        hold the value.  If not provided, is is determined automatically
        from the input value.

    copy : bool, optional
        If `True` (default), then the value is copied.  Otherwise, a copy will
        only be made if ``__array__`` returns a copy, if value is a nested
        sequence, or if a copy is needed to satisfy an explicitly given
        ``dtype``.  (The `False` option is intended mostly for internal use,
        to speed up initialization where a copy is known to have been made.
        Use with care.)

    Examples
    --------
    Typically, use is made of an `~astropy.units.FunctionQuantity`
    subclass, as in::

        >>> import astropy.units as u
        >>> u.Magnitude(-2.5)
        <Magnitude -2.5 mag>
        >>> u.Magnitude(10.*u.count/u.second)
        <Magnitude -2.5 mag(ct / s)>
        >>> u.Decibel(1.*u.W, u.DecibelUnit(u.mW))  # doctest: +FLOAT_CMP
        <Decibel 30. dB(mW)>

    """
    _unit_class = LogUnit
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __iadd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __isub__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __imul__(self, other): ...
    def __truediv__(self, other): ...
    def __itruediv__(self, other): ...
    def __pow__(self, other): ...
    def __ilshift__(self, other): ...
    def var(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0): ...
    def std(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0): ...
    def ptp(self, axis: Incomplete | None = None, out: Incomplete | None = None): ...
    def __array_function__(self, function, types, args, kwargs): ...
    def diff(self, n: int = 1, axis: int = -1): ...
    def ediff1d(self, to_end: Incomplete | None = None, to_begin: Incomplete | None = None): ...
    _supported_functions: Incomplete

class Dex(LogQuantity):
    _unit_class = DexUnit

class Decibel(LogQuantity):
    _unit_class = DecibelUnit

class Magnitude(LogQuantity):
    _unit_class = MagUnit
