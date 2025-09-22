import numpy as np
from .core import dimensionless_unscaled
from .typing import QuantityLike
from _typeshed import Incomplete
from astropy import config as _config
from astropy.utils.data_info import ParentDtypeInfo
from collections.abc import Generator
from typing import Self

__all__ = ['Quantity', 'SpecificTypeQuantity', 'QuantityInfoBase', 'QuantityInfo', 'allclose', 'isclose']

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for Quantity.
    """
    latex_array_threshold: Incomplete

class QuantityIterator:
    """
    Flat iterator object to iterate over Quantities.

    A `QuantityIterator` iterator is returned by ``q.flat`` for any Quantity
    ``q``.  It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    Quantity.flatten : Returns a flattened copy of an array.

    Notes
    -----
    `QuantityIterator` is inspired by `~numpy.ma.core.MaskedIterator`.  It
    is not exported by the `~astropy.units` module.  Instead of
    instantiating a `QuantityIterator` directly, use `Quantity.flat`.
    """
    _quantity: Incomplete
    _dataiter: Incomplete
    def __init__(self, q) -> None: ...
    def __iter__(self): ...
    def __getitem__(self, indx): ...
    def __setitem__(self, index, value) -> None: ...
    def __next__(self): ...
    next = __next__
    def __len__(self) -> int: ...
    @property
    def base(self):
        """A reference to the array that is iterated over."""
    @property
    def coords(self):
        """An N-dimensional tuple of current coordinates."""
    @property
    def index(self):
        """Current flat index into the array."""
    def copy(self):
        """Get a copy of the iterator as a 1-D array."""

class QuantityInfoBase(ParentDtypeInfo):
    attrs_from_parent: Incomplete
    _supports_indexing: bool
    @staticmethod
    def default_format(val): ...
    @staticmethod
    def possible_string_format_functions(format_) -> Generator[Incomplete, None, Incomplete]:
        """Iterate through possible string-derived format functions.

        A string can either be a format specifier for the format built-in,
        a new-style format string, or an old-style format string.

        This method is overridden in order to suppress printing the unit
        in each row since it is already at the top in the column header.
        """

class QuantityInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_attrs: Incomplete
    _construct_from_dict_args: Incomplete
    _represent_as_dict_primary_data: str
    def new_like(self, cols, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new Quantity instance which is consistent with the
        input ``cols`` and has ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        cols : list
            List of input columns
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : `~astropy.units.Quantity` (or subclass)
            Empty instance of this class consistent with ``cols``

        """
    def get_sortable_arrays(self):
        """
        Return a list of arrays which can be lexically sorted to represent
        the order of the parent column.

        For Quantity this is just the quantity itself.


        Returns
        -------
        arrays : list of ndarray
        """

class Quantity(np.ndarray):
    """A `~astropy.units.Quantity` represents a number with some associated unit.

    See also: https://docs.astropy.org/en/stable/units/quantity.html

    Parameters
    ----------
    value : number, `~numpy.ndarray`, `~astropy.units.Quantity` (sequence), or str
        The numerical value of this quantity in the units given by unit.  If a
        `Quantity` or sequence of them (or any other valid object with a
        ``unit`` attribute), creates a new `Quantity` object, converting to
        `unit` units as needed.  If a string, it is converted to a number or
        `Quantity`, depending on whether a unit is present.

    unit : unit-like
        An object that represents the unit associated with the input value.
        Must be an `~astropy.units.UnitBase` object or a string parseable by
        the :mod:`~astropy.units` package.

    dtype : ~numpy.dtype, optional
        The dtype of the resulting Numpy array or scalar that will
        hold the value.  If not provided, it is determined from the input,
        except that any integer and (non-Quantity) object inputs are converted
        to float by default.
        If `None`, the normal `numpy.dtype` introspection is used, e.g.
        preventing upcasting of integers.

    copy : bool, optional
        If `True` (default), then the value is copied.  Otherwise, a copy will
        only be made if ``__array__`` returns a copy, if value is a nested
        sequence, or if a copy is needed to satisfy an explicitly given
        ``dtype``.  (The `False` option is intended mostly for internal use,
        to speed up initialization where a copy is known to have been made.
        Use with care.)

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  As in `~numpy.array`.  This parameter
        is ignored if the input is a `Quantity` and ``copy=False``.

    subok : bool, optional
        If `False` (default), the returned array will be forced to be a
        `Quantity`.  Otherwise, `Quantity` subclasses will be passed through,
        or a subclass appropriate for the unit will be used (such as
        `~astropy.units.Dex` for ``u.dex(u.AA)``).

    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array
        should have.  Ones will be prepended to the shape as needed to meet
        this requirement.  This parameter is ignored if the input is a
        `Quantity` and ``copy=False``.

    Raises
    ------
    TypeError
        If the value provided is not a Python numeric type.
    TypeError
        If the unit provided is not either a :class:`~astropy.units.Unit`
        object or a parseable string unit.

    Notes
    -----
    Quantities can also be created by multiplying a number or array with a
    :class:`~astropy.units.Unit`. See https://docs.astropy.org/en/latest/units/

    Unless the ``dtype`` argument is explicitly specified, integer
    or (non-Quantity) object inputs are converted to `float` by default.
    """
    _equivalencies: Incomplete
    _default_unit = dimensionless_unscaled
    _unit: Incomplete
    __array_priority__: int
    def __class_getitem__(cls, unit_shape_dtype):
        '''Quantity Type Hints.

        Unit-aware type hints are ``Annotated`` objects that encode the class,
        the unit, and possibly shape and dtype information, depending on the
        python and :mod:`numpy` versions.

        Schematically, ``Annotated[cls[shape, dtype], unit]``

        As a classmethod, the type is the class, ie ``Quantity``
        produces an ``Annotated[Quantity, ...]`` while a subclass
        like :class:`~astropy.coordinates.Angle` returns
        ``Annotated[Angle, ...]``.

        Parameters
        ----------
        unit_shape_dtype : :class:`~astropy.units.UnitBase`, str, `~astropy.units.PhysicalType`, or tuple
            Unit specification, can be the physical type (ie str or class).
            If tuple, then the first element is the unit specification
            and all other elements are for `numpy.ndarray` type annotations.
            Whether they are included depends on the python and :mod:`numpy`
            versions.

        Returns
        -------
        `typing.Annotated`, `astropy.units.Unit`, or `astropy.units.PhysicalType`
            Return type in this preference order:
            * `typing.Annotated`
            * `astropy.units.Unit` or `astropy.units.PhysicalType`

        Raises
        ------
        TypeError
            If the unit/physical_type annotation is not Unit-like or
            PhysicalType-like.

        Examples
        --------
        Create a unit-aware Quantity type annotation

            >>> Quantity[Unit("s")]
            Annotated[Quantity, Unit("s")]

        See Also
        --------
        `~astropy.units.quantity_input`
            Use annotations for unit checks on function arguments and results.

        Notes
        -----
        |Quantity| types are also static-type compatible.
        '''
    def __new__(cls, value: QuantityLike, unit: Incomplete | None = None, dtype=..., copy: bool = True, order: Incomplete | None = None, subok: bool = False, ndmin: int = 0) -> Self: ...
    info: Incomplete
    def __array_finalize__(self, obj) -> None: ...
    def __array_wrap__(self, obj, context: Incomplete | None = None, return_scalar: bool = False): ...
    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        """Wrap numpy ufuncs, taking care of units.

        Parameters
        ----------
        function : callable
            ufunc to wrap.
        method : str
            Ufunc method: ``__call__``, ``at``, ``reduce``, etc.
        inputs : tuple
            Input arrays.
        kwargs : keyword arguments
            As passed on, with ``out`` containing possible quantity output.

        Returns
        -------
        result : `~astropy.units.Quantity` or `NotImplemented`
            Results of the ufunc, with the unit set properly.
        """
    def _result_as_quantity(self, result, unit, out):
        """Turn result into a quantity with the given unit.

        If no output is given, it will take a view of the array as a quantity,
        and set the unit.  If output is given, those should be quantity views
        of the result arrays, and the function will just set the unit.

        Parameters
        ----------
        result : ndarray or tuple thereof
            Array(s) which need to be turned into quantity.
        unit : `~astropy.units.Unit`
            Unit for the quantities to be returned (or `None` if the result
            should not be a quantity).  Should be tuple if result is a tuple.
        out : `~astropy.units.Quantity` or None
            Possible output quantity. Should be `None` or a tuple if result
            is a tuple.

        Returns
        -------
        out : `~astropy.units.Quantity`
           With units set.
        """
    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.

        Parameters
        ----------
        unit : UnitBase
            The unit for which the appropriate class should be returned

        Returns
        -------
        tuple :
            - `~astropy.units.Quantity` subclass
            - bool: True if subclasses of the given class are ok
        """
    def _new_view(self, obj: Incomplete | None = None, unit: Incomplete | None = None, propagate_info: bool = True):
        """Create a Quantity view of some array-like input, and set the unit.

        By default, return a view of ``obj`` of the same class as ``self`` and
        with the same unit.  Subclasses can override the type of class for a
        given unit using ``__quantity_subclass__``, and can ensure properties
        other than the unit are copied using ``__array_finalize__``.

        If the given unit defines a ``_quantity_class`` of which ``self``
        is not an instance, a view using this class is taken.

        Parameters
        ----------
        obj : ndarray or scalar, optional
            The array to create a view of.  If obj is a numpy or python scalar,
            it will be converted to an array scalar.  By default, ``self``
            is converted.

        unit : unit-like, optional
            The unit of the resulting object.  It is used to select a
            subclass, and explicitly assigned to the view if given.
            If not given, the subclass and unit will be that of ``self``.

        propagate_info : bool, optional
            Whether to transfer ``info`` if present.  Default: `True`, as
            appropriate for, e.g., unit conversions or slicing, where the
            nature of the object does not change.

        Returns
        -------
        view : `~astropy.units.Quantity` subclass

        """
    def _set_unit(self, unit) -> None:
        """Set the unit.

        This is used anywhere the unit is set or modified, i.e., in the
        initializer, in ``__imul__`` and ``__itruediv__`` for in-place
        multiplication and division by another unit, as well as in
        ``__array_finalize__`` for wrapping up views.  For Quantity, it just
        sets the unit, but subclasses can override it to check that, e.g.,
        a unit is consistent.
        """
    def __deepcopy__(self, memo): ...
    def __reduce__(self): ...
    def __setstate__(self, state) -> None: ...
    def _to_value(self, unit, equivalencies=[]):
        """Helper method for to and to_value."""
    def to(self, unit, equivalencies=[], copy: bool = True):
        """
        Return a new `~astropy.units.Quantity` object with the specified unit.

        Parameters
        ----------
        unit : unit-like
            An object that represents the unit to convert to. Must be
            an `~astropy.units.UnitBase` object or a string parseable
            by the `~astropy.units` package.

        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`astropy:unit_equivalencies`.
            If not provided or ``[]``, class default equivalencies will be used
            (none for `~astropy.units.Quantity`, but may be set for subclasses)
            If `None`, no equivalencies will be applied at all, not even any
            set globally or within a context.

        copy : bool, optional
            If `True` (default), then the value is copied.  Otherwise, a copy
            will only be made if necessary.

        See Also
        --------
        to_value : get the numerical value in a given unit.
        """
    def to_value(self, unit: Incomplete | None = None, equivalencies=[]):
        """
        The numerical value, possibly in a different unit.

        Parameters
        ----------
        unit : unit-like, optional
            The unit in which the value should be given. If not given or `None`,
            use the current unit.

        equivalencies : list of tuple, optional
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If not provided
            or ``[]``, class default equivalencies will be used (none for
            `~astropy.units.Quantity`, but may be set for subclasses).
            If `None`, no equivalencies will be applied at all, not even any
            set globally or within a context.

        Returns
        -------
        value : ndarray or scalar
            The value in the units specified. For arrays, this will be a view
            of the data if no unit conversion was necessary.

        See Also
        --------
        to : Get a new instance in a different unit.
        """
    value: Incomplete
    @property
    def unit(self):
        """
        A `~astropy.units.UnitBase` object representing the unit of this
        quantity.
        """
    @property
    def equivalencies(self):
        """
        A list of equivalencies that will be applied by default during
        unit conversions.
        """
    def _recursively_apply(self, func):
        """Apply function recursively to every field.

        Returns a copy with the result.
        """
    @property
    def si(self):
        """
        Returns a copy of the current `Quantity` instance with SI units. The
        value of the resulting object will be scaled.
        """
    @property
    def cgs(self):
        """
        Returns a copy of the current `Quantity` instance with CGS units. The
        value of the resulting object will be scaled.
        """
    @property
    def isscalar(self):
        """
        True if the `value` of this quantity is a scalar, or False if it
        is an array-like object.

        .. note::
            This is subtly different from `numpy.isscalar` in that
            `numpy.isscalar` returns False for a zero-dimensional array
            (e.g. ``np.array(1)``), while this is True for quantities,
            since quantities cannot represent true numpy scalars.
        """
    _include_easy_conversion_members: bool
    def __dir__(self):
        """
        Quantities are able to directly convert to other units that
        have the same physical type.  This function is implemented in
        order to make autocompletion still work correctly in IPython.
        """
    def __getattr__(self, attr):
        """
        Quantities are able to directly convert to other units that
        have the same physical type.
        """
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __lshift__(self, other): ...
    def __ilshift__(self, other): ...
    def __rlshift__(self, other): ...
    def __rrshift__(self, other): ...
    def __rshift__(self, other): ...
    def __irshift__(self, other): ...
    def __mul__(self, other): ...
    def __imul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __itruediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __pow__(self, other): ...
    def __hash__(self): ...
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def __setitem__(self, i, value) -> None: ...
    def __bool__(self) -> bool:
        """This method raises ValueError, since truthiness of quantities is ambiguous,
        especially for logarithmic units and temperatures. Use explicit comparisons.
        """
    def __len__(self) -> int: ...
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __round__(self, ndigits: int = 0): ...
    def __index__(self) -> int: ...
    @property
    def _unitstr(self): ...
    def to_string(self, unit: Incomplete | None = None, precision: Incomplete | None = None, format: Incomplete | None = None, subfmt: Incomplete | None = None, *, formatter: Incomplete | None = None):
        """
        Generate a string representation of the quantity and its unit.

        The behavior of this function can be altered via the
        `numpy.set_printoptions` function and its various keywords.  The
        exception to this is the ``threshold`` keyword, which is controlled via
        the ``[units.quantity]`` configuration item ``latex_array_threshold``.
        This is treated separately because the numpy default of 1000 is too big
        for most browsers to handle.

        Parameters
        ----------
        unit : unit-like, optional
            Specifies the unit.  If not provided,
            the unit used to initialize the quantity will be used.

        precision : number, optional
            The level of decimal precision. If `None`, or not provided,
            it will be determined from NumPy print options.

        format : str, optional
            The format of the result. If not provided, an unadorned
            string is returned. Supported values are:

            - 'latex': Return a LaTeX-formatted string

            - 'latex_inline': Return a LaTeX-formatted string that uses
              negative exponents instead of fractions

        formatter : str, callable, dict, optional
            The formatter to use for the value. If a string, it should be a
            valid format specifier using Python's mini-language. If a callable,
            it will be treated as the default formatter for all values and will
            overwrite default Latex formatting for exponential notation and complex
            numbers. If a dict, it should map a specific type to a callable to be
            directly passed into `numpy.array2string`. If not provided, the default
            formatter will be used.

        subfmt : str, optional
            Subformat of the result. For the moment, only used for
            ``format='latex'`` and ``format='latex_inline'``. Supported
            values are:

            - 'inline': Use ``$ ... $`` as delimiters.

            - 'display': Use ``$\\displaystyle ... $`` as delimiters.

        Returns
        -------
        str
            A string with the contents of this Quantity
        """
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _repr_latex_(self):
        """
        Generate a latex representation of the quantity and its unit.

        Returns
        -------
        lstr
            A LaTeX string with the contents of this Quantity
        """
    def __format__(self, format_spec) -> str: ...
    def decompose(self, bases=[]):
        """
        Generates a new `Quantity` with the units
        decomposed. Decomposed units have only irreducible units in
        them (see `astropy.units.UnitBase.decompose`).

        Parameters
        ----------
        bases : sequence of `~astropy.units.UnitBase`, optional
            The bases to decompose into.  When not provided,
            decomposes down to any irreducible units.  When provided,
            the decomposed result will only contain the given units.
            This will raises a `~astropy.units.UnitsError` if it's not possible
            to do so.

        Returns
        -------
        newq : `~astropy.units.Quantity`
            A new object equal to this quantity with units decomposed.
        """
    def _decompose(self, allowscaledunits: bool = False, bases=[]):
        """
        Generates a new `Quantity` with the units decomposed. Decomposed
        units have only irreducible units in them (see
        `astropy.units.UnitBase.decompose`).

        Parameters
        ----------
        allowscaledunits : bool
            If True, the resulting `Quantity` may have a scale factor
            associated with it.  If False, any scaling in the unit will
            be subsumed into the value of the resulting `Quantity`

        bases : sequence of UnitBase, optional
            The bases to decompose into.  When not provided,
            decomposes down to any irreducible units.  When provided,
            the decomposed result will only contain the given units.
            This will raises a `~astropy.units.UnitsError` if it's not possible
            to do so.

        Returns
        -------
        newq : `~astropy.units.Quantity`
            A new object equal to this quantity with units decomposed.

        """
    def item(self, *args):
        """Copy an element of an array to a scalar Quantity and return it.

        Like :meth:`~numpy.ndarray.item` except that it always
        returns a `Quantity`, not a Python scalar.

        """
    def tolist(self) -> None: ...
    def _to_own_unit(self, value, check_precision: bool = True, *, unit: Incomplete | None = None):
        """Convert value to one's own unit (or that given).

        Here, non-quantities are treated as dimensionless, and care is taken
        for values of 0, infinity or nan, which are allowed to have any unit.

        Parameters
        ----------
        value : anything convertible to `~astropy.units.Quantity`
            The value to be converted to the requested unit.
        check_precision : bool
            Whether to forbid conversion of float to integer if that changes
            the input number.  Default: `True`.
        unit : `~astropy.units.Unit` or None
            The unit to convert to.  By default, the unit of ``self``.

        Returns
        -------
        value : number or `~numpy.ndarray`
            In the requested units.

        """
    def itemset(self, *args) -> None: ...
    def tostring(self, order: str = 'C') -> None:
        """Not implemented, use ``.value.tostring()`` instead."""
    def tobytes(self, order: str = 'C') -> None:
        """Not implemented, use ``.value.tobytes()`` instead."""
    def tofile(self, fid, sep: str = '', format: str = '%s') -> None:
        """Not implemented, use ``.value.tofile()`` instead."""
    def dump(self, file) -> None:
        """Not implemented, use ``.value.dump()`` instead."""
    def dumps(self) -> None:
        """Not implemented, use ``.value.dumps()`` instead."""
    def fill(self, value) -> None: ...
    @property
    def flat(self):
        """A 1-D iterator over the Quantity array.

        This returns a ``QuantityIterator`` instance, which behaves the same
        as the `~numpy.flatiter` instance returned by `~numpy.ndarray.flat`,
        and is similar to, but not a subclass of, Python's built-in iterator
        object.
        """
    @flat.setter
    def flat(self, value) -> None: ...
    def take(self, indices, axis: Incomplete | None = None, out: Incomplete | None = None, mode: str = 'raise'): ...
    def put(self, indices, values, mode: str = 'raise') -> None: ...
    def choose(self, choices, out: Incomplete | None = None, mode: str = 'raise') -> None: ...
    def argsort(self, axis: int = -1, kind: Incomplete | None = None, order: Incomplete | None = None): ...
    def argsort(self, axis: int = -1, kind: Incomplete | None = None, order: Incomplete | None = None, *, stable: Incomplete | None = None): ...
    def searchsorted(self, v, *args, **kwargs): ...
    def argmax(self, axis: Incomplete | None = None, out: Incomplete | None = None, *, keepdims: bool = False): ...
    def argmin(self, axis: Incomplete | None = None, out: Incomplete | None = None, *, keepdims: bool = False): ...
    def __array_function__(self, function, types, args, kwargs):
        """Wrap numpy functions, taking care of units.

        Parameters
        ----------
        function : callable
            Numpy function to wrap
        types : iterable of classes
            Classes that provide an ``__array_function__`` override. Can
            in principle be used to interact with other classes. Below,
            mostly passed on to `~numpy.ndarray`, which can only interact
            with subclasses.
        args : tuple
            Positional arguments provided in the function call.
        kwargs : dict
            Keyword arguments provided in the function call.

        Returns
        -------
        result: `~astropy.units.Quantity`, `~numpy.ndarray`
            As appropriate for the function.  If the function is not
            supported, `NotImplemented` is returned, which will lead to
            a `TypeError` unless another argument overrode the function.

        Raises
        ------
        ~astropy.units.UnitsError
            If operands have incompatible units.
        """
    def _not_implemented_or_raise(self, function, types): ...
    def _wrap_function(self, function, *args, unit: Incomplete | None = None, out: Incomplete | None = None, **kwargs):
        """Wrap a numpy function that processes self, returning a Quantity.

        Parameters
        ----------
        function : callable
            Numpy function to wrap.
        args : positional arguments
            Any positional arguments to the function beyond the first argument
            (which will be set to ``self``).
        kwargs : keyword arguments
            Keyword arguments to the function.

        If present, the following arguments are treated specially:

        unit : `~astropy.units.Unit`
            Unit of the output result.  If not given, the unit of ``self``.
        out : `~astropy.units.Quantity`
            A Quantity instance in which to store the output.

        Notes
        -----
        Output should always be assigned via a keyword argument, otherwise
        no proper account of the unit is taken.

        Returns
        -------
        out : `~astropy.units.Quantity`
            Result of the function call, with the unit set properly.
        """
    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Incomplete | None = None, out: Incomplete | None = None): ...
    def var(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0, keepdims: bool = False, *, where: bool = True): ...
    def std(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0, keepdims: bool = False, *, where: bool = True): ...
    def mean(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False, *, where: bool = True): ...
    def round(self, decimals: int = 0, out: Incomplete | None = None): ...
    def dot(self, b, out: Incomplete | None = None): ...
    def all(self, axis: Incomplete | None = None, out: Incomplete | None = None) -> None: ...
    def any(self, axis: Incomplete | None = None, out: Incomplete | None = None) -> None: ...
    def diff(self, n: int = 1, axis: int = -1): ...
    def ediff1d(self, to_end: Incomplete | None = None, to_begin: Incomplete | None = None): ...
    def insert(self, obj, values, axis: Incomplete | None = None):
        """
        Insert values along the given axis before the given indices and return
        a new `~astropy.units.Quantity` object.

        This is a thin wrapper around the `numpy.insert` function.

        Parameters
        ----------
        obj : int, slice or sequence of int
            Object that defines the index or indices before which ``values`` is
            inserted.
        values : array-like
            Values to insert.  If the type of ``values`` is different
            from that of quantity, ``values`` is converted to the matching type.
            ``values`` should be shaped so that it can be broadcast appropriately
            The unit of ``values`` must be consistent with this quantity.
        axis : int, optional
            Axis along which to insert ``values``.  If ``axis`` is None then
            the quantity array is flattened before insertion.

        Returns
        -------
        out : `~astropy.units.Quantity`
            A copy of quantity with ``values`` inserted.  Note that the
            insertion does not occur in-place: a new quantity array is returned.

        Examples
        --------
        >>> import astropy.units as u
        >>> q = [1, 2] * u.m
        >>> q.insert(0, 50 * u.cm)
        <Quantity [ 0.5,  1.,  2.] m>

        >>> q = [[1, 2], [3, 4]] * u.m
        >>> q.insert(1, [10, 20] * u.m, axis=0)
        <Quantity [[  1.,  2.],
                   [ 10., 20.],
                   [  3.,  4.]] m>

        >>> q.insert(1, 10 * u.m, axis=1)
        <Quantity [[  1., 10.,  2.],
                   [  3., 10.,  4.]] m>

        """

class SpecificTypeQuantity(Quantity):
    """Superclass for Quantities of specific physical type.

    Subclasses of these work just like :class:`~astropy.units.Quantity`, except
    that they are for specific physical types (and may have methods that are
    only appropriate for that type).  Astropy examples are
    :class:`~astropy.coordinates.Angle` and
    :class:`~astropy.coordinates.Distance`

    At a minimum, subclasses should set ``_equivalent_unit`` to the unit
    associated with the physical type.
    """
    _equivalent_unit: Incomplete
    _unit: Incomplete
    _default_unit: Incomplete
    __array_priority__: Incomplete
    def __quantity_subclass__(self, unit): ...
    def _set_unit(self, unit) -> None: ...

def isclose(a, b, rtol: float = 1e-05, atol: Incomplete | None = None, equal_nan: bool = False):
    """
    Return a boolean array where two arrays are element-wise equal
    within a tolerance.

    Parameters
    ----------
    a, b : array-like or `~astropy.units.Quantity`
        Input values or arrays to compare
    rtol : array-like or `~astropy.units.Quantity`
        The relative tolerance for the comparison, which defaults to
        ``1e-5``.  If ``rtol`` is a :class:`~astropy.units.Quantity`,
        then it must be dimensionless.
    atol : number or `~astropy.units.Quantity`
        The absolute tolerance for the comparison.  The units (or lack
        thereof) of ``a``, ``b``, and ``atol`` must be consistent with
        each other.  If `None`, ``atol`` defaults to zero in the
        appropriate units.
    equal_nan : `bool`
        Whether to compare NaN’s as equal. If `True`, NaNs in ``a`` will
        be considered equal to NaN’s in ``b``.

    Notes
    -----
    This is a :class:`~astropy.units.Quantity`-aware version of
    :func:`numpy.isclose`. However, this differs from the `numpy` function in
    that the default for the absolute tolerance here is zero instead of
    ``atol=1e-8`` in `numpy`, as there is no natural way to set a default
    *absolute* tolerance given two inputs that may have differently scaled
    units.

    Raises
    ------
    `~astropy.units.UnitsError`
        If the dimensions of ``a``, ``b``, or ``atol`` are incompatible,
        or if ``rtol`` is not dimensionless.

    See Also
    --------
    allclose
    """
def allclose(a, b, rtol: float = 1e-05, atol: Incomplete | None = None, equal_nan: bool = False) -> bool:
    """
    Whether two arrays are element-wise equal within a tolerance.

    Parameters
    ----------
    a, b : array-like or `~astropy.units.Quantity`
        Input values or arrays to compare
    rtol : array-like or `~astropy.units.Quantity`
        The relative tolerance for the comparison, which defaults to
        ``1e-5``.  If ``rtol`` is a :class:`~astropy.units.Quantity`,
        then it must be dimensionless.
    atol : number or `~astropy.units.Quantity`
        The absolute tolerance for the comparison.  The units (or lack
        thereof) of ``a``, ``b``, and ``atol`` must be consistent with
        each other.  If `None`, ``atol`` defaults to zero in the
        appropriate units.
    equal_nan : `bool`
        Whether to compare NaN’s as equal. If `True`, NaNs in ``a`` will
        be considered equal to NaN’s in ``b``.

    Notes
    -----
    This is a :class:`~astropy.units.Quantity`-aware version of
    :func:`numpy.allclose`. However, this differs from the `numpy` function in
    that the default for the absolute tolerance here is zero instead of
    ``atol=1e-8`` in `numpy`, as there is no natural way to set a default
    *absolute* tolerance given two inputs that may have differently scaled
    units.

    Raises
    ------
    `~astropy.units.UnitsError`
        If the dimensions of ``a``, ``b``, or ``atol`` are incompatible,
        or if ``rtol`` is not dimensionless.

    See Also
    --------
    isclose
    """
