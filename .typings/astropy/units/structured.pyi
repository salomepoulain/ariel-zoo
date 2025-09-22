import numpy as np
from _typeshed import Incomplete
from functools import cached_property

__all__ = ['StructuredUnit']

class StructuredUnit:
    '''Container for units for a structured Quantity.

    Parameters
    ----------
    units : unit-like, tuple of unit-like, or `~astropy.units.StructuredUnit`
        Tuples can be nested.  If a `~astropy.units.StructuredUnit` is passed
        in, it will be returned unchanged unless different names are requested.
    names : tuple of str, tuple or list; `~numpy.dtype`; or `~astropy.units.StructuredUnit`, optional
        Field names for the units, possibly nested. Can be inferred from a
        structured `~numpy.dtype` or another `~astropy.units.StructuredUnit`.
        For nested tuples, by default the name of the upper entry will be the
        concatenation of the names of the lower levels.  One can pass in a
        list with the upper-level name and a tuple of lower-level names to
        avoid this.  For tuples, not all levels have to be given; for any level
        not passed in, default field names of \'f0\', \'f1\', etc., will be used.

    Notes
    -----
    It is recommended to initialize the class indirectly, using
    `~astropy.units.Unit`.  E.g., ``u.Unit(\'AU,AU/day\')``.

    When combined with a structured array to produce a structured
    `~astropy.units.Quantity`, array field names will take precedence.
    Generally, passing in ``names`` is needed only if the unit is used
    unattached to a `~astropy.units.Quantity` and one needs to access its
    fields.

    Examples
    --------
    Various ways to initialize a `~astropy.units.StructuredUnit`::

        >>> import astropy.units as u
        >>> su = u.Unit(\'(AU,AU/day),yr\')
        >>> su
        Unit("((AU, AU / d), yr)")
        >>> su.field_names
        ([\'f0\', (\'f0\', \'f1\')], \'f1\')
        >>> su[\'f1\']
        Unit("yr")
        >>> su2 = u.StructuredUnit(((u.AU, u.AU/u.day), u.yr), names=((\'p\', \'v\'), \'t\'))
        >>> su2 == su
        True
        >>> su2.field_names
        ([\'pv\', (\'p\', \'v\')], \'t\')
        >>> su3 = u.StructuredUnit((su2[\'pv\'], u.day), names=([\'p_v\', (\'p\', \'v\')], \'t\'))
        >>> su3.field_names
        ([\'p_v\', (\'p\', \'v\')], \'t\')
        >>> su3.keys()
        (\'p_v\', \'t\')
        >>> su3.values()
        (Unit("(AU, AU / d)"), Unit("d"))

    Structured units share most methods with regular units::

        >>> su.physical_type
        astropy.units.structured.Structure((astropy.units.structured.Structure((PhysicalType(\'length\'), PhysicalType({\'speed\', \'velocity\'})), dtype=[(\'f0\', \'O\'), (\'f1\', \'O\')]), PhysicalType(\'time\')), dtype=[(\'f0\', \'O\'), (\'f1\', \'O\')])
        >>> su.si
        Unit("((1.49598e+11 m, 1.73146e+06 m / s), 3.15576e+07 s)")

    '''
    _units: Incomplete
    def __new__(cls, units, names: Incomplete | None = None): ...
    def __getnewargs__(self):
        """When de-serializing, e.g. pickle, start with a blank structure."""
    @property
    def field_names(self):
        """Possibly nested tuple of the field names of the parts."""
    def __len__(self) -> int: ...
    def __getitem__(self, item): ...
    def values(self): ...
    def keys(self): ...
    def items(self): ...
    def __iter__(self): ...
    def _recursively_apply(self, func, cls: Incomplete | None = None):
        """Apply func recursively.

        Parameters
        ----------
        func : callable
            Function to apply to all parts of the structured unit,
            recursing as needed.
        cls : type, optional
            If given, should be a subclass of `~numpy.void`. By default,
            will return a new `~astropy.units.StructuredUnit` instance.
        """
    def _recursively_get_dtype(self, value, enter_lists: bool = True):
        """Get structured dtype according to value, using our field names.

        This is useful since ``np.array(value)`` would treat tuples as lower
        levels of the array, rather than as elements of a structured array.
        The routine does presume that the type of the first tuple is
        representative of the rest.  Used in ``get_converter``.

        For the special value of ``UNITY``, all fields are assumed to be 1.0,
        and hence this will return an all-float dtype.

        """
    @property
    def si(self):
        """The `StructuredUnit` instance in SI units."""
    @property
    def cgs(self):
        """The `StructuredUnit` instance in cgs units."""
    @cached_property
    def _physical_type_id(self): ...
    @property
    def physical_type(self):
        """Physical types of all the fields."""
    def decompose(self, bases=...):
        """The `StructuredUnit` composed of only irreducible units.

        Parameters
        ----------
        bases : sequence of `~astropy.units.UnitBase`, optional
            The bases to decompose into.  When not provided,
            decomposes down to any irreducible units.  When provided,
            the decomposed result will only contain the given units.
            This will raises a `UnitsError` if it's not possible
            to do so.

        Returns
        -------
        `~astropy.units.StructuredUnit`
            With the unit for each field containing only irreducible units.
        """
    def is_equivalent(self, other, equivalencies=[]):
        """`True` if all fields are equivalent to the other's fields.

        Parameters
        ----------
        other : `~astropy.units.StructuredUnit`
            The structured unit to compare with, or what can initialize one.
        equivalencies : list of tuple, optional
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`unit_equivalencies`.
            The list will be applied to all fields.

        Returns
        -------
        bool
        """
    def get_converter(self, other, equivalencies=[]): ...
    def to(self, other, value=..., equivalencies=[]):
        """Return values converted to the specified unit.

        Parameters
        ----------
        other : `~astropy.units.StructuredUnit`
            The unit to convert to.  If necessary, will be converted to
            a `~astropy.units.StructuredUnit` using the dtype of ``value``.
        value : array-like, optional
            Value(s) in the current unit to be converted to the
            specified unit.  If a sequence, the first element must have
            entries of the correct type to represent all elements (i.e.,
            not have, e.g., a ``float`` where other elements have ``complex``).
            If not given, assumed to have 1. in all fields.
        equivalencies : list of tuple, optional
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`unit_equivalencies`.
            This list is in addition to possible global defaults set by, e.g.,
            `set_enabled_equivalencies`.
            Use `None` to turn off all equivalencies.

        Returns
        -------
        values : scalar or array
            Converted value(s).

        Raises
        ------
        UnitsError
            If units are inconsistent
        """
    def to_string(self, format: str = 'generic'):
        """Output the unit in the given format as a string.

        Units are separated by commas.

        Parameters
        ----------
        format : `astropy.units.format.Base` subclass or str
            The name of a format or a formatter class.  If not
            provided, defaults to the generic format.

        Notes
        -----
        Structured units can be written to all formats, but can be
        re-read only with 'generic'.

        """
    def _repr_latex_(self): ...
    __array_ufunc__: Incomplete
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __rlshift__(self, m): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...

class Structure(np.void):
    """Single element structure for physical type IDs, etc.

    Behaves like a `~numpy.void` and thus mostly like a tuple which can also
    be indexed with field names, but overrides ``__eq__`` and ``__ne__`` to
    compare only the contents, not the field names.  Furthermore, this way no
    `FutureWarning` about comparisons is given.

    """
    def __eq__(self, other): ...
    def __ne__(self, other): ...
