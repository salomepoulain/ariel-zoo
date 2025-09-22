from _typeshed import Incomplete

__all__ = ['def_physical_type', 'get_physical_type', 'PhysicalType']

class PhysicalType:
    '''
    Represents the physical type(s) that are dimensionally compatible
    with a set of units.

    Instances of this class should be accessed through either
    `get_physical_type` or by using the
    `~astropy.units.core.UnitBase.physical_type` attribute of units.
    This class is not intended to be instantiated directly in user code.

    For a list of physical types, see `astropy.units.physical`.

    Parameters
    ----------
    unit : `~astropy.units.Unit`
        The unit to be represented by the physical type.

    physical_types : `str` or `set` of `str`
        A `str` representing the name of the physical type of the unit,
        or a `set` containing strings that represent one or more names
        of physical types.

    Notes
    -----
    A physical type will be considered equal to an equivalent
    `PhysicalType` instance (recommended) or a string that contains a
    name of the physical type.  The latter method is not recommended
    in packages, as the names of some physical types may change in the
    future.

    To maintain backwards compatibility, two physical type names may be
    included in one string if they are separated with a slash (e.g.,
    ``"momentum/impulse"``).  String representations of physical types
    may include underscores instead of spaces.

    Examples
    --------
    `PhysicalType` instances may be accessed via the
    `~astropy.units.core.UnitBase.physical_type` attribute of units.

    >>> import astropy.units as u
    >>> u.meter.physical_type
    PhysicalType(\'length\')

    `PhysicalType` instances may also be accessed by calling
    `get_physical_type`. This function will accept a unit, a string
    containing the name of a physical type, or the number one.

    >>> u.get_physical_type(u.m ** -3)
    PhysicalType(\'number density\')
    >>> u.get_physical_type("volume")
    PhysicalType(\'volume\')
    >>> u.get_physical_type(1)
    PhysicalType(\'dimensionless\')

    Some units are dimensionally compatible with multiple physical types.
    A pascal is intended to represent pressure and stress, but the unit
    decomposition is equivalent to that of energy density.

    >>> pressure = u.get_physical_type("pressure")
    >>> pressure
    PhysicalType({\'energy density\', \'pressure\', \'stress\'})
    >>> \'energy density\' in pressure
    True

    Physical types can be tested for equality against other physical
    type objects or against strings that may contain the name of a
    physical type.

    >>> area = (u.m ** 2).physical_type
    >>> area == u.barn.physical_type
    True
    >>> area == "area"
    True

    Multiplication, division, and exponentiation are enabled so that
    physical types may be used for dimensional analysis.

    >>> length = u.pc.physical_type
    >>> area = (u.cm ** 2).physical_type
    >>> length * area
    PhysicalType(\'volume\')
    >>> area / length
    PhysicalType(\'length\')
    >>> length ** 3
    PhysicalType(\'volume\')

    may also be performed using a string that contains the name of a
    physical type.

    >>> "length" * area
    PhysicalType(\'volume\')
    >>> "area" / length
    PhysicalType(\'length\')

    Unknown physical types are labelled as ``"unknown"``.

    >>> (u.s ** 13).physical_type
    PhysicalType(\'unknown\')

    Dimensional analysis may be performed for unknown physical types too.

    >>> length_to_19th_power = (u.m ** 19).physical_type
    >>> length_to_20th_power = (u.m ** 20).physical_type
    >>> length_to_20th_power / length_to_19th_power
    PhysicalType(\'length\')
    '''
    _unit: Incomplete
    _physical_type: Incomplete
    _physical_type_list: Incomplete
    def __init__(self, unit, physical_types) -> None: ...
    def __iter__(self): ...
    def __eq__(self, other):
        """
        Return `True` if ``other`` represents a physical type that is
        consistent with the physical type of the `PhysicalType` instance.
        """
    def __ne__(self, other): ...
    def _name_string_as_ordered_set(self): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def _dimensionally_compatible_unit(obj):
        """
        Return a unit that corresponds to the provided argument.

        If a unit is passed in, return that unit.  If a physical type
        (or a `str` with the name of a physical type) is passed in,
        return a unit that corresponds to that physical type.  If the
        number equal to ``1`` is passed in, return a dimensionless unit.
        Otherwise, return `NotImplemented`.
        """
    def _dimensional_analysis(self, other, operation): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __pow__(self, power): ...
    def __hash__(self): ...
    def __len__(self) -> int: ...
    __array__: Incomplete

def def_physical_type(unit, name) -> None:
    '''
    Add a mapping between a unit and the corresponding physical type(s).

    If a physical type already exists for a unit, add new physical type
    names so long as those names are not already in use for other
    physical types.

    Parameters
    ----------
    unit : `~astropy.units.Unit`
        The unit to be represented by the physical type.

    name : `str` or `set` of `str`
        A `str` representing the name of the physical type of the unit,
        or a `set` containing strings that represent one or more names
        of physical types.

    Raises
    ------
    ValueError
        If a physical type name is already in use for another unit, or
        if attempting to name a unit as ``"unknown"``.

    Notes
    -----
    For a list of physical types, see `astropy.units.physical`.
    '''
def get_physical_type(obj):
    '''
    Return the physical type that corresponds to a unit (or another
    physical type representation).

    Parameters
    ----------
    obj : quantity-like or `~astropy.units.PhysicalType`-like
        An object that (implicitly or explicitly) has a corresponding
        physical type. This object may be a unit, a
        `~astropy.units.Quantity`, an object that can be converted to a
        `~astropy.units.Quantity` (such as a number or array), a string
        that contains a name of a physical type, or a
        `~astropy.units.PhysicalType` instance.

    Returns
    -------
    `~astropy.units.PhysicalType`
        A representation of the physical type(s) of the unit.

    Notes
    -----
    For a list of physical types, see `astropy.units.physical`.

    Examples
    --------
    The physical type may be retrieved from a unit or a
    `~astropy.units.Quantity`.

    >>> import astropy.units as u
    >>> u.get_physical_type(u.meter ** -2)
    PhysicalType(\'column density\')
    >>> u.get_physical_type(0.62 * u.barn * u.Mpc)
    PhysicalType(\'volume\')

    The physical type may also be retrieved by providing a `str` that
    contains the name of a physical type.

    >>> u.get_physical_type("energy")
    PhysicalType({\'energy\', \'torque\', \'work\'})

    Numbers and arrays of numbers correspond to a dimensionless physical
    type.

    >>> u.get_physical_type(1)
    PhysicalType(\'dimensionless\')
    '''
physical_type = _name_physical_mapping[name]
