from _typeshed import Incomplete
from astropy.units.quantity import Quantity

__all__ = ['Constant', 'EMConstant']

class ConstantMeta(type):
    """Metaclass for `~astropy.constants.Constant`. The primary purpose of this
    is to wrap the double-underscore methods of `~astropy.units.Quantity`
    which is the superclass of `~astropy.constants.Constant`.

    In particular this wraps the operator overloads such as `__add__` to
    prevent their use with constants such as ``e`` from being used in
    expressions without specifying a system.  The wrapper checks to see if the
    constant is listed (by name) in ``Constant._has_incompatible_units``, a set
    of those constants that are defined in different systems of units are
    physically incompatible.  It also performs this check on each `Constant` if
    it hasn't already been performed (the check is deferred until the
    `Constant` is actually used in an expression to speed up import times,
    among other reasons).
    """
    _checked_units: bool
    def __new__(mcls, name, bases, d): ...

class Constant(Quantity, metaclass=ConstantMeta):
    """A physical or astronomical constant.

    These objects are quantities that are meant to represent physical
    constants.

    Parameters
    ----------
    abbrev : str
        A typical ASCII text abbreviation of the constant, generally
        the same as the Python variable used for this constant.
    name : str
        Full constant name.
    value : numbers.Real
        Constant value. Note that this should be a bare number, not a
        |Quantity|.
    unit : str
        String representation of the constant units.
    uncertainty : numbers.Real
        Absolute uncertainty in constant value. Note that this should be
        a bare number, not a |Quantity|.
    reference : str, optional
        Reference where the value is taken from.
    system : str
        System of units in which the constant is defined. This can be
        `None` when the constant's units can be directly converted
        between systems.
    """
    _registry: Incomplete
    _has_incompatible_units: Incomplete
    def __new__(cls, abbrev, name, value, unit, uncertainty, reference: Incomplete | None = None, system: Incomplete | None = None): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __quantity_subclass__(self, unit): ...
    def copy(self):
        """
        Return a copy of this `Constant` instance.  Since they are by
        definition immutable, this merely returns another reference to
        ``self``.
        """
    __deepcopy__ = copy
    __copy__ = copy
    @property
    def abbrev(self):
        """A typical ASCII text abbreviation of the constant, also generally
        the same as the Python variable used for this constant.
        """
    @property
    def name(self):
        """The full name of the constant."""
    def _unit(self):
        """The unit(s) in which this constant is defined."""
    @property
    def uncertainty(self):
        """The known absolute uncertainty in this constant's value."""
    @property
    def reference(self):
        """The source used for the value of this constant."""
    @property
    def system(self):
        """The system of units in which this constant is defined (typically
        `None` so long as the constant's units can be directly converted
        between systems).
        """
    def _instance_or_super(self, key): ...
    @property
    def si(self):
        """If the Constant is defined in the SI system return that instance of
        the constant, else convert to a Quantity in the appropriate SI units.
        """
    @property
    def cgs(self):
        """If the Constant is defined in the CGS system return that instance of
        the constant, else convert to a Quantity in the appropriate CGS units.
        """
    _checked_units: Incomplete
    def __array_finalize__(self, obj) -> None: ...

class EMConstant(Constant):
    """An electromagnetic constant."""
    @property
    def cgs(self) -> None:
        """Overridden for EMConstant to raise a `TypeError`
        emphasizing that there are multiple EM extensions to CGS.
        """
