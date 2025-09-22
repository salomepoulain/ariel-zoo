import astropy.units as u
from ._converter import FValidateCallable
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import Any

__all__ = ['Parameter']

class Sentinel(Enum):
    """Sentinel values for Parameter fields."""
    MISSING = ...
    def __repr__(self) -> str: ...

@dataclass(frozen=True)
class _UnitField:
    def __get__(self, obj: Parameter | None, objcls: type[Parameter] | None) -> u.Unit | None: ...
    def __set__(self, obj: Parameter, value: Any) -> None: ...

@dataclass(frozen=True)
class _FValidateField:
    default: FValidateCallable | str = ...
    def __get__(self, obj: Parameter | None, objcls: type[Parameter] | None) -> FValidateCallable | str: ...
    def __set__(self, obj: Parameter, value: Any) -> None: ...

@dataclass(frozen=True)
class Parameter:
    '''Cosmological parameter (descriptor).

    Should only be used with a :class:`~astropy.cosmology.Cosmology` subclass.

    Parameters
    ----------
    default : Any (optional, keyword-only)
        Default value of the Parameter. If not given the
        Parameter must be set when initializing the cosmology.
    derived : bool (optional, keyword-only)
        Whether the Parameter is \'derived\', default `False`.
        Derived parameters behave similarly to normal parameters, but are not
        sorted by the |Cosmology| signature (probably not there) and are not
        included in all methods. For reference, see ``Ode0`` in
        ``FlatFLRWMixin``, which removes :math:`\\Omega_{de,0}`` as an
        independent parameter (:math:`\\Omega_{de,0} \\equiv 1 - \\Omega_{tot}`).
    unit : unit-like or None (optional, keyword-only)
        The `~astropy.units.Unit` for the Parameter. If None (default) no
        unit as assumed.
    equivalencies : `~astropy.units.Equivalency` or sequence thereof
        Unit equivalencies for this Parameter.
    fvalidate : callable[[object, object, Any], Any] or str (optional, keyword-only)
        Function to validate the Parameter value from instances of the
        cosmology class. If "default", uses default validator to assign units
        (with equivalencies), if Parameter has units.
        For other valid string options, see ``Parameter._registry_validators``.
        \'fvalidate\' can also be set through a decorator with
        :meth:`~astropy.cosmology.Parameter.validator`.
    doc : str or None (optional, keyword-only)
        Parameter description.

    Examples
    --------
    For worked examples see :class:`~astropy.cosmology.FLRW`.
    '''
    _: KW_ONLY
    default: Any = ...
    derived: bool = ...
    unit: _UnitField = ...
    equivalencies: u.Equivalency | Sequence[u.Equivalency] = field(default_factory=list)
    fvalidate: _FValidateField = ...
    doc: str | None = ...
    name: str = field(init=False, compare=True, default=None, repr=False)
    _fvalidate_in: FValidateCallable | str
    _fvalidate: FValidateCallable
    def __post_init__(self) -> None: ...
    def __set_name__(self, cosmo_cls: type, name: str) -> None: ...
    def __get__(self, cosmology, cosmo_cls: Incomplete | None = None): ...
    def __set__(self, cosmology, value) -> None:
        """Allows attribute setting once.

        Raises AttributeError subsequently.
        """
    def validator(self, fvalidate):
        """Make new Parameter with custom ``fvalidate``.

        Note: ``Parameter.fvalidator`` must be the top-most descriptor decorator.

        Parameters
        ----------
        fvalidate : callable[[type, type, Any], Any]

        Returns
        -------
        `~astropy.cosmology.Parameter`
            Copy of this Parameter but with custom ``fvalidate``.
        """
    def validate(self, cosmology, value):
        """Run the validator on this Parameter.

        Parameters
        ----------
        cosmology : `~astropy.cosmology.Cosmology` instance
        value : Any
            The object to validate.

        Returns
        -------
        Any
            The output of calling ``fvalidate(cosmology, self, value)``
            (yes, that parameter order).
        """
    @staticmethod
    def register_validator(key, fvalidate: Incomplete | None = None):
        """Decorator to register a new kind of validator function.

        Parameters
        ----------
        key : str
        fvalidate : callable[[object, object, Any], Any] or None, optional
            Value validation function.

        Returns
        -------
        ``validator`` or callable[``validator``]
            if validator is None returns a function that takes and registers a
            validator. This allows ``register_validator`` to be used as a
            decorator.
        """
    def clone(self, **kw):
        '''Clone this `Parameter`, changing any constructor argument.

        Parameters
        ----------
        **kw
            Passed to constructor. The current values, eg. ``fvalidate`` are
            used as the default values, so an empty ``**kw`` is an exact copy.

        Examples
        --------
        >>> p = Parameter()
        >>> p
        Parameter(derived=False, unit=None, equivalencies=[],
                  fvalidate=\'default\', doc=None)

        >>> p.clone(unit="km")
        Parameter(derived=False, unit=Unit("km"), equivalencies=[],
                  fvalidate=\'default\', doc=None)
        '''
    def __repr__(self) -> str:
        """Return repr(self)."""
