from _typeshed import Incomplete
from collections.abc import Callable
from typing import Any

__all__: Incomplete
FValidateCallable = Callable[[object, object, Any], Any]
_REGISTRY_FVALIDATORS: dict[str, FValidateCallable]

def _register_validator(key, fvalidate: Incomplete | None = None):
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
def _validate_with_unit(cosmology, param, value):
    """Default Parameter value validator.

    Adds/converts units if Parameter has a unit.
    """
def _validate_to_float(cosmology, param, value):
    """Parameter value validator with units, and converted to float."""
def _validate_to_scalar(cosmology, param, value): ...
def _validate_non_negative(cosmology, param, value):
    """Parameter value validator where value is a positive float."""
