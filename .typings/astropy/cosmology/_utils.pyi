from ._signature_deprecations import _depr_kws_wrap as _depr_kws_wrap
from _typeshed import Incomplete
from astropy.cosmology import Parameter as Parameter
from astropy.units import Quantity as Quantity
from collections.abc import Callable
from dataclasses import Field
from typing import Any, TypeVar

__all__: list[str]
_F = TypeVar('_F', bound=Callable[..., Any])

def vectorize_redshift_method(func: Incomplete | None = None, nin: int = 1):
    """Vectorize a method of redshift(s).

    Parameters
    ----------
    func : callable or None
        method to wrap. If `None` returns a :func:`functools.partial`
        with ``nin`` loaded.
    nin : int
        Number of positional redshift arguments.

    Returns
    -------
    wrapper : callable
        :func:`functools.wraps` of ``func`` where the first ``nin``
        arguments are converted from |Quantity| to :class:`numpy.ndarray`.
    """
def aszarr(z):
    '''Redshift as a `~numbers.Number` or |ndarray| / |Quantity| / |Column|.

    Allows for any ndarray ducktype by checking for attribute "shape".
    '''
def all_cls_vars(obj: object | type, /) -> dict[str, Any]:
    """Return all variables in the whole class hierarchy."""
def all_parameters(obj: object, /) -> dict[str, Field | Parameter]:
    """Get all fields of a dataclass, including those not-yet finalized.

    Parameters
    ----------
    obj : object | type
        A dataclass.

    Returns
    -------
    dict[str, Field | Parameter]
        All fields of the dataclass, including those not yet finalized in the class, if
        it's still under construction, e.g. in ``__init_subclass__``.
    """
def deprecated_keywords(*kws, since):
    """Deprecate calling one or more arguments as keywords.

    Parameters
    ----------
    *kws: str
        Names of the arguments that will become positional-only.

    since : str or number or sequence of str or number
        The release at which the old argument became deprecated.
    """
def _depr_kws(func: _F, /, kws: tuple[str, ...], since: str) -> _F: ...
