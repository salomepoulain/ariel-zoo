from astropy import table as table
from astropy.cosmology.core import Cosmology as Cosmology
from collections.abc import Callable as Callable
from dataclasses import dataclass
from numpy import ndarray
from typing import Any, TypeAlias

__all__: list[str]
_FormatType: TypeAlias
_FormatsType: TypeAlias
_COSMO_AOK: set[Any]
_CANT_BROADCAST: tuple[type, ...]

@dataclass(frozen=True)
class _CosmologyWrapper:
    """A private wrapper class to hide things from :mod:`numpy`.

    This should never be exposed to the user.
    """
    __slots__ = ...
    wrapped: Any

def _parse_format(cosmo: Any, format: _FormatType, /) -> Cosmology:
    """Parse Cosmology-like input into Cosmologies, given a format hint.

    Parameters
    ----------
    cosmo : |Cosmology|-like, positional-only
        |Cosmology| to parse.
    format : bool or None or str, positional-only
        Whether to allow, before equivalence is checked, the object to be
        converted to a |Cosmology|. This allows, e.g. a |Table| to be equivalent
        to a |Cosmology|. `False` (default) will not allow conversion. `True` or
        `None` will, and will use the auto-identification to try to infer the
        correct format. A `str` is assumed to be the correct format to use when
        converting.

    Returns
    -------
    |Cosmology| or generator thereof

    Raises
    ------
    TypeError
        If ``cosmo`` is not a |Cosmology| and ``format`` equals `False`.
    TypeError
        If ``cosmo`` is a |Cosmology| and ``format`` is not `None` or equal to
        `True`.
    """
def _parse_formats(*cosmos: object, format: _FormatsType) -> ndarray:
    """Parse Cosmology-like to |Cosmology|, using provided formats.

    ``format`` is broadcast to match the shape of the cosmology arguments. Note
    that the cosmology arguments are not broadcast against ``format``, so it
    cannot determine the output shape.

    Parameters
    ----------
    *cosmos : |Cosmology|-like
        The objects to compare. Must be convertible to |Cosmology|, as specified
        by the corresponding ``format``.

    format : bool or None or str or array-like thereof, positional-only
        Whether to allow, before equivalence is checked, the object to be
        converted to a |Cosmology|. This allows, e.g. a |Table| to be equivalent
        to a |Cosmology|. `False` (default) will not allow conversion. `True` or
        `None` will, and will use the auto-identification to try to infer the
        correct format. A `str` is assumed to be the correct format to use when
        converting. Note ``format`` is broadcast as an object array to match the
        shape of ``cosmos`` so ``format`` cannot determine the output shape.

    Raises
    ------
    TypeError
        If any in ``cosmos`` is not a |Cosmology| and the corresponding
        ``format`` equals `False`.
    """
def _comparison_decorator(pyfunc: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to make wrapper function that parses |Cosmology|-like inputs.

    Parameters
    ----------
    pyfunc : Python function object
        An arbitrary Python function.

    Returns
    -------
    callable[..., Any]
        Wrapped `pyfunc`, as described above.

    Notes
    -----
    All decorated functions should add the following to 'Parameters'.

    format : bool or None or str or array-like thereof, optional keyword-only
        Whether to allow the arguments to be converted to a |Cosmology|. This
        allows, e.g. a |Table| to be given instead a |Cosmology|. `False`
        (default) will not allow conversion. `True` or `None` will, and will use
        the auto-identification to try to infer the correct format. A `str` is
        assumed to be the correct format to use when converting. Note ``format``
        is broadcast as an object array to match the shape of ``cosmos`` so
        ``format`` cannot determine the output shape.
    """
def cosmology_equal(cosmo1: Any, cosmo2: Any, /, *, allow_equivalent: bool = False) -> bool:
    '''Return element-wise equality check on the cosmologies.

    .. note::

        Cosmologies are currently scalar in their parameters.

    Parameters
    ----------
    cosmo1, cosmo2 : |Cosmology|-like
        The objects to compare. Must be convertible to |Cosmology|, as specified
        by ``format``.

    format : bool or None or str or tuple thereof, optional keyword-only
        Whether to allow the arguments to be converted to a |Cosmology|. This
        allows, e.g. a |Table| to be given instead a |Cosmology|. `False`
        (default) will not allow conversion. `True` or `None` will, and will use
        the auto-identification to try to infer the correct format. A `str` is
        assumed to be the correct format to use when converting. Note ``format``
        is broadcast as an object array to match the shape of ``cosmos`` so
        ``format`` cannot determine the output shape.

    allow_equivalent : bool, optional keyword-only
        Whether to allow cosmologies to be equal even if not of the same class.
        For example, an instance of |LambdaCDM| might have :math:`\\Omega_0=1`
        and :math:`\\Omega_k=0` and therefore be flat, like |FlatLambdaCDM|.

    Examples
    --------
    Assuming the following imports

        >>> import astropy.units as u
        >>> from astropy.cosmology import FlatLambdaCDM

    Two identical cosmologies are equal.

        >>> cosmo1 = FlatLambdaCDM(70 * (u.km/u.s/u.Mpc), 0.3)
        >>> cosmo2 = FlatLambdaCDM(70 * (u.km/u.s/u.Mpc), 0.3)
        >>> cosmology_equal(cosmo1, cosmo2)
        True

    And cosmologies with different parameters are not.

        >>> cosmo3 = FlatLambdaCDM(70 * (u.km/u.s/u.Mpc), 0.4)
        >>> cosmology_equal(cosmo1, cosmo3)
        False

    Two cosmologies may be equivalent even if not of the same class. In these
    examples the |LambdaCDM| has :attr:`~astropy.cosmology.LambdaCDM.Ode0` set
    to the same value calculated in |FlatLambdaCDM|.

        >>> from astropy.cosmology import LambdaCDM
        >>> cosmo3 = LambdaCDM(70 * (u.km/u.s/u.Mpc), 0.3, 0.7)
        >>> cosmology_equal(cosmo1, cosmo3)
        False
        >>> cosmology_equal(cosmo1, cosmo3, allow_equivalent=True)
        True

    While in this example, the cosmologies are not equivalent.

        >>> cosmo4 = FlatLambdaCDM(70 * (u.km/u.s/u.Mpc), 0.3, Tcmb0=3 * u.K)
        >>> cosmology_equal(cosmo3, cosmo4, allow_equivalent=True)
        False

    Also, using the keyword argument, the notion of equality is extended to any
    Python object that can be converted to a |Cosmology|.

        >>> mapping = cosmo2.to_format("mapping")
        >>> cosmology_equal(cosmo1, mapping, format=True)
        True

    Either (or both) arguments can be |Cosmology|-like.

        >>> cosmology_equal(mapping, cosmo2, format=True)
        True

    The list of valid formats, e.g. the |Table| in this example, may be checked
    with ``Cosmology.from_format.list_formats()``.

    As can be seen in the list of formats, not all formats can be
    auto-identified by ``Cosmology.from_format.registry``. Objects of these
    kinds can still be checked for equality, but the correct format string must
    be used.

        >>> yml = cosmo2.to_format("yaml")
        >>> cosmology_equal(cosmo1, yml, format=(None, "yaml"))
        True

    This also works with an array of ``format`` matching the number of
    cosmologies.

        >>> cosmology_equal(mapping, yml, format=[True, "yaml"])
        True
    '''
def _cosmology_not_equal(cosmo1: Any, cosmo2: Any, /, *, allow_equivalent: bool = False) -> bool:
    """Return element-wise cosmology non-equality check.

    .. note::

        Cosmologies are currently scalar in their parameters.

    Parameters
    ----------
    cosmo1, cosmo2 : |Cosmology|-like
        The objects to compare. Must be convertible to |Cosmology|, as specified
        by ``format``.

    out : ndarray, None, optional
        A location into which the result is stored. If provided, it must have a
        shape that the inputs broadcast to. If not provided or None, a
        freshly-allocated array is returned.

    format : bool or None or str or tuple thereof, optional keyword-only
        Whether to allow the arguments to be converted to a |Cosmology|. This
        allows, e.g. a |Table| to be given instead a Cosmology. `False`
        (default) will not allow conversion. `True` or `None` will, and will use
        the auto-identification to try to infer the correct format. A `str` is
        assumed to be the correct format to use when converting. ``format`` is
        broadcast to match the shape of the cosmology arguments. Note that the
        cosmology arguments are not broadcast against ``format``, so it cannot
        determine the output shape.

    allow_equivalent : bool, optional keyword-only
        Whether to allow cosmologies to be equal even if not of the same class.
        For example, an instance of |LambdaCDM| might have :math:`\\Omega_0=1`
        and :math:`\\Omega_k=0` and therefore be flat, like |FlatLambdaCDM|.

    See Also
    --------
    astropy.cosmology.cosmology_equal
        Element-wise equality check, with argument conversion to Cosmology.
    """
