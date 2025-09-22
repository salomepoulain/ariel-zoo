from .core import Cosmology
from _typeshed import Incomplete
from astropy.utils.state import ScienceState
from typing import ClassVar

__all__ = ['available', 'default_cosmology', 'WMAP1', 'WMAP3', 'WMAP5', 'WMAP7', 'WMAP9', 'Planck13', 'Planck15', 'Planck18']

available: Incomplete

class default_cosmology(ScienceState):
    """The default cosmology to use.

    To change it::

        >>> from astropy.cosmology import default_cosmology, WMAP7
        >>> with default_cosmology.set(WMAP7):
        ...     # WMAP7 cosmology in effect
        ...     pass

    Or, you may use a string::

        >>> with default_cosmology.set('WMAP7'):
        ...     # WMAP7 cosmology in effect
        ...     pass

    To get the default cosmology:

        >>> default_cosmology.get()
        FlatLambdaCDM(name='Planck18', H0=<Quantity 67.66 km / (Mpc s)>,
                      Om0=0.30966, ...
    """
    _default_value: ClassVar[str]
    _value: ClassVar[str | Cosmology]
    @classmethod
    def validate(cls, value: Cosmology | str | None) -> Cosmology | None:
        """Return a Cosmology given a value.

        Parameters
        ----------
        value : None, str, or `~astropy.cosmology.Cosmology`

        Returns
        -------
        `~astropy.cosmology.Cosmology` instance

        Raises
        ------
        TypeError
            If ``value`` is not a string or |Cosmology|.
        """
    @classmethod
    def _get_from_registry(cls, name: str) -> Cosmology:
        """Get a registered Cosmology realization.

        Parameters
        ----------
        name : str
            The built-in |Cosmology| realization to retrieve.

        Returns
        -------
        `astropy.cosmology.Cosmology`
            The cosmology realization of `name`.

        Raises
        ------
        ValueError
            If ``name`` is a str, but not for a built-in Cosmology.
        TypeError
            If ``name`` is for a non-Cosmology object.
        """

# Names in __all__ with no definition:
#   Planck13
#   Planck15
#   Planck18
#   WMAP1
#   WMAP3
#   WMAP5
#   WMAP7
#   WMAP9
