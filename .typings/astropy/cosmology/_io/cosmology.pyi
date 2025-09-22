from astropy.cosmology._typing import _CosmoT as _CosmoT
from astropy.cosmology.connect import convert_registry as convert_registry
from astropy.cosmology.core import Cosmology as Cosmology, _COSMOLOGY_CLASSES as _COSMOLOGY_CLASSES

__all__: list[str]

def from_cosmology(cosmo: _CosmoT, /, cosmology: type[_CosmoT] | str | None = None, **kwargs: object) -> _CosmoT:
    '''Return the |Cosmology| unchanged.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
        The cosmology to return.
    cosmology : type[`~astropy.cosmology.Cosmology`] | str | None, optional
        The |Cosmology| class to check against. If not `None`, ``cosmo`` is checked
        for correctness.
    **kwargs
        This argument is required for compatibility with the standard set of
        keyword arguments in format |Cosmology.from_format|.

    Returns
    -------
    `~astropy.cosmology.Cosmology` subclass instance
        Just ``cosmo`` passed through.

    Raises
    ------
    TypeError
        If the |Cosmology| object is not an instance of ``cosmo`` (and
        ``cosmology`` is not `None`).

    Examples
    --------
    >>> from astropy.cosmology import Cosmology, Planck18
    >>> print(Cosmology.from_format(Planck18))
    FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                  Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)
    '''
def to_cosmology(cosmo: _CosmoT, *args: object) -> _CosmoT:
    '''Return the |Cosmology| unchanged.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
        The cosmology to return.
    *args : object
        Not used.

    Returns
    -------
    `~astropy.cosmology.Cosmology` subclass instance
        Just ``cosmo`` passed through.

    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> Planck18.to_format("astropy.cosmology") is Planck18
    True
    '''
def cosmology_identify(origin: str, format: str | None, *args: object, **kwargs: object) -> bool:
    '''Identify if object is a `~astropy.cosmology.Cosmology`.

    This checks if the 2nd argument is a |Cosmology| instance and the format is
    "astropy.cosmology" or `None`.

    Returns
    -------
    bool
    '''
