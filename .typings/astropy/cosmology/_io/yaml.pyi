from .mapping import from_mapping as from_mapping
from astropy.cosmology._typing import _CosmoT as _CosmoT
from astropy.cosmology.connect import convert_registry as convert_registry
from astropy.cosmology.core import Cosmology as Cosmology, _COSMOLOGY_CLASSES as _COSMOLOGY_CLASSES
from astropy.io.misc.yaml import AstropyDumper as AstropyDumper, AstropyLoader as AstropyLoader, dump as dump, load as load
from collections.abc import Callable as Callable
from yaml import MappingNode as MappingNode

__all__: list[str]
_representer_doc: str

def yaml_representer(tag: str) -> Callable[[AstropyDumper, Cosmology], str]:
    """`yaml <https://yaml.org>`_ representation of |Cosmology| object.

    Parameters
    ----------
    tag : str
        The class tag, e.g. '!astropy.cosmology.LambdaCDM'

    Returns
    -------
    representer : callable[[`~astropy.io.misc.yaml.AstropyDumper`, |Cosmology|], str]
        Function to construct :mod:`yaml` representation of |Cosmology| object.
    """
def yaml_constructor(cls) -> Callable[[AstropyLoader, MappingNode], _CosmoT]:
    """Cosmology| object from :mod:`yaml` representation.

    Parameters
    ----------
    cls : type
        The class type, e.g. `~astropy.cosmology.LambdaCDM`.

    Returns
    -------
    constructor : callable
        Function to construct |Cosmology| object from :mod:`yaml` representation.
    """
def register_cosmology_yaml(cosmo_cls: type[Cosmology]) -> None:
    """Register :mod:`yaml` for Cosmology class.

    Parameters
    ----------
    cosmo_cls : `~astropy.cosmology.Cosmology` class
    """
def from_yaml(yml: str, *, cosmology: type[_CosmoT] | None = None) -> _CosmoT:
    '''Load `~astropy.cosmology.Cosmology` from :mod:`yaml` object.

    Parameters
    ----------
    yml : str
        :mod:`yaml` representation of |Cosmology| object
    cosmology : str, |Cosmology| class, or None (optional, keyword-only)
        The expected cosmology class (or string name thereof). This argument is
        is only checked for correctness if not `None`.

    Returns
    -------
    `~astropy.cosmology.Cosmology` subclass instance

    Raises
    ------
    TypeError
        If the |Cosmology| object loaded from ``yml`` is not an instance of
        the ``cosmology`` (and ``cosmology`` is not `None`).

    Examples
    --------
    >>> from astropy.cosmology import Cosmology, Planck18
    >>> yml = Planck18.to_format("yaml")
    >>> print(Cosmology.from_format(yml, format="yaml"))
    FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                  Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)
    '''
def to_yaml(cosmology: Cosmology, *args: object) -> str:
    '''Return the cosmology class, parameters, and metadata as a :mod:`yaml` object.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.Cosmology` subclass instance
        The cosmology to serialize.
    *args : Any
        Not used. Needed for compatibility with
        `~astropy.io.registry.UnifiedReadWriteMethod`

    Returns
    -------
    str
        :mod:`yaml` representation of |Cosmology| object

    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> Planck18.to_format("yaml")
    "!astropy.cosmology...FlatLambdaCDM\\nH0: !astropy.units.Quantity...
    '''
