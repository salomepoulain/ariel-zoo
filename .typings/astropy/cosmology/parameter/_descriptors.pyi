from astropy.cosmology.core import Cosmology as Cosmology
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NoReturn

__all__: list[str]

@dataclass(frozen=True, slots=True)
class ParametersAttribute:
    """Immutable mapping of the :class:`~astropy.cosmology.Parameter` objects or values.

    If accessed from the :class:`~astropy.cosmology.Cosmology` class, this returns a
    mapping of the :class:`~astropy.cosmology.Parameter` objects themselves.  If
    accessed from an instance, this returns a mapping of the values of the Parameters.

    This class is used to implement :obj:`astropy.cosmology.Cosmology.parameters`.

    Parameters
    ----------
    attr_name : str
        The name of the class attribute that is a `~types.MappingProxyType[str,
        astropy.cosmology.Parameter]` of all the cosmology's parameters. When accessed
        from the class, this attribute is returned. When accessed from an instance, a
        mapping of the cosmology instance's values for each key is returned.

    Examples
    --------
    The normal usage of this class is the ``parameters`` attribute of
    :class:`~astropy.cosmology.Cosmology`.

        >>> from astropy.cosmology import FlatLambdaCDM, Planck18

        >>> FlatLambdaCDM.parameters
        mappingproxy({'H0': Parameter(...), ...})

        >>> Planck18.parameters
        mappingproxy({'H0': <Quantity 67.66 km / (Mpc s)>, ...})
    """
    attr_name: str
    _name: str = field(init=False)
    def __set_name__(self, owner: Any, name: str) -> None: ...
    def __get__(self, instance: Cosmology | None, owner: type[Cosmology] | None) -> MappingProxyType[str, Any]: ...
    def __set__(self, instance: Any, value: Any) -> NoReturn: ...
