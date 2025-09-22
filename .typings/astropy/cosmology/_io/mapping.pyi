from astropy.cosmology._typing import _CosmoT as _CosmoT
from astropy.cosmology.connect import convert_registry as convert_registry
from astropy.cosmology.core import Cosmology as Cosmology, _COSMOLOGY_CLASSES as _COSMOLOGY_CLASSES
from collections.abc import Mapping, MutableMapping
from typing import Any, TypeVar

__all__: list[str]
_MapT = TypeVar('_MapT', MutableMapping[str, Any])

def _rename_map(map: Mapping[str, Any], /, renames: Mapping[str, str]) -> dict[str, Any]:
    """Apply rename to map."""
def _get_cosmology_class(cosmology: type[_CosmoT] | str | None, params: dict[str, Any], /) -> type[_CosmoT]: ...
def from_mapping(mapping: Mapping[str, Any], /, *, move_to_meta: bool = False, cosmology: str | type[_CosmoT] | None = None, rename: Mapping[str, str] | None = None) -> _CosmoT:
    '''Load `~astropy.cosmology.Cosmology` from mapping object.

    Parameters
    ----------
    mapping : Mapping
        Arguments into the class -- like "name" or "meta". If \'cosmology\' is None, must
        have field "cosmology" which can be either the string name of the cosmology
        class (e.g. "FlatLambdaCDM") or the class itself.

    move_to_meta : bool (optional, keyword-only)
        Whether to move keyword arguments that are not in the Cosmology class\' signature
        to the Cosmology\'s metadata. This will only be applied if the Cosmology does NOT
        have a keyword-only argument (e.g. ``**kwargs``). Arguments moved to the
        metadata will be merged with existing metadata, preferring specified metadata in
        the case of a merge conflict (e.g. for ``Cosmology(meta={\'key\':10}, key=42)``,
        the ``Cosmology.meta`` will be ``{\'key\': 10}``).

    cosmology : str, |Cosmology| class, or None (optional, keyword-only)
        The cosmology class (or string name thereof) to use when constructing the
        cosmology instance. The class also provides default parameter values, filling in
        any non-mandatory arguments missing in \'map\'.

    rename : Mapping[str, str] or None (optional, keyword-only)
        A mapping of keys in ``map`` to fields of the `~astropy.cosmology.Cosmology`.

    Returns
    -------
    `~astropy.cosmology.Cosmology` subclass instance

    Examples
    --------
    To see loading a `~astropy.cosmology.Cosmology` from a dictionary with
    ``from_mapping``, we will first make a mapping using
    :meth:`~astropy.cosmology.Cosmology.to_format`.

        >>> from astropy.cosmology import Cosmology, Planck18
        >>> cm = Planck18.to_format(\'mapping\')
        >>> cm
        {\'cosmology\': <class \'astropy.cosmology...FlatLambdaCDM\'>,
         \'name\': \'Planck18\', \'H0\': <Quantity 67.66 km / (Mpc s)>, \'Om0\': 0.30966,
         \'Tcmb0\': <Quantity 2.7255 K>, \'Neff\': 3.046,
         \'m_nu\': <Quantity [0. , 0. , 0.06] eV>, \'Ob0\': 0.04897,
         \'meta\': ...

    Now this dict can be used to load a new cosmological instance identical to the
    |Planck18| cosmology from which it was generated.

        >>> cosmo = Cosmology.from_format(cm, format="mapping")
        >>> cosmo
        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                      Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)

    The ``cosmology`` field can be omitted if the cosmology class (or its string name)
    is passed as the ``cosmology`` keyword argument to |Cosmology.from_format|.

        >>> del cm["cosmology"]  # remove cosmology class
        >>> Cosmology.from_format(cm, cosmology="FlatLambdaCDM")
        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                      Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)

    Alternatively, specific cosmology classes can be used to parse the data.

        >>> from astropy.cosmology import FlatLambdaCDM
        >>> FlatLambdaCDM.from_format(cm)
        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                      Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)

    When using a specific cosmology class, the class\' default parameter values are used
    to fill in any missing information.

        >>> del cm["Tcmb0"]  # show FlatLambdaCDM provides default
        >>> FlatLambdaCDM.from_format(cm)
        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                      Tcmb0=0.0 K, Neff=3.046, m_nu=None, Ob0=0.04897)

    The ``move_to_meta`` keyword argument can be used to move fields that are not in the
    Cosmology constructor to the Cosmology\'s metadata. This is useful when the
    dictionary contains extra information that is not part of the Cosmology.

        >>> cm2 = cm | {"extra": 42, "cosmology": "FlatLambdaCDM"}
        >>> cosmo = Cosmology.from_format(cm2, move_to_meta=True)
        >>> cosmo.meta
        OrderedDict([(\'extra\', 42), ...])

    The ``rename`` keyword argument can be used to rename keys in the mapping to fields
    of the |Cosmology|. This is crucial when the mapping has keys that are not valid
    arguments to the |Cosmology| constructor.

        >>> cm3 = dict(cm)  # copy
        >>> cm3["cosmo_cls"] = "FlatLambdaCDM"
        >>> cm3["cosmo_name"] = cm3.pop("name")

        >>> rename = {\'cosmo_cls\': \'cosmology\', \'cosmo_name\': \'name\'}
        >>> Cosmology.from_format(cm3, rename=rename)
        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                      Tcmb0=0.0 K, Neff=3.046, m_nu=None, Ob0=0.04897)
    '''
def to_mapping(cosmology: Cosmology, *args: object, cls: type[_MapT] = ..., cosmology_as_str: bool = False, move_from_meta: bool = False, rename: Mapping[str, str] | None = None) -> _MapT:
    '''Return the cosmology class, parameters, and metadata as a `dict`.

    Parameters
    ----------
    cosmology : :class:`~astropy.cosmology.Cosmology`
        The cosmology instance to convert to a mapping.
    *args : object
        Not used. Needed for compatibility with
        `~astropy.io.registry.UnifiedReadWriteMethod`
    cls : type (optional, keyword-only)
        `dict` or `collections.Mapping` subclass.
        The mapping type to return. Default is `dict`.
    cosmology_as_str : bool (optional, keyword-only)
        Whether the cosmology value is the class (if `False`, default) or
        the semi-qualified name (if `True`).
    move_from_meta : bool (optional, keyword-only)
        Whether to add the Cosmology\'s metadata as an item to the mapping (if
        `False`, default) or to merge with the rest of the mapping, preferring
        the original values (if `True`)
    rename : Mapping[str, str] or None (optional, keyword-only)
        A mapping of field names of the :class:`~astropy.cosmology.Cosmology` to keys in
        the map.

    Returns
    -------
    MutableMapping[str, Any]
        A mapping of type ``cls``, by default a `dict`.
        Has key-values for the cosmology parameters and also:
        - \'cosmology\' : the class
        - \'meta\' : the contents of the cosmology\'s metadata attribute.
                   If ``move_from_meta`` is `True`, this key is missing and the
                   contained metadata are added to the main `dict`.

    Examples
    --------
    A Cosmology as a mapping will have the cosmology\'s name and
    parameters as items, and the metadata as a nested dictionary.

        >>> from astropy.cosmology import Planck18
        >>> Planck18.to_format(\'mapping\')
        {\'cosmology\': <class \'astropy.cosmology...FlatLambdaCDM\'>,
         \'name\': \'Planck18\', \'H0\': <Quantity 67.66 km / (Mpc s)>, \'Om0\': 0.30966,
         \'Tcmb0\': <Quantity 2.7255 K>, \'Neff\': 3.046,
         \'m_nu\': <Quantity [0.  , 0.  , 0.06] eV>, \'Ob0\': 0.04897,
         \'meta\': ...

    The dictionary type may be changed with the ``cls`` keyword argument:

        >>> from collections import OrderedDict
        >>> Planck18.to_format(\'mapping\', cls=OrderedDict)
        OrderedDict([(\'cosmology\', <class \'astropy.cosmology...FlatLambdaCDM\'>),
          (\'name\', \'Planck18\'), (\'H0\', <Quantity 67.66 km / (Mpc s)>),
          (\'Om0\', 0.30966), (\'Tcmb0\', <Quantity 2.7255 K>), (\'Neff\', 3.046),
          (\'m_nu\', <Quantity [0.  , 0.  , 0.06] eV>), (\'Ob0\', 0.04897),
          (\'meta\', ...

    Sometimes it is more useful to have the name of the cosmology class, not
    the type itself. The keyword argument ``cosmology_as_str`` may be used:

        >>> Planck18.to_format(\'mapping\', cosmology_as_str=True)
        {\'cosmology\': \'FlatLambdaCDM\', ...

    The metadata is normally included as a nested mapping. To move the metadata
    into the main mapping, use the keyword argument ``move_from_meta``. This
    kwarg inverts ``move_to_meta`` in
    ``Cosmology.to_format("mapping", move_to_meta=...)`` where extra items
    are moved to the metadata (if the cosmology constructor does not have a
    variable keyword-only argument -- ``**kwargs``).

        >>> from astropy.cosmology import Planck18
        >>> Planck18.to_format(\'mapping\', move_from_meta=True)
        {\'cosmology\': <class \'astropy.cosmology...FlatLambdaCDM\'>,
         \'name\': \'Planck18\', \'Oc0\': 0.2607, \'n\': 0.9665, \'sigma8\': 0.8102, ...

    Lastly, the keys in the mapping may be renamed with the ``rename`` keyword.

        >>> rename = {\'cosmology\': \'cosmo_cls\', \'name\': \'cosmo_name\'}
        >>> Planck18.to_format(\'mapping\', rename=rename)
        {\'cosmo_cls\': <class \'astropy.cosmology...FlatLambdaCDM\'>,
         \'cosmo_name\': \'Planck18\', ...
    '''
def mapping_identify(origin: str, format: str | None, *args: object, **kwargs: object) -> bool:
    """Identify if object uses the mapping format.

    Returns
    -------
    bool
    """
