from .earth import EarthLocation as EarthLocation
from .errors import UnknownSiteException as UnknownSiteException
from _typeshed import Incomplete
from astropy.utils.data import get_file_contents as get_file_contents, get_pkg_data_contents as get_pkg_data_contents
from collections.abc import Mapping

class SiteRegistry(Mapping):
    """
    A bare-bones registry of EarthLocation objects.

    This acts as a mapping (dict-like object) but with the important caveat that
    it's always transforms its inputs to lower-case.  So keys are always all
    lower-case, and even if you ask for something that's got mixed case, it will
    be interpreted as the all lower-case version.
    """
    _lowercase_names_to_locations: Incomplete
    _names: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, site_name):
        """
        Returns an EarthLocation for a known site in this registry.

        Parameters
        ----------
        site_name : str
            Name of the observatory (case-insensitive).

        Returns
        -------
        site : `~astropy.coordinates.EarthLocation`
            The location of the observatory.
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, site_name) -> bool: ...
    @property
    def names(self):
        """
        The names in this registry.  Note that these are *not* exactly the same
        as the keys: keys are always lower-case, while `names` is what you
        should use for the actual readable names (which may be case-sensitive).

        Returns
        -------
        site : list of str
            The names of the sites in this registry
        """
    def add_site(self, names, locationobj) -> None:
        """
        Adds a location to the registry.

        Parameters
        ----------
        names : list of str
            All the names this site should go under
        locationobj : `~astropy.coordinates.EarthLocation`
            The actual site object
        """
    @classmethod
    def from_json(cls, jsondb): ...

def get_builtin_sites():
    """
    Load observatory database from data/observatories.json and parse them into
    a SiteRegistry.
    """
def get_downloaded_sites(jsonurl: Incomplete | None = None):
    """
    Load observatory database from data.astropy.org and parse into a SiteRegistry.
    """
