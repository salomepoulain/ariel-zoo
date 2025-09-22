from . import config as _config
from .logger import _teardown_log as _teardown_log
from .utils.misc import find_api_page as find_api_page
from .utils.state import ScienceState
from _typeshed import Incomplete
from astropy.utils.system_info import system_info as system_info

online_docs_root: Incomplete

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy`.
    """
    unicode_output: Incomplete
    use_color: Incomplete
    max_lines: Incomplete
    max_width: Incomplete

conf: Incomplete

class base_constants_version(ScienceState):
    """
    Base class for the real version-setters below.
    """
    _value: str
    _versions: Incomplete
    @classmethod
    def validate(cls, value): ...
    @classmethod
    def set(cls, value):
        """
        Set the current constants value.
        """

class physical_constants(base_constants_version):
    """
    The version of physical constants to use.
    """
    _value: str
    _versions: Incomplete

class astronomical_constants(base_constants_version):
    """
    The version of astronomical constants to use.
    """
    _value: str
    _versions: Incomplete

test: Incomplete

def _initialize_astropy() -> None: ...
def _get_bibtex(): ...

__citation__: Incomplete

__bibtex__: Incomplete
log: Incomplete

def online_help(query) -> None:
    """
    Search the online Astropy documentation for the given query.
    Opens the results in the default web browser.  Requires an active
    Internet connection.

    Parameters
    ----------
    query : str
        The search query.
    """

__dir_inc__: Incomplete
