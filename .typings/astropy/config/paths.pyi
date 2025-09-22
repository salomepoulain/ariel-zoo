from _typeshed import Incomplete
from pathlib import Path

__all__ = ['get_config_dir', 'get_config_dir_path', 'get_cache_dir', 'get_cache_dir_path', 'set_temp_config', 'set_temp_cache']

def get_config_dir_path(rootname: str = 'astropy') -> Path:
    """
    Determines the package configuration directory name and creates the
    directory if it doesn't exist.

    This directory is typically ``$HOME/.astropy/config``, but if the
    XDG_CONFIG_HOME environment variable is set and the
    ``$XDG_CONFIG_HOME/astropy`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Parameters
    ----------
    rootname : str
        Name of the root configuration directory. For example, if ``rootname =
        'pkgname'``, the configuration directory would be ``<home>/.pkgname/``
        rather than ``<home>/.astropy`` (depending on platform).

    Returns
    -------
    configdir : Path
        The absolute path to the configuration directory.

    """
def get_config_dir(rootname: str = 'astropy') -> str: ...
def get_cache_dir_path(rootname: str = 'astropy') -> Path:
    """
    Determines the Astropy cache directory name and creates the directory if it
    doesn't exist.

    This directory is typically ``$HOME/.astropy/cache``, but if the
    XDG_CACHE_HOME environment variable is set and the
    ``$XDG_CACHE_HOME/astropy`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Parameters
    ----------
    rootname : str
        Name of the root cache directory. For example, if
        ``rootname = 'pkgname'``, the cache directory will be
        ``<cache>/.pkgname/``.

    Returns
    -------
    cachedir : Path
        The absolute path to the cache directory.

    """
def get_cache_dir(rootname: str = 'astropy') -> str: ...

class _SetTempPath:
    _temp_path: Incomplete
    _default_path_getter: Incomplete
    _path: Incomplete
    _delete: Incomplete
    _prev_path: Incomplete
    def __init__(self, path: Incomplete | None = None, delete: bool = False) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
    def __call__(self, func):
        """Implements use as a decorator."""

class set_temp_config(_SetTempPath):
    """
    Context manager to set a temporary path for the Astropy config, primarily
    for use with testing.

    If the path set by this context manager does not already exist it will be
    created, if possible.

    This may also be used as a decorator on a function to set the config path
    just within that function.

    Parameters
    ----------
    path : str, optional
        The directory (which must exist) in which to find the Astropy config
        files, or create them if they do not already exist.  If None, this
        restores the config path to the user's default config path as returned
        by `get_config_dir` as though this context manager were not in effect
        (this is useful for testing).  In this case the ``delete`` argument is
        always ignored.

    delete : bool, optional
        If True, cleans up the temporary directory after exiting the temp
        context (default: False).
    """
    _default_path_getter: Incomplete
    _cfgobjs_copy: Incomplete
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...

class set_temp_cache(_SetTempPath):
    """
    Context manager to set a temporary path for the Astropy download cache,
    primarily for use with testing (though there may be other applications
    for setting a different cache directory, for example to switch to a cache
    dedicated to large files).

    If the path set by this context manager does not already exist it will be
    created, if possible.

    This may also be used as a decorator on a function to set the cache path
    just within that function.

    Parameters
    ----------
    path : str
        The directory (which must exist) in which to find the Astropy cache
        files, or create them if they do not already exist.  If None, this
        restores the cache path to the user's default cache path as returned
        by `get_cache_dir` as though this context manager were not in effect
        (this is useful for testing).  In this case the ``delete`` argument is
        always ignored.

    delete : bool, optional
        If True, cleans up the temporary directory after exiting the temp
        context (default: False).
    """
    _default_path_getter: Incomplete
