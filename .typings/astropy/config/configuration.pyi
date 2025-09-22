from _typeshed import Incomplete
from astropy.utils.exceptions import AstropyWarning
from collections.abc import Generator

__all__ = ['InvalidConfigurationItemWarning', 'get_config', 'reload_config', 'ConfigNamespace', 'ConfigItem', 'generate_config', 'create_config_file']

class InvalidConfigurationItemWarning(AstropyWarning):
    """A Warning that is issued when the configuration value specified in the
    astropy configuration file does not match the type expected for that
    configuration value.
    """
class ConfigurationDefaultMissingError(ValueError):
    """An exception that is raised when the configuration defaults (which
    should be generated at build-time) are missing.
    """
class ConfigurationDefaultMissingWarning(AstropyWarning):
    """A warning that is issued when the configuration defaults (which
    should be generated at build-time) are missing.
    """
class ConfigurationChangedWarning(AstropyWarning):
    """
    A warning that the configuration options have changed.
    """

class _ConfigNamespaceMeta(type):
    def __init__(cls, name, bases, dict) -> None: ...

class ConfigNamespace(metaclass=_ConfigNamespaceMeta):
    """
    A namespace of configuration items.  Each subpackage with
    configuration items should define a subclass of this class,
    containing `ConfigItem` instances as members.

    For example::

        class Conf(_config.ConfigNamespace):
            unicode_output = _config.ConfigItem(
                False,
                'Use Unicode characters when outputting values, ...')
            use_color = _config.ConfigItem(
                sys.platform != 'win32',
                'When True, use ANSI color escape sequences when ...',
                aliases=['astropy.utils.console.USE_COLOR'])
        conf = Conf()
    """
    def __iter__(self): ...
    def __str__(self) -> str: ...
    keys = __iter__
    def values(self) -> Generator[Incomplete]:
        """Iterate over configuration item values."""
    def items(self) -> Generator[Incomplete]:
        """Iterate over configuration item ``(name, value)`` pairs."""
    def help(self, name: Incomplete | None = None) -> None:
        '''Print info about configuration items.

        Parameters
        ----------
        name : `str`, optional
            Name of the configuration item to be described. If no name is
            provided then info about all the configuration items will be
            printed.

        Examples
        --------
        >>> from astropy import conf
        >>> conf.help("unicode_output")
        ConfigItem: unicode_output
          cfgtype=\'boolean\'
          defaultvalue=False
          description=\'When True, use Unicode characters when outputting values, and displaying widgets at the console.\'
          module=astropy
          value=False
        '''
    def set_temp(self, attr, value):
        """
        Temporarily set a configuration value.

        Parameters
        ----------
        attr : str
            Configuration item name

        value : object
            The value to set temporarily.

        Examples
        --------
        >>> import astropy
        >>> with astropy.conf.set_temp('use_color', False):
        ...     pass
        ...     # console output will not contain color
        >>> # console output contains color again...
        """
    def reload(self, attr: Incomplete | None = None):
        """
        Reload a configuration item from the configuration file.

        Parameters
        ----------
        attr : str, optional
            The name of the configuration parameter to reload.  If not
            provided, reload all configuration parameters.
        """
    def reset(self, attr: Incomplete | None = None) -> None:
        """
        Reset a configuration item to its default.

        Parameters
        ----------
        attr : str, optional
            The name of the configuration parameter to reload.  If not
            provided, reset all configuration parameters.
        """

class ConfigItem:
    """
    A setting and associated value stored in a configuration file.

    These objects should be created as members of
    `ConfigNamespace` subclasses, for example::

        class _Conf(config.ConfigNamespace):
            unicode_output = config.ConfigItem(
                False,
                'Use Unicode characters when outputting values, and writing widgets '
                'to the console.')
        conf = _Conf()

    Parameters
    ----------
    defaultvalue : object, optional
        The default value for this item. If this is a list of strings, this
        item will be interpreted as an 'options' value - this item must be one
        of those values, and the first in the list will be taken as the default
        value.

    description : str or None, optional
        A description of this item (will be shown as a comment in the
        configuration file)

    cfgtype : str or None, optional
        A type specifier like those used as the *values* of a particular key
        in a ``configspec`` file of ``configobj``. If None, the type will be
        inferred from the default value.

    module : str or None, optional
        The full module name that this item is associated with. The first
        element (e.g. 'astropy' if this is 'astropy.config.configuration')
        will be used to determine the name of the configuration file, while
        the remaining items determine the section. If None, the package will be
        inferred from the package within which this object's initializer is
        called.

    aliases : str, or list of str, optional
        The deprecated location(s) of this configuration item.  If the
        config item is not found at the new location, it will be
        searched for at all of the old locations.

    Raises
    ------
    RuntimeError
        If ``module`` is `None`, but the module this item is created from
        cannot be determined.
    """
    _validator: Incomplete
    cfgtype: Incomplete
    rootname: str
    module: Incomplete
    description: Incomplete
    __doc__: Incomplete
    defaultvalue: Incomplete
    aliases: Incomplete
    def __init__(self, defaultvalue: str = '', description: Incomplete | None = None, cfgtype: Incomplete | None = None, module: Incomplete | None = None, aliases: Incomplete | None = None) -> None: ...
    def __set__(self, obj, value): ...
    def __get__(self, obj, objtype: Incomplete | None = None): ...
    def set(self, value) -> None:
        """
        Sets the current value of this ``ConfigItem``.

        This also updates the comments that give the description and type
        information.

        Parameters
        ----------
        value
            The value this item should be set to.

        Raises
        ------
        TypeError
            If the provided ``value`` is not valid for this ``ConfigItem``.
        """
    def set_temp(self, value) -> Generator[None]:
        """
        Sets this item to a specified value only inside a with block.

        Use as::

            ITEM = ConfigItem('ITEM', 'default', 'description')

            with ITEM.set_temp('newval'):
                #... do something that wants ITEM's value to be 'newval' ...
                print(ITEM)

            # ITEM is now 'default' after the with block

        Parameters
        ----------
        value
            The value to set this item to inside the with block.

        """
    def reload(self):
        """Reloads the value of this ``ConfigItem`` from the relevant
        configuration file.

        Returns
        -------
        val : object
            The new value loaded from the configuration file.

        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __call__(self):
        """Returns the value of this ``ConfigItem``.

        Returns
        -------
        val : object
            This item's value, with a type determined by the ``cfgtype``
            attribute.

        Raises
        ------
        TypeError
            If the configuration value as stored is not this item's type.

        """
    def _validate_val(self, val):
        """Validates the provided value based on cfgtype and returns the
        type-cast value.

        throws the underlying configobj exception if it fails
        """

def get_config(packageormod: Incomplete | None = None, reload: bool = False, rootname: Incomplete | None = None):
    """Gets the configuration object or section associated with a particular
    package or module.

    Parameters
    ----------
    packageormod : str or None
        The package for which to retrieve the configuration object. If a
        string, it must be a valid package name, or if ``None``, the package from
        which this function is called will be used.

    reload : bool, optional
        Reload the file, even if we have it cached.

    rootname : str or None
        Name of the root configuration directory. If ``None`` and
        ``packageormod`` is ``None``, this defaults to be the name of
        the package from which this function is called. If ``None`` and
        ``packageormod`` is not ``None``, this defaults to ``astropy``.

    Returns
    -------
    cfgobj : ``configobj.ConfigObj`` or ``configobj.Section``
        If the requested package is a base package, this will be the
        ``configobj.ConfigObj`` for that package, or if it is a subpackage or
        module, it will return the relevant ``configobj.Section`` object.

    Raises
    ------
    RuntimeError
        If ``packageormod`` is `None`, but the package this item is created
        from cannot be determined.
    """
def generate_config(pkgname: str = 'astropy', filename: Incomplete | None = None, verbose: bool = False):
    """Generates a configuration file, from the list of `ConfigItem`
    objects for each subpackage.

    .. versionadded:: 4.1

    Parameters
    ----------
    pkgname : str or None
        The package for which to retrieve the configuration object.
    filename : str or file-like or None
        If None, the default configuration path is taken from `get_config`.

    """
def reload_config(packageormod: Incomplete | None = None, rootname: Incomplete | None = None) -> None:
    """Reloads configuration settings from a configuration file for the root
    package of the requested package/module.

    This overwrites any changes that may have been made in `ConfigItem`
    objects.  This applies for any items that are based on this file, which
    is determined by the *root* package of ``packageormod``
    (e.g. ``'astropy.cfg'`` for the ``'astropy.config.configuration'``
    module).

    Parameters
    ----------
    packageormod : str or None
        The package or module name - see `get_config` for details.
    rootname : str or None
        Name of the root configuration directory - see `get_config`
        for details.
    """
def create_config_file(pkg, rootname: str = 'astropy', overwrite: bool = False):
    """
    Create the default configuration file for the specified package.
    If the file already exists, it is updated only if it has not been
    modified.  Otherwise the ``overwrite`` flag is needed to overwrite it.

    Parameters
    ----------
    pkg : str
        The package to be updated.
    rootname : str
        Name of the root configuration directory.
    overwrite : bool
        Force updating the file if it already exists.

    Returns
    -------
    updated : bool
        If the profile was updated, `True`, otherwise `False`.

    """
