from _typeshed import Incomplete

__all__ = ['MetaData', 'MetaAttribute']

class MetaData:
    '''
    A descriptor for classes that have a ``meta`` property.

    This can be set to any valid :class:`~collections.abc.Mapping`.

    Parameters
    ----------
    doc : `str`, optional
        Documentation for the attribute of the class.
        Default is ``""``.

        .. versionadded:: 1.2

    copy : `bool`, optional
        If ``True`` the value is deepcopied before setting, otherwise it
        is saved as reference.
        Default is ``True``.

        .. versionadded:: 1.2

    default_factory : Callable[[], Mapping], optional keyword-only
        The factory to use to create the default value of the ``meta``
        attribute.  This must be a callable that returns a `Mapping` object.
        Default is `OrderedDict`, creating an empty `OrderedDict`.

        .. versionadded:: 6.0

    Examples
    --------
    ``MetaData`` can be used as a descriptor to define a ``meta`` attribute`.

        >>> class Foo:
        ...     meta = MetaData()
        ...     def __init__(self, meta=None):
        ...         self.meta = meta

    ``Foo`` can be instantiated with a ``meta`` argument.

        >>> foo = Foo(meta={\'a\': 1, \'b\': 2})
        >>> foo.meta
        {\'a\': 1, \'b\': 2}

    The default value of ``meta`` is an empty :class:`~collections.OrderedDict`.
    This can be set by passing ``None`` to the ``meta`` argument.

        >>> foo = Foo()
        >>> foo.meta
        OrderedDict()

    If an :class:`~collections.OrderedDict` is not a good default metadata type then
    the ``default_factory`` keyword can be used to set the default to a different
    `Mapping` type, when the class is defined.\'

        >>> class Bar:
        ...     meta = MetaData(default_factory=dict)
        ...     def __init__(self, meta=None):
        ...         self.meta = meta

        >>> Bar().meta
        {}

    When accessed from the class ``.meta`` returns `None` since metadata is
    on the class\' instances, not the class itself.

        >>> print(Foo.meta)
        None
    '''
    __doc__: Incomplete
    copy: Incomplete
    _default_factory: Incomplete
    def __init__(self, doc: str = '', copy: bool = True, *, default_factory=...) -> None: ...
    @property
    def default_factory(self): ...
    def __get__(self, instance, owner): ...
    def __set__(self, instance, value) -> None: ...

class MetaAttribute:
    '''
    Descriptor to define custom attribute which gets stored in the object
    ``meta`` dict and can have a defined default.

    This descriptor is intended to provide a convenient way to add attributes
    to a subclass of a complex class such as ``Table`` or ``NDData``.

    This requires that the object has an attribute ``meta`` which is a
    dict-like object.  The value of the MetaAttribute will be stored in a
    new dict meta[\'__attributes__\'] that is created when required.

    Classes that define MetaAttributes are encouraged to support initializing
    the attributes via the class ``__init__``.  For example::

        for attr in list(kwargs):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                setattr(self, attr, kwargs.pop(attr))

    The name of a ``MetaAttribute`` cannot be the same as any of the following:

    - Keyword argument in the owner class ``__init__``
    - Method or attribute of the "parent class", where the parent class is
      taken to be ``owner.__mro__[1]``.

    Parameters
    ----------
    default : Any, optional
        Default value for the attribute, by default `None`.
    '''
    default: Incomplete
    def __init__(self, default: Incomplete | None = None) -> None: ...
    def __get__(self, instance, owner): ...
    def __set__(self, instance, value) -> None: ...
    def __delete__(self, instance) -> None: ...
    name: Incomplete
    def __set_name__(self, owner, name) -> None: ...
    def __repr__(self) -> str: ...
