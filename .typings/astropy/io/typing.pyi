from typing import Protocol, TypeAlias, TypeVar

__all__ = ['PathLike', 'ReadableFileLike', 'WriteableFileLike']

_T_co = TypeVar('_T_co', covariant=True)
_T_contra = TypeVar('_T_contra', contravariant=True)
PathLike: TypeAlias

class ReadableFileLike(Protocol[_T_co]):
    """A file-like object that supports reading with a method ``read``.

    This is a :class:`~typing.Protocol` that can be used to annotate file-like
    objects. It is also runtime-checkable and can be used with :func:`isinstance`.
    See :func:`~typing.runtime_checkable` for more information about how runtime
    checking with Protocols works.
    """
    def read(self) -> _T_co: ...

class WriteableFileLike(Protocol[_T_contra]):
    """A file-like object that supports writing with a method ``write``.

    This is a :class:`~typing.Protocol` that can be used to annotate file-like
    objects. It is also runtime-checkable and can be used with :func:`isinstance`.
    See :func:`~typing.runtime_checkable` for more information about how runtime
    checking with Protocols works.
    """
    def write(self, data: _T_contra) -> None: ...
