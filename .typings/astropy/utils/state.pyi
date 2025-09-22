from _typeshed import Incomplete

__all__ = ['ScienceState']

class _ScienceStateContext:
    _value: Incomplete
    _parent: Incomplete
    def __init__(self, parent, value) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, tb: types.TracebackType | None) -> None: ...
    def __repr__(self) -> str: ...

class ScienceState:
    '''
    Science state subclasses are used to manage global items that can
    affect science results.  Subclasses will generally override
    `validate` to convert from any of the acceptable inputs (such as
    strings) to the appropriate internal objects, and set an initial
    value to the ``_value`` member so it has a default.

    Examples
    --------
    ::

        class MyState(ScienceState):
            @classmethod
            def validate(cls, value):
                if value not in (\'A\', \'B\', \'C\'):
                    raise ValueError("Must be one of A, B, C")
                return value
    '''
    def __init__(self) -> None: ...
    @classmethod
    def get(cls):
        """
        Get the current science state value.
        """
    @classmethod
    def set(cls, value):
        """Set the current science state value."""
    @classmethod
    def validate(cls, value):
        """
        Validate the value and convert it to its native type, if
        necessary.
        """
