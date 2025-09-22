import _cython_3_0_11

__pyx_unpickle__ColumnGetitemShim: _cython_3_0_11.cython_function_or_method
__pyx_unpickle__MaskedColumnGetitemShim: _cython_3_0_11.cython_function_or_method
__test__: dict

class _ColumnGetitemShim:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class _MaskedColumnGetitemShim(_ColumnGetitemShim):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
