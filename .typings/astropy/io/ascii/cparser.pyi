import _cython_3_0_11
import astropy.io.ascii.core as core
from _typeshed import Incomplete
from astropy.utils.data import get_readable_fileobj as get_readable_fileobj
from astropy.utils.exceptions import AstropyWarning as AstropyWarning
from typing import ClassVar

ERR_CODES: dict
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
_copy_cparser: _cython_3_0_11.cython_function_or_method
_read_chunk: _cython_3_0_11.cython_function_or_method
get_fill_values: _cython_3_0_11.cython_function_or_method

class CParser:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    header_chars: Incomplete
    header_start: Incomplete
    source: Incomplete
    width: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _deduplicate_names(self, *args, **kwargs):
        """Ensure there are no duplicates in ``self.header_names``
                Cythonic version of  core._deduplicate_names.
        """
    def get_header_names(self, *args, **kwargs): ...
    def get_names(self, *args, **kwargs): ...
    def read(self, *args, **kwargs): ...
    def read_header(self, *args, **kwargs): ...
    def set_names(self, *args, **kwargs): ...
    def setup_tokenizer(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CParserError(Exception): ...

class FastWriter:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _write_header(self, *args, **kwargs): ...
    def write(self, *args, **kwargs): ...
    def __reduce__(self): ...

class FileString:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def splitlines(self, *args, **kwargs):
        """
        Return a generator yielding lines from the memory map.
        """
    def __getitem__(self, index):
        """Return self[key]."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
