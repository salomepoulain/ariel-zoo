from .util import _array_from_file as _array_from_file, _array_to_file as _array_to_file, _write_string as _write_string, fileobj_closed as fileobj_closed, fileobj_mode as fileobj_mode, fileobj_name as fileobj_name, isfile as isfile, isreadable as isreadable, iswritable as iswritable, path_like as path_like
from _typeshed import Incomplete
from astropy.utils.compat.optional_deps import HAS_BZ2 as HAS_BZ2
from astropy.utils.data import _is_url as _is_url, _requires_fsspec as _requires_fsspec, download_file as download_file, get_readable_fileobj as get_readable_fileobj
from astropy.utils.decorators import classproperty as classproperty
from astropy.utils.exceptions import AstropyUserWarning as AstropyUserWarning
from astropy.utils.misc import NOT_OVERWRITING_MSG as NOT_OVERWRITING_MSG

IO_FITS_MODES: Incomplete
FILE_MODES: Incomplete
TEXT_RE: Incomplete
MEMMAP_MODES: Incomplete
GZIP_MAGIC: bytes
PKZIP_MAGIC: bytes
BZIP2_MAGIC: bytes

def _is_bz2file(fileobj): ...
def _normalize_fits_mode(mode): ...

class _File:
    """
    Represents a FITS file on disk (or in some other file-like object).
    """
    strict_memmap: Incomplete
    _file: Incomplete
    closed: bool
    binary: bool
    mode: Incomplete
    memmap: Incomplete
    compression: Incomplete
    readonly: bool
    writeonly: bool
    close_on_error: bool
    _mmap: Incomplete
    simulateonly: bool
    name: Incomplete
    file_like: bool
    fileobj_mode: Incomplete
    size: int
    def __init__(self, fileobj: Incomplete | None = None, mode: Incomplete | None = None, memmap: Incomplete | None = None, overwrite: bool = False, cache: bool = True, *, use_fsspec: Incomplete | None = None, fsspec_kwargs: Incomplete | None = None, decompress_in_memory: bool = False) -> None: ...
    def __repr__(self) -> str: ...
    def __enter__(self): ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def readable(self): ...
    def read(self, size: Incomplete | None = None): ...
    def readarray(self, size: Incomplete | None = None, offset: int = 0, dtype=..., shape: Incomplete | None = None):
        """
        Similar to file.read(), but returns the contents of the underlying
        file as a numpy array (or mmap'd array if memmap=True) rather than a
        string.

        Usually it's best not to use the `size` argument with this method, but
        it's provided for compatibility.
        """
    def writable(self): ...
    def write(self, string) -> None: ...
    def writearray(self, array) -> None:
        """
        Similar to file.write(), but writes a numpy array instead of a string.

        Also like file.write(), a flush() or close() may be needed before
        the file on disk reflects the data written.
        """
    def flush(self) -> None: ...
    def seek(self, offset, whence: int = 0) -> None: ...
    def tell(self): ...
    def truncate(self, size: Incomplete | None = None) -> None: ...
    def close(self) -> None:
        """
        Close the 'physical' FITS file.
        """
    def _maybe_close_mmap(self, refcount_delta: int = 0) -> None:
        """
        When mmap is in use these objects hold a reference to the mmap of the
        file (so there is only one, shared by all HDUs that reference this
        file).

        This will close the mmap if there are no arrays referencing it.
        """
    def _overwrite_existing(self, overwrite, fileobj, closed) -> None:
        """Overwrite an existing file if ``overwrite`` is ``True``, otherwise
        raise an OSError.  The exact behavior of this method depends on the
        _File object state and is only meant for use within the ``_open_*``
        internal methods.
        """
    def _try_read_compressed(self, obj_or_name, magic, mode, ext: str = ''):
        """Attempt to determine if the given file is compressed."""
    def _open_fileobj(self, fileobj, mode, overwrite) -> None:
        """Open a FITS file from a file object (including compressed files)."""
    def _open_filelike(self, fileobj, mode, overwrite) -> None:
        """Open a FITS file from a file-like object, i.e. one that has
        read and/or write methods.
        """
    def _open_filename(self, filename, mode, overwrite) -> None:
        """Open a FITS file from a filename string."""
    def _mmap_available(cls):
        """Tests that mmap, and specifically mmap.flush works.  This may
        be the case on some uncommon platforms (see
        https://github.com/astropy/astropy/issues/968).

        If mmap.flush is found not to work, ``self.memmap = False`` is
        set and a warning is issued.
        """
    def _open_zipfile(self, fileobj, mode) -> None:
        """Limited support for zipfile.ZipFile objects containing a single
        a file.  Allows reading only for now by extracting the file to a
        tempfile.
        """
