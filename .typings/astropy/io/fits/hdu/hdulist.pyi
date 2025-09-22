from _typeshed import Incomplete
from astropy.io.fits.verify import _Verify

__all__ = ['HDUList', 'fitsopen']

def fitsopen(name, mode: str = 'readonly', memmap: Incomplete | None = None, save_backup: bool = False, cache: bool = True, lazy_load_hdus: Incomplete | None = None, ignore_missing_simple: bool = False, *, use_fsspec: Incomplete | None = None, fsspec_kwargs: Incomplete | None = None, decompress_in_memory: bool = False, **kwargs):
    '''Factory function to open a FITS file and return an `HDUList` object.

    Parameters
    ----------
    name : str, file-like or `pathlib.Path`
        File to be opened.

    mode : str, optional
        Open mode, \'readonly\', \'update\', \'append\', \'denywrite\', or
        \'ostream\'. Default is \'readonly\'.

        If ``name`` is a file object that is already opened, ``mode`` must
        match the mode the file was opened with, readonly (rb), update (rb+),
        append (ab+), ostream (w), denywrite (rb)).

    memmap : bool, optional
        Is memory mapping to be used? This value is obtained from the
        configuration item ``astropy.io.fits.Conf.use_memmap``.
        Default is `True`.

    save_backup : bool, optional
        If the file was opened in update or append mode, this ensures that
        a backup of the original file is saved before any changes are flushed.
        The backup has the same name as the original file with ".bak" appended.
        If "file.bak" already exists then "file.bak.1" is used, and so on.
        Default is `False`.

    cache : bool, optional
        If the file name is a URL, `~astropy.utils.data.download_file` is used
        to open the file.  This specifies whether or not to save the file
        locally in Astropy\'s download cache. Default is `True`.

    lazy_load_hdus : bool, optional
        To avoid reading all the HDUs and headers in a FITS file immediately
        upon opening.  This is an optimization especially useful for large
        files, as FITS has no way of determining the number and offsets of all
        the HDUs in a file without scanning through the file and reading all
        the headers. Default is `True`.

        To disable lazy loading and read all HDUs immediately (the old
        behavior) use ``lazy_load_hdus=False``.  This can lead to fewer
        surprises--for example with lazy loading enabled, ``len(hdul)``
        can be slow, as it means the entire FITS file needs to be read in
        order to determine the number of HDUs.  ``lazy_load_hdus=False``
        ensures that all HDUs have already been loaded after the file has
        been opened.

        .. versionadded:: 1.3

    uint : bool, optional
        Interpret signed integer data where ``BZERO`` is the central value and
        ``BSCALE == 1`` as unsigned integer data.  For example, ``int16`` data
        with ``BZERO = 32768`` and ``BSCALE = 1`` would be treated as
        ``uint16`` data. Default is `True` so that the pseudo-unsigned
        integer convention is assumed.

    ignore_missing_end : bool, optional
        Do not raise an exception when opening a file that is missing an
        ``END`` card in the last header. Default is `False`.

    ignore_missing_simple : bool, optional
        Do not raise an exception when the SIMPLE keyword is missing. Note
        that io.fits will raise a warning if a SIMPLE card is present but
        written in a way that does not follow the FITS Standard.
        Default is `False`.

        .. versionadded:: 4.2

    checksum : bool, str, optional
        If `True`, verifies that both ``DATASUM`` and ``CHECKSUM`` card values
        (when present in the HDU header) match the header and data of all HDU\'s
        in the file.  Updates to a file that already has a checksum will
        preserve and update the existing checksums unless this argument is
        given a value of \'remove\', in which case the CHECKSUM and DATASUM
        values are not checked, and are removed when saving changes to the
        file. Default is `False`.

    disable_image_compression : bool, optional
        If `True`, treats compressed image HDU\'s like normal binary table
        HDU\'s.  Default is `False`.

    do_not_scale_image_data : bool, optional
        If `True`, image data is not scaled using BSCALE/BZERO values
        when read.  Default is `False`.

    character_as_bytes : bool, optional
        Whether to return bytes for string columns, otherwise unicode strings
        are returned, but this does not respect memory mapping and loads the
        whole column in memory when accessed. Default is `False`.

    ignore_blank : bool, optional
        If `True`, the BLANK keyword is ignored if present.
        Default is `False`.

    scale_back : bool, optional
        If `True`, when saving changes to a file that contained scaled image
        data, restore the data to the original type and reapply the original
        BSCALE/BZERO values. This could lead to loss of accuracy if scaling
        back to integer values after performing floating point operations on
        the data. Default is `False`.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``,
        ``"silentfix"``, ``"ignore"``, ``"warn"``, or
        ``"exception"``.  May also be any combination of ``"fix"`` or
        ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
        (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.

    use_fsspec : bool, optional
        Use `fsspec.open` to open the file? Defaults to `False` unless
        ``name`` starts with the Amazon S3 storage prefix ``s3://`` or the
        Google Cloud Storage prefix ``gs://``.  Can also be used for paths
        with other prefixes (e.g., ``http://``) but in this case you must
        explicitly pass ``use_fsspec=True``.
        Use of this feature requires the optional ``fsspec`` package.
        A ``ModuleNotFoundError`` will be raised if the dependency is missing.

        .. versionadded:: 5.2

    fsspec_kwargs : dict, optional
        Keyword arguments passed on to `fsspec.open`. This can be used to
        configure cloud storage credentials and caching behavior.
        For example, pass ``fsspec_kwargs={"anon": True}`` to enable
        anonymous access to Amazon S3 open data buckets.
        See ``fsspec``\'s documentation for available parameters.

        .. versionadded:: 5.2

    decompress_in_memory : bool, optional
        By default files are decompressed progressively depending on what data
        is needed.  This is good for memory usage, avoiding decompression of
        the whole file, but it can be slow. With decompress_in_memory=True it
        is possible to decompress instead the whole file in memory.

        .. versionadded:: 6.0

    Returns
    -------
    hdulist : `HDUList`
        `HDUList` containing all of the header data units in the file.

    '''

class HDUList(list, _Verify):
    """
    HDU list class.  This is the top-level FITS object.  When a FITS
    file is opened, a `HDUList` object is returned.
    """
    _data: Incomplete
    _file: Incomplete
    _open_kwargs: Incomplete
    _in_read_next_hdu: bool
    _read_all: bool
    def __init__(self, hdus=[], file: Incomplete | None = None) -> None:
        """
        Construct a `HDUList` object.

        Parameters
        ----------
        hdus : BaseHDU or sequence thereof, optional
            The HDU object(s) to comprise the `HDUList`.  Should be
            instances of HDU classes like `ImageHDU` or `BinTableHDU`.

        file : file-like, bytes, optional
            The opened physical file associated with the `HDUList`
            or a bytes object containing the contents of the FITS
            file.
        """
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...
    def __getitem__(self, key):
        """
        Get an HDU from the `HDUList`, indexed by number or name.
        """
    def __contains__(self, item) -> bool:
        """
        Returns `True` if ``item`` is an ``HDU`` _in_ ``self`` or a valid
        extension specification (e.g., integer extension number, extension
        name, or a tuple of extension name and an extension version)
        of a ``HDU`` in ``self``.

        """
    _resize: bool
    _truncate: bool
    def __setitem__(self, key, hdu) -> None:
        """
        Set an HDU to the `HDUList`, indexed by number or name.
        """
    def __delitem__(self, key) -> None:
        """
        Delete an HDU from the `HDUList`, indexed by number or name.
        """
    def __enter__(self): ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    @classmethod
    def fromfile(cls, fileobj, mode: Incomplete | None = None, memmap: Incomplete | None = None, save_backup: bool = False, cache: bool = True, lazy_load_hdus: bool = True, ignore_missing_simple: bool = False, **kwargs):
        """
        Creates an `HDUList` instance from a file-like object.

        The actual implementation of ``fitsopen()``, and generally shouldn't
        be used directly.  Use :func:`open` instead (and see its
        documentation for details of the parameters accepted by this method).
        """
    @classmethod
    def fromstring(cls, data, **kwargs):
        """
        Creates an `HDUList` instance from a string or other in-memory data
        buffer containing an entire FITS file.  Similar to
        :meth:`HDUList.fromfile`, but does not accept the mode or memmap
        arguments, as they are only relevant to reading from a file on disk.

        This is useful for interfacing with other libraries such as CFITSIO,
        and may also be useful for streaming applications.

        Parameters
        ----------
        data : str, buffer-like, etc.
            A string or other memory buffer containing an entire FITS file.
            Buffer-like objects include :class:`~bytes`, :class:`~bytearray`,
            :class:`~memoryview`, and :class:`~numpy.ndarray`.
            It should be noted that if that memory is read-only (such as a
            Python string) the returned :class:`HDUList`'s data portions will
            also be read-only.
        **kwargs : dict
            Optional keyword arguments.  See
            :func:`astropy.io.fits.open` for details.

        Returns
        -------
        hdul : HDUList
            An :class:`HDUList` object representing the in-memory FITS file.
        """
    def fileinfo(self, index):
        """
        Returns a dictionary detailing information about the locations
        of the indexed HDU within any associated file.  The values are
        only valid after a read or write of the associated file with
        no intervening changes to the `HDUList`.

        Parameters
        ----------
        index : int
            Index of HDU for which info is to be returned.

        Returns
        -------
        fileinfo : dict or None

            The dictionary details information about the locations of
            the indexed HDU within an associated file.  Returns `None`
            when the HDU is not associated with a file.

            Dictionary contents:

            ========== ========================================================
            Key        Value
            ========== ========================================================
            file       File object associated with the HDU
            filename   Name of associated file object
            filemode   Mode in which the file was opened (readonly,
                       update, append, denywrite, ostream)
            resized    Flag that when `True` indicates that the data has been
                       resized since the last read/write so the returned values
                       may not be valid.
            hdrLoc     Starting byte location of header in file
            datLoc     Starting byte location of data block in file
            datSpan    Data size including padding
            ========== ========================================================

        """
    def __copy__(self):
        """
        Return a shallow copy of an HDUList.

        Returns
        -------
        copy : `HDUList`
            A shallow copy of this `HDUList` object.

        """
    copy = __copy__
    def __deepcopy__(self, memo: Incomplete | None = None): ...
    def pop(self, index: int = -1):
        """Remove an item from the list and return it.

        Parameters
        ----------
        index : int, str, tuple of (string, int), optional
            An integer value of ``index`` indicates the position from which
            ``pop()`` removes and returns an HDU. A string value or a tuple
            of ``(string, int)`` functions as a key for identifying the
            HDU to be removed and returned. If ``key`` is a tuple, it is
            of the form ``(key, ver)`` where ``ver`` is an ``EXTVER``
            value that must match the HDU being searched for.

            If the key is ambiguous (e.g. there are multiple 'SCI' extensions)
            the first match is returned.  For a more precise match use the
            ``(name, ver)`` pair.

            If even the ``(name, ver)`` pair is ambiguous the numeric index
            must be used to index the duplicate HDU.

        Returns
        -------
        hdu : BaseHDU
            The HDU object at position indicated by ``index`` or having name
            and version specified by ``index``.
        """
    def insert(self, index, hdu) -> None:
        """
        Insert an HDU into the `HDUList` at the given ``index``.

        Parameters
        ----------
        index : int
            Index before which to insert the new HDU.

        hdu : BaseHDU
            The HDU object to insert
        """
    def append(self, hdu) -> None:
        """
        Append a new HDU to the `HDUList`.

        Parameters
        ----------
        hdu : BaseHDU
            HDU to add to the `HDUList`.
        """
    def index_of(self, key):
        """
        Get the index of an HDU from the `HDUList`.

        Parameters
        ----------
        key : int, str, tuple of (string, int) or BaseHDU
            The key identifying the HDU.  If ``key`` is a tuple, it is of the
            form ``(name, ver)`` where ``ver`` is an ``EXTVER`` value that must
            match the HDU being searched for.

            If the key is ambiguous (e.g. there are multiple 'SCI' extensions)
            the first match is returned.  For a more precise match use the
            ``(name, ver)`` pair.

            If even the ``(name, ver)`` pair is ambiguous (it shouldn't be
            but it's not impossible) the numeric index must be used to index
            the duplicate HDU.

            When ``key`` is an HDU object, this function returns the
            index of that HDU object in the ``HDUList``.

        Returns
        -------
        index : int
            The index of the HDU in the `HDUList`.

        Raises
        ------
        ValueError
            If ``key`` is an HDU object and it is not found in the ``HDUList``.
        KeyError
            If an HDU specified by the ``key`` that is an extension number,
            extension name, or a tuple of extension name and version is not
            found in the ``HDUList``.

        """
    def _positive_index_of(self, key):
        """
        Same as index_of, but ensures always returning a positive index
        or zero.

        (Really this should be called non_negative_index_of but it felt
        too long.)

        This means that if the key is a negative integer, we have to
        convert it to the corresponding positive index.  This means
        knowing the length of the HDUList, which in turn means loading
        all HDUs.  Therefore using negative indices on HDULists is inherently
        inefficient.
        """
    def readall(self) -> None:
        """
        Read data of all HDUs into memory.
        """
    def flush(self, output_verify: str = 'fix', verbose: bool = False) -> None:
        '''
        Force a write of the `HDUList` back to the file (for append and
        update modes only).

        Parameters
        ----------
        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.

        verbose : bool
            When `True`, print verbose messages
        '''
    def update_extend(self):
        """
        Make sure that if the primary header needs the keyword ``EXTEND`` that
        it has it and it is correct.
        """
    def writeto(self, fileobj, output_verify: str = 'exception', overwrite: bool = False, checksum: bool = False) -> None:
        '''
        Write the `HDUList` to a new file.

        Parameters
        ----------
        fileobj : str, file-like or `pathlib.Path`
            File to write to.  If a file object, must be opened in a
            writeable mode.

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.

        overwrite : bool, optional
            If ``True``, overwrite the output file if it exists. Raises an
            ``OSError`` if ``False`` and the output file exists. Default is
            ``False``.

        checksum : bool
            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the headers of all HDU\'s written to the file.

        Notes
        -----
        gzip, zip and bzip2 compression algorithms are natively supported.
        Compression mode is determined from the filename extension
        (\'.gz\', \'.zip\' or \'.bz2\' respectively).  It is also possible to pass a
        compressed file object, e.g. `gzip.GzipFile`.
        '''
    def close(self, output_verify: str = 'exception', verbose: bool = False, closed: bool = True) -> None:
        '''
        Close the associated FITS file and memmap object, if any.

        Parameters
        ----------
        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.

        verbose : bool
            When `True`, print out verbose messages.

        closed : bool
            When `True`, close the underlying file object.
        '''
    def info(self, output: Incomplete | None = None):
        """
        Summarize the info of the HDUs in this `HDUList`.

        Note that this function prints its results to the console---it
        does not return a value.

        Parameters
        ----------
        output : file-like or bool, optional
            A file-like object to write the output to.  If `False`, does not
            output to a file and instead returns a list of tuples representing
            the HDU info.  Writes to ``sys.stdout`` by default.
        """
    def filename(self):
        """
        Return the file name associated with the HDUList object if one exists.
        Otherwise returns None.

        Returns
        -------
        filename : str
            A string containing the file name associated with the HDUList
            object if an association exists.  Otherwise returns None.

        """
    @classmethod
    def _readfrom(cls, fileobj: Incomplete | None = None, data: Incomplete | None = None, mode: Incomplete | None = None, memmap: Incomplete | None = None, cache: bool = True, lazy_load_hdus: bool = True, ignore_missing_simple: bool = False, *, use_fsspec: Incomplete | None = None, fsspec_kwargs: Incomplete | None = None, decompress_in_memory: bool = False, **kwargs):
        """
        Provides the implementations from HDUList.fromfile and
        HDUList.fromstring, both of which wrap this method, as their
        implementations are largely the same.
        """
    def _try_while_unread_hdus(self, func, *args, **kwargs):
        """
        Attempt an operation that accesses an HDU by index/name
        that can fail if not all HDUs have been read yet.  Keep
        reading HDUs until the operation succeeds or there are no
        more HDUs to read.
        """
    def _read_next_hdu(self):
        """
        Lazily load a single HDU from the fileobj or data string the `HDUList`
        was opened from, unless no further HDUs are found.

        Returns True if a new HDU was loaded, or False otherwise.
        """
    def _verify(self, option: str = 'warn'): ...
    def _flush_update(self) -> None:
        """Implements flushing changes to a file in update mode."""
    def _flush_resize(self) -> None:
        """
        Implements flushing changes in update mode when parts of one or more HDU
        need to be resized.
        """
    def _wasresized(self, verbose: bool = False):
        """
        Determine if any changes to the HDUList will require a file resize
        when flushing the file.

        Side effect of setting the objects _resize attribute.
        """
