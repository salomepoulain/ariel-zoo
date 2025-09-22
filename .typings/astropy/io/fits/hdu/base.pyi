from _typeshed import Incomplete
from astropy.io.fits.verify import _Verify

__all__ = ['DELAYED', 'InvalidHDUException', 'ExtensionHDU', 'NonstandardExtHDU']

class _Delayed: ...

DELAYED: Incomplete

class InvalidHDUException(Exception):
    """
    A custom exception class used mainly to signal to _BaseHDU.__new__ that
    an HDU cannot possibly be considered valid, and must be assumed to be
    corrupted.
    """

class _BaseHDU:
    """Base class for all HDU (header data unit) classes."""
    _hdu_registry: Incomplete
    _standard: bool
    _padding_byte: str
    _default_name: str
    _header: Incomplete
    _header_str: Incomplete
    _file: Incomplete
    _buffer: Incomplete
    _header_offset: Incomplete
    _data_offset: Incomplete
    _data_size: Incomplete
    _data_replaced: bool
    _data_needs_rescale: bool
    _new: bool
    _output_checksum: bool
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None, *args, **kwargs) -> None: ...
    def __init_subclass__(cls, **kwargs): ...
    @property
    def header(self): ...
    @header.setter
    def header(self, value) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, value) -> None: ...
    @property
    def ver(self): ...
    @ver.setter
    def ver(self, value) -> None: ...
    @property
    def level(self): ...
    @level.setter
    def level(self, value) -> None: ...
    @property
    def is_image(self): ...
    @property
    def _data_loaded(self): ...
    @property
    def _has_data(self): ...
    @classmethod
    def register_hdu(cls, hducls) -> None: ...
    @classmethod
    def unregister_hdu(cls, hducls) -> None: ...
    @classmethod
    def match_header(cls, header) -> None: ...
    @classmethod
    def fromstring(cls, data, checksum: bool = False, ignore_missing_end: bool = False, **kwargs):
        """
        Creates a new HDU object of the appropriate type from a string
        containing the HDU's entire header and, optionally, its data.

        Note: When creating a new HDU from a string without a backing file
        object, the data of that HDU may be read-only.  It depends on whether
        the underlying string was an immutable Python str/bytes object, or some
        kind of read-write memory buffer such as a `memoryview`.

        Parameters
        ----------
        data : str, bytes, memoryview, ndarray
            A byte string containing the HDU's header and data.

        checksum : bool, optional
            Check the HDU's checksum and/or datasum.

        ignore_missing_end : bool, optional
            Ignore a missing end card in the header data.  Note that without the
            end card the end of the header may be ambiguous and resulted in a
            corrupt HDU.  In this case the assumption is that the first 2880
            block that does not begin with valid FITS header data is the
            beginning of the data.

        **kwargs : optional
            May consist of additional keyword arguments specific to an HDU
            type--these correspond to keywords recognized by the constructors of
            different HDU classes such as `PrimaryHDU`, `ImageHDU`, or
            `BinTableHDU`.  Any unrecognized keyword arguments are simply
            ignored.
        """
    @classmethod
    def readfrom(cls, fileobj, checksum: bool = False, ignore_missing_end: bool = False, **kwargs):
        """
        Read the HDU from a file.  Normally an HDU should be opened with
        :func:`open` which reads the entire HDU list in a FITS file.  But this
        method is still provided for symmetry with :func:`writeto`.

        Parameters
        ----------
        fileobj : file-like
            Input FITS file.  The file's seek pointer is assumed to be at the
            beginning of the HDU.

        checksum : bool
            If `True`, verifies that both ``DATASUM`` and ``CHECKSUM`` card
            values (when present in the HDU header) match the header and data
            of all HDU's in the file.

        ignore_missing_end : bool
            Do not issue an exception when opening a file that is missing an
            ``END`` card in the last header.
        """
    def writeto(self, name, output_verify: str = 'exception', overwrite: bool = False, checksum: bool = False) -> None:
        '''
        Write the HDU to a new file. This is a convenience method to
        provide a user easier output interface if only one HDU needs
        to be written to a file.

        Parameters
        ----------
        name : path-like or file-like
            Output FITS file.  If the file object is already opened, it must
            be opened in a writeable mode.

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``"+warn"``, or ``"+exception"``
            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.

        overwrite : bool, optional
            If ``True``, overwrite the output file if it exists. Raises an
            ``OSError`` if ``False`` and the output file exists. Default is
            ``False``.

        checksum : bool
            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the header of the HDU when written to the file.

        Notes
        -----
        gzip, zip and bzip2 compression algorithms are natively supported.
        Compression mode is determined from the filename extension
        (\'.gz\', \'.zip\' or \'.bz2\' respectively).  It is also possible to pass a
        compressed file object, e.g. `gzip.GzipFile`.
        '''
    @classmethod
    def _from_data(cls, data, header, **kwargs):
        """
        Instantiate the HDU object after guessing the HDU class from the
        FITS Header.
        """
    @classmethod
    def _readfrom_internal(cls, data, header: Incomplete | None = None, checksum: bool = False, ignore_missing_end: bool = False, **kwargs):
        """
        Provides the bulk of the internal implementation for readfrom and
        fromstring.

        For some special cases, supports using a header that was already
        created, and just using the input data for the actual array data.
        """
    def _get_raw_data(self, shape, code, offset):
        """
        Return raw array from either the HDU's memory buffer or underlying
        file.
        """
    def _prewriteto(self, checksum: bool = False, inplace: bool = False) -> None: ...
    def _update_pseudo_int_scale_keywords(self) -> None:
        """
        If the data is signed int 8, unsigned int 16, 32, or 64,
        add BSCALE/BZERO cards to header.
        """
    def _update_checksum(self, checksum, checksum_keyword: str = 'CHECKSUM', datasum_keyword: str = 'DATASUM') -> None:
        """Update the 'CHECKSUM' and 'DATASUM' keywords in the header (or
        keywords with equivalent semantics given by the ``checksum_keyword``
        and ``datasum_keyword`` arguments--see for example ``CompImageHDU``
        for an example of why this might need to be overridden).
        """
    def _postwriteto(self) -> None: ...
    def _writeheader(self, fileobj): ...
    def _writedata(self, fileobj): ...
    def _writedata_internal(self, fileobj):
        """
        The beginning and end of most _writedata() implementations are the
        same, but the details of writing the data array itself can vary between
        HDU types, so that should be implemented in this method.

        Should return the size in bytes of the data written.
        """
    def _writedata_direct_copy(self, fileobj):
        """Copies the data directly from one file/buffer to the new file.

        For now this is handled by loading the raw data from the existing data
        (including any padding) via a memory map or from an already in-memory
        buffer and using Numpy's existing file-writing facilities to write to
        the new file.

        If this proves too slow a more direct approach may be used.
        """
    def _writeto(self, fileobj, inplace: bool = False, copy: bool = False) -> None: ...
    def _writeto_internal(self, fileobj, inplace, copy) -> None: ...
    def _close(self, closed: bool = True) -> None: ...
_AllHDU = _BaseHDU

class _CorruptedHDU(_BaseHDU):
    """
    A Corrupted HDU class.

    This class is used when one or more mandatory `Card`s are
    corrupted (unparsable), such as the ``BITPIX``, ``NAXIS``, or
    ``END`` cards.  A corrupted HDU usually means that the data size
    cannot be calculated or the ``END`` card is not found.  In the case
    of a missing ``END`` card, the `Header` may also contain the binary
    data

    .. note::
       In future, it may be possible to decipher where the last block
       of the `Header` ends, but this task may be difficult when the
       extension is a `TableHDU` containing ASCII data.
    """
    @property
    def size(self):
        """
        Returns the size (in bytes) of the HDU's data part.
        """
    def _summary(self): ...
    def verify(self) -> None: ...

class _NonstandardHDU(_BaseHDU, _Verify):
    """
    A Non-standard HDU class.

    This class is used for a Primary HDU when the ``SIMPLE`` Card has
    a value of `False`.  A non-standard HDU comes from a file that
    resembles a FITS file but departs from the standards in some
    significant way.  One example would be files where the numbers are
    in the DEC VAX internal storage format rather than the standard
    FITS most significant byte first.  The header for this HDU should
    be valid.  The data for this HDU is read from the file as a byte
    stream that begins at the first byte after the header ``END`` card
    and continues until the end of the file.
    """
    _standard: bool
    @classmethod
    def match_header(cls, header):
        """
        Matches any HDU that has the 'SIMPLE' keyword but is not a standard
        Primary or Groups HDU.
        """
    @property
    def size(self):
        """
        Returns the size (in bytes) of the HDU's data part.
        """
    def _writedata(self, fileobj):
        """
        Differs from the base class :class:`_writedata` in that it doesn't
        automatically add padding, and treats the data as a string of raw bytes
        instead of an array.
        """
    def _summary(self): ...
    def data(self):
        """
        Return the file data.
        """
    def _verify(self, option: str = 'warn'): ...

class _ValidHDU(_BaseHDU, _Verify):
    """
    Base class for all HDUs which are not corrupted.
    """
    _checksum: Incomplete
    _checksum_valid: Incomplete
    _datasum: Incomplete
    _datasum_valid: Incomplete
    name: Incomplete
    ver: Incomplete
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None, name: Incomplete | None = None, ver: Incomplete | None = None, **kwargs) -> None: ...
    @classmethod
    def match_header(cls, header):
        """
        Matches any HDU that is not recognized as having either the SIMPLE or
        XTENSION keyword in its header's first card, but is nonetheless not
        corrupted.

        TODO: Maybe it would make more sense to use _NonstandardHDU in this
        case?  Not sure...
        """
    @property
    def size(self):
        """
        Size (in bytes) of the data portion of the HDU.
        """
    def filebytes(self):
        """
        Calculates and returns the number of bytes that this HDU will write to
        a file.
        """
    def fileinfo(self):
        """
        Returns a dictionary detailing information about the locations
        of this HDU within any associated file.  The values are only
        valid after a read or write of the associated file with no
        intervening changes to the `HDUList`.

        Returns
        -------
        dict or None
            The dictionary details information about the locations of
            this HDU within an associated file.  Returns `None` when
            the HDU is not associated with a file.

            Dictionary contents:

            ========== ================================================
            Key        Value
            ========== ================================================
            file       File object associated with the HDU
            filemode   Mode in which the file was opened (readonly, copyonwrite,
                       update, append, ostream)
            hdrLoc     Starting byte location of header in file
            datLoc     Starting byte location of data block in file
            datSpan    Data size including padding
            ========== ================================================
        """
    def copy(self):
        """
        Make a copy of the HDU, both header and data are copied.
        """
    def _verify(self, option: str = 'warn'): ...
    def req_cards(self, keyword, pos, test, fix_value, option, errlist):
        '''
        Check the existence, location, and value of a required `Card`.

        Parameters
        ----------
        keyword : str
            The keyword to validate

        pos : int, callable
            If an ``int``, this specifies the exact location this card should
            have in the header.  Remember that Python is zero-indexed, so this
            means ``pos=0`` requires the card to be the first card in the
            header.  If given a callable, it should take one argument--the
            actual position of the keyword--and return `True` or `False`.  This
            can be used for custom evaluation.  For example if
            ``pos=lambda idx: idx > 10`` this will check that the keyword\'s
            index is greater than 10.

        test : callable
            This should be a callable (generally a function) that is passed the
            value of the given keyword and returns `True` or `False`.  This can
            be used to validate the value associated with the given keyword.

        fix_value : str, int, float, complex, bool, None
            A valid value for a FITS keyword to use if the given ``test``
            fails to replace an invalid value.  In other words, this provides
            a default value to use as a replacement if the keyword\'s current
            value is invalid.  If `None`, there is no replacement value and the
            keyword is unfixable.

        option : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.

        errlist : list
            A list of validation errors already found in the FITS file; this is
            used primarily for the validation system to collect errors across
            multiple HDUs and multiple calls to `req_cards`.

        Notes
        -----
        If ``pos=None``, the card can be anywhere in the header.  If the card
        does not exist, the new card will have the ``fix_value`` as its value
        when created.  Also check the card\'s value by using the ``test``
        argument.
        '''
    def add_datasum(self, when: Incomplete | None = None, datasum_keyword: str = 'DATASUM'):
        """
        Add the ``DATASUM`` card to this HDU with the value set to the
        checksum calculated for the data.

        Parameters
        ----------
        when : str, optional
            Comment string for the card that by default represents the
            time when the checksum was calculated

        datasum_keyword : str, optional
            The name of the header keyword to store the datasum value in;
            this is typically 'DATASUM' per convention, but there exist
            use cases in which a different keyword should be used

        Returns
        -------
        checksum : int
            The calculated datasum

        Notes
        -----
        For testing purposes, provide a ``when`` argument to enable the comment
        value in the card to remain consistent.  This will enable the
        generation of a ``CHECKSUM`` card with a consistent value.
        """
    def add_checksum(self, when: Incomplete | None = None, override_datasum: bool = False, checksum_keyword: str = 'CHECKSUM', datasum_keyword: str = 'DATASUM') -> None:
        """
        Add the ``CHECKSUM`` and ``DATASUM`` cards to this HDU with
        the values set to the checksum calculated for the HDU and the
        data respectively.  The addition of the ``DATASUM`` card may
        be overridden.

        Parameters
        ----------
        when : str, optional
            comment string for the cards; by default the comments
            will represent the time when the checksum was calculated
        override_datasum : bool, optional
            add the ``CHECKSUM`` card only
        checksum_keyword : str, optional
            The name of the header keyword to store the checksum value in; this
            is typically 'CHECKSUM' per convention, but there exist use cases
            in which a different keyword should be used

        datasum_keyword : str, optional
            See ``checksum_keyword``

        Notes
        -----
        For testing purposes, first call `add_datasum` with a ``when``
        argument, then call `add_checksum` with a ``when`` argument and
        ``override_datasum`` set to `True`.  This will provide consistent
        comments for both cards and enable the generation of a ``CHECKSUM``
        card with a consistent value.
        """
    def verify_datasum(self):
        """
        Verify that the value in the ``DATASUM`` keyword matches the value
        calculated for the ``DATASUM`` of the current HDU data.

        Returns
        -------
        valid : int
            - 0 - failure
            - 1 - success
            - 2 - no ``DATASUM`` keyword present
        """
    def verify_checksum(self):
        """
        Verify that the value in the ``CHECKSUM`` keyword matches the
        value calculated for the current HDU CHECKSUM.

        Returns
        -------
        valid : int
            - 0 - failure
            - 1 - success
            - 2 - no ``CHECKSUM`` keyword present
        """
    def _verify_checksum_datasum(self) -> None:
        """
        Verify the checksum/datasum values if the cards exist in the header.
        Simply displays warnings if either the checksum or datasum don't match.
        """
    def _get_timestamp(self):
        """
        Return the current timestamp in ISO 8601 format, with microseconds
        stripped off.

        Ex.: 2007-05-30T19:05:11
        """
    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
    def _calculate_checksum(self, datasum, checksum_keyword: str = 'CHECKSUM'):
        """
        Calculate the value of the ``CHECKSUM`` card in the HDU.
        """
    def _compute_checksum(self, data, sum32: int = 0):
        """
        Compute the ones-complement checksum of a sequence of bytes.

        Parameters
        ----------
        data
            a memory region to checksum

        sum32
            incremental checksum value from another region

        Returns
        -------
        ones complement checksum
        """
    _MASK: Incomplete
    _EXCLUDE: Incomplete
    def _encode_byte(self, byte):
        """
        Encode a single byte.
        """
    def _char_encode(self, value):
        """
        Encodes the checksum ``value`` using the algorithm described
        in SPR section A.7.2 and returns it as a 16 character string.

        Parameters
        ----------
        value
            a checksum

        Returns
        -------
        ascii encoded checksum
        """

class ExtensionHDU(_ValidHDU):
    """
    An extension HDU class.

    This class is the base class for the `TableHDU`, `ImageHDU`, and
    `BinTableHDU` classes.
    """
    _extension: str
    @classmethod
    def match_header(cls, header) -> None:
        """
        This class should never be instantiated directly.  Either a standard
        extension HDU type should be used for a specific extension, or
        NonstandardExtHDU should be used.
        """
    def writeto(self, name, output_verify: str = 'exception', overwrite: bool = False, checksum: bool = False) -> None:
        """
        Works similarly to the normal writeto(), but prepends a default
        `PrimaryHDU` are required by extension HDUs (which cannot stand on
        their own).
        """
    def _verify(self, option: str = 'warn'): ...

class NonstandardExtHDU(ExtensionHDU):
    """
    A Non-standard Extension HDU class.

    This class is used for an Extension HDU when the ``XTENSION``
    `Card` has a non-standard value.  In this case, Astropy can figure
    out how big the data is but not what it is.  The data for this HDU
    is read from the file as a byte stream that begins at the first
    byte after the header ``END`` card and continues until the
    beginning of the next header or the end of the file.
    """
    _standard: bool
    @classmethod
    def match_header(cls, header):
        """
        Matches any extension HDU that is not one of the standard extension HDU
        types.
        """
    def _summary(self): ...
    def data(self):
        """
        Return the file data.
        """
