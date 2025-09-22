from .base import ExtensionHDU, _ValidHDU
from _typeshed import Incomplete

__all__ = ['Section', 'PrimaryHDU', 'ImageHDU']

class _ImageBaseHDU(_ValidHDU):
    """FITS image HDU base class.

    Attributes
    ----------
    header
        image header

    data
        image data
    """
    standard_keyword_comments: Incomplete
    _header: Incomplete
    _do_not_scale_image_data: Incomplete
    _uint: Incomplete
    _scale_back: Incomplete
    _bzero: Incomplete
    _bscale: Incomplete
    _axes: Incomplete
    _bitpix: Incomplete
    _gcount: Incomplete
    _pcount: Incomplete
    _blank: Incomplete
    _orig_bitpix: Incomplete
    _orig_blank: Incomplete
    _orig_bzero: Incomplete
    _orig_bscale: Incomplete
    name: Incomplete
    ver: Incomplete
    _modified: bool
    _data_needs_rescale: bool
    data: Incomplete
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None, do_not_scale_image_data: bool = False, uint: bool = True, scale_back: bool = False, ignore_blank: bool = False, **kwargs) -> None: ...
    @classmethod
    def match_header(cls, header) -> None:
        """
        _ImageBaseHDU is sort of an abstract class for HDUs containing image
        data (as opposed to table data) and should never be used directly.
        """
    @property
    def is_image(self): ...
    @property
    def section(self):
        """
        Access a section of the image array without loading the entire array
        into memory.  The :class:`Section` object returned by this attribute is
        not meant to be used directly by itself.  Rather, slices of the section
        return the appropriate slice of the data, and loads *only* that section
        into memory.

        Sections are useful for retrieving a small subset of data from a remote
        file that has been opened with the ``use_fsspec=True`` parameter.
        For example, you can use this feature to download a small cutout from
        a large FITS image hosted in the Amazon S3 cloud (see the
        :ref:`astropy:fits-cloud-files` section of the Astropy
        documentation for more details.)

        For local files, sections are mostly obsoleted by memmap support, but
        should still be used to deal with very large scaled images.

        Note that sections cannot currently be written to.  Moreover, any
        in-memory updates to the image's ``.data`` property may not be
        reflected in the slices obtained via ``.section``. See the
        :ref:`astropy:data-sections` section of the documentation for
        more details.
        """
    @property
    def shape(self):
        """
        Shape of the image array--should be equivalent to ``self.data.shape``.
        """
    @property
    def header(self): ...
    @header.setter
    def header(self, header) -> None: ...
    @property
    def _data_shape(self): ...
    def update_header(self) -> None:
        """
        Update the header keywords to agree with the data.
        """
    def _update_header_scale_info(self, dtype: Incomplete | None = None) -> None:
        """
        Delete BSCALE/BZERO from header if necessary.
        """
    def scale(self, type: Incomplete | None = None, option: str = 'old', bscale: Incomplete | None = None, bzero: Incomplete | None = None) -> None:
        '''
        Scale image data by using ``BSCALE``/``BZERO``.

        Call to this method will scale `data` and update the keywords of
        ``BSCALE`` and ``BZERO`` in the HDU\'s header.  This method should only
        be used right before writing to the output file, as the data will be
        scaled and is therefore not very usable after the call.

        Parameters
        ----------
        type : str, optional
            destination data type, use a string representing a numpy
            dtype name, (e.g. ``\'uint8\'``, ``\'int16\'``, ``\'float32\'``
            etc.).  If is `None`, use the current data type.

        option : str, optional
            How to scale the data: ``"old"`` uses the original ``BSCALE`` and
            ``BZERO`` values from when the data was read/created (defaulting to
            1 and 0 if they don\'t exist). For integer data only, ``"minmax"``
            uses the minimum and maximum of the data to scale. User-specified
            ``bscale``/``bzero`` values always take precedence.

        bscale, bzero : int, optional
            User-specified ``BSCALE`` and ``BZERO`` values
        '''
    def _scale_internal(self, type: Incomplete | None = None, option: str = 'old', bscale: Incomplete | None = None, bzero: Incomplete | None = None, blank: int = 0) -> None:
        """
        This is an internal implementation of the `scale` method, which
        also supports handling BLANK properly.

        TODO: This is only needed for fixing #3865 without introducing any
        public API changes.  We should support BLANK better when rescaling
        data, and when that is added the need for this internal interface
        should go away.

        Note: the default of ``blank=0`` merely reflects the current behavior,
        and is not necessarily a deliberate choice (better would be to disallow
        conversion of floats to ints without specifying a BLANK if there are
        NaN/inf values).
        """
    def _verify(self, option: str = 'warn'): ...
    def _verify_blank(self) -> None: ...
    def _prewriteto(self, checksum: bool = False, inplace: bool = False): ...
    def _writedata_internal(self, fileobj): ...
    def _writeinternal_dask(self, fileobj): ...
    def _dtype_for_bitpix(self):
        """
        Determine the dtype that the data should be converted to depending on
        the BITPIX value in the header, and possibly on the BSCALE value as
        well.  Returns None if there should not be any change.
        """
    def _convert_pseudo_integer(self, data):
        '''
        Handle "pseudo-unsigned" integers, if the user requested it.  Returns
        the converted data array if so; otherwise returns None.

        In this case case, we don\'t need to handle BLANK to convert it to NAN,
        since we can\'t do NaNs with integers, anyway, i.e. the user is
        responsible for managing blanks.
        '''
    def _get_scaled_image_data(self, offset, shape):
        """
        Internal function for reading image data from a file and apply scale
        factors to it.  Normally this is used for the entire image, but it
        supports alternate offset/shape for Section support.
        """
    def _scale_data(self, raw_data): ...
    def _summary(self):
        """
        Summarize the HDU: name, dimensions, and formats.
        """
    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """

class Section:
    """
    Class enabling subsets of ImageHDU data to be loaded lazily via slicing.

    Slices of this object load the corresponding section of an image array from
    the underlying FITS file, and applies any BSCALE/BZERO factors.

    Section slices cannot be assigned to, and modifications to a section are
    not saved back to the underlying file.

    See the :ref:`astropy:data-sections` section of the Astropy documentation
    for more details.
    """
    hdu: Incomplete
    def __init__(self, hdu) -> None: ...
    @property
    def shape(self): ...
    def __getitem__(self, key):
        """Returns a slice of HDU data specified by `key`.

        If the image HDU is backed by a file handle, this method will only read
        the chunks of the file needed to extract `key`, which is useful in
        situations where the file is located on a slow or remote file system
        (e.g., cloud storage).
        """
    def _getdata(self, keys): ...

class PrimaryHDU(_ImageBaseHDU):
    """
    FITS primary HDU class.
    """
    _default_name: str
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None, do_not_scale_image_data: bool = False, ignore_blank: bool = False, uint: bool = True, scale_back: Incomplete | None = None) -> None:
        """
        Construct a primary HDU.

        Parameters
        ----------
        data : array or ``astropy.io.fits.hdu.base.DELAYED``, optional
            The data in the HDU.

        header : `~astropy.io.fits.Header`, optional
            The header to be used (as a template).  If ``header`` is `None`, a
            minimal header will be provided.

        do_not_scale_image_data : bool, optional
            If `True`, image data is not scaled using BSCALE/BZERO values
            when read. (default: False)

        ignore_blank : bool, optional
            If `True`, the BLANK header keyword will be ignored if present.
            Otherwise, pixels equal to this value will be replaced with
            NaNs. (default: False)

        uint : bool, optional
            Interpret signed integer data where ``BZERO`` is the
            central value and ``BSCALE == 1`` as unsigned integer
            data.  For example, ``int16`` data with ``BZERO = 32768``
            and ``BSCALE = 1`` would be treated as ``uint16`` data.
            (default: True)

        scale_back : bool, optional
            If `True`, when saving changes to a file that contained scaled
            image data, restore the data to the original type and reapply the
            original BSCALE/BZERO values.  This could lead to loss of accuracy
            if scaling back to integer values after performing floating point
            operations on the data.  Pseudo-unsigned integers are automatically
            rescaled unless scale_back is explicitly set to `False`.
            (default: None)
        """
    @classmethod
    def match_header(cls, header): ...
    def update_header(self) -> None: ...
    def _verify(self, option: str = 'warn'): ...

class ImageHDU(_ImageBaseHDU, ExtensionHDU):
    """
    FITS image extension HDU class.
    """
    _extension: str
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None, name: Incomplete | None = None, do_not_scale_image_data: bool = False, uint: bool = True, scale_back: Incomplete | None = None, ver: Incomplete | None = None) -> None:
        """
        Construct an image HDU.

        Parameters
        ----------
        data : array
            The data in the HDU.

        header : `~astropy.io.fits.Header`
            The header to be used (as a template).  If ``header`` is
            `None`, a minimal header will be provided.

        name : str, optional
            The name of the HDU, will be the value of the keyword
            ``EXTNAME``.

        do_not_scale_image_data : bool, optional
            If `True`, image data is not scaled using BSCALE/BZERO values
            when read. (default: False)

        uint : bool, optional
            Interpret signed integer data where ``BZERO`` is the
            central value and ``BSCALE == 1`` as unsigned integer
            data.  For example, ``int16`` data with ``BZERO = 32768``
            and ``BSCALE = 1`` would be treated as ``uint16`` data.
            (default: True)

        scale_back : bool, optional
            If `True`, when saving changes to a file that contained scaled
            image data, restore the data to the original type and reapply the
            original BSCALE/BZERO values.  This could lead to loss of accuracy
            if scaling back to integer values after performing floating point
            operations on the data.  Pseudo-unsigned integers are automatically
            rescaled unless scale_back is explicitly set to `False`.
            (default: None)

        ver : int > 0 or None, optional
            The ver of the HDU, will be the value of the keyword ``EXTVER``.
            If not given or None, it defaults to the value of the ``EXTVER``
            card of the ``header`` or 1.
            (default: None)
        """
    @classmethod
    def match_header(cls, header): ...
    def _verify(self, option: str = 'warn'):
        """
        ImageHDU verify method.
        """

class _IndexInfo:
    npts: int
    offset: Incomplete
    contiguous: bool
    def __init__(self, indx, naxis) -> None: ...
