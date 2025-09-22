from .compat import NDDataArray
from _typeshed import Incomplete

__all__ = ['CCDData', 'fits_ccddata_reader', 'fits_ccddata_writer']

class CCDData(NDDataArray):
    '''A class describing basic CCD data.

    The CCDData class is based on the NDData object and includes a data array,
    uncertainty frame, mask frame, flag frame, meta data, units, and WCS
    information for a single CCD image.

    Parameters
    ----------
    data : `~astropy.nddata.CCDData`-like or array-like
        The actual data contained in this `~astropy.nddata.CCDData` object.
        Note that the data will always be saved by *reference*, so you should
        make a copy of the ``data`` before passing it in if that\'s the desired
        behavior.

    uncertainty : `~astropy.nddata.StdDevUncertainty`,             `~astropy.nddata.VarianceUncertainty`,             `~astropy.nddata.InverseVariance`, `numpy.ndarray` or             None, optional
        Uncertainties on the data. If the uncertainty is a `numpy.ndarray`, it
        it assumed to be, and stored as, a `~astropy.nddata.StdDevUncertainty`.
        Default is ``None``.

    mask : `numpy.ndarray` or None, optional
        Mask for the data, given as a boolean Numpy array with a shape
        matching that of the data. The values must be `False` where
        the data is *valid* and `True` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.
        Default is ``None``.

    flags : `numpy.ndarray` or `~astropy.nddata.FlagCollection` or None,             optional
        Flags giving information about each pixel. These can be specified
        either as a Numpy array of any type with a shape matching that of the
        data, or as a `~astropy.nddata.FlagCollection` instance which has a
        shape matching that of the data.
        Default is ``None``.

    wcs : `~astropy.wcs.WCS` or None, optional
        WCS-object containing the world coordinate system for the data.
        Default is ``None``.

    meta : dict-like object or None, optional
        Metadata for this object. "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object, e.g. creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.

    unit : `~astropy.units.Unit` or str, optional
        The units of the data.
        Default is ``None``.

        .. warning::

            If the unit is ``None`` or not otherwise specified it will raise a
            ``ValueError``

    psf : `numpy.ndarray` or None, optional
        Image representation of the PSF at the center of this image. In order
        for convolution to be flux-preserving, this should generally be
        normalized to sum to unity.

    Raises
    ------
    ValueError
        If the ``uncertainty`` or ``mask`` inputs cannot be broadcast (e.g.,
        match shape) onto ``data``.

    Methods
    -------
    read(\\*args, \\**kwargs)
        ``Classmethod`` to create an CCDData instance based on a ``FITS`` file.
        This method uses :func:`fits_ccddata_reader` with the provided
        parameters.
    write(\\*args, \\**kwargs)
        Writes the contents of the CCDData instance into a new ``FITS`` file.
        This method uses :func:`fits_ccddata_writer` with the provided
        parameters.

    Attributes
    ----------
    known_invalid_fits_unit_strings
        A dictionary that maps commonly-used fits unit name strings that are
        technically invalid to the correct valid unit type (or unit string).
        This is primarily for variant names like "ELECTRONS/S" which are not
        formally valid, but are unambiguous and frequently enough encountered
        that it is convenient to map them to the correct unit.

    Notes
    -----
    `~astropy.nddata.CCDData` objects can be easily converted to a regular
     Numpy array using `numpy.asarray`.

    For example::

        >>> from astropy.nddata import CCDData
        >>> import numpy as np
        >>> x = CCDData([1,2,3], unit=\'adu\')
        >>> np.asarray(x)
        array([1, 2, 3])

    This is useful, for example, when plotting a 2D image using
    matplotlib.

        >>> from astropy.nddata import CCDData
        >>> from matplotlib import pyplot as plt   # doctest: +SKIP
        >>> x = CCDData([[1,2,3], [4,5,6]], unit=\'adu\')
        >>> plt.imshow(x)   # doctest: +SKIP

    '''
    _wcs: Incomplete
    def __init__(self, *args, **kwd) -> None: ...
    def _slice_wcs(self, item):
        """
        Override the WCS slicing behaviour so that the wcs attribute continues
        to be an `astropy.wcs.WCS`.
        """
    @property
    def data(self): ...
    _data: Incomplete
    @data.setter
    def data(self, value) -> None: ...
    @property
    def wcs(self): ...
    @wcs.setter
    def wcs(self, value) -> None: ...
    @property
    def unit(self): ...
    _unit: Incomplete
    @unit.setter
    def unit(self, value) -> None: ...
    @property
    def psf(self): ...
    _psf: Incomplete
    @psf.setter
    def psf(self, value) -> None: ...
    @property
    def header(self): ...
    meta: Incomplete
    @header.setter
    def header(self, value) -> None: ...
    @property
    def uncertainty(self): ...
    _uncertainty: Incomplete
    @uncertainty.setter
    def uncertainty(self, value) -> None: ...
    def to_hdu(self, hdu_mask: str = 'MASK', hdu_uncertainty: str = 'UNCERT', hdu_flags: Incomplete | None = None, wcs_relax: bool = True, key_uncertainty_type: str = 'UTYPE', as_image_hdu: bool = False, hdu_psf: str = 'PSFIMAGE'):
        """Creates an HDUList object from a CCDData object.

        Parameters
        ----------
        hdu_mask, hdu_uncertainty, hdu_flags, hdu_psf : str or None, optional
            If it is a string append this attribute to the HDUList as
            `~astropy.io.fits.ImageHDU` with the string as extension name.
            Flags are not supported at this time. If ``None`` this attribute
            is not appended.
            Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty,
            ``'PSFIMAGE'`` for psf, and `None` for flags.

        wcs_relax : bool
            Value of the ``relax`` parameter to use in converting the WCS to a
            FITS header using `~astropy.wcs.WCS.to_header`. The common
            ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
            ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
            preserved.

        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
            Default is ``UTYPE``.

            .. versionadded:: 3.1

        as_image_hdu : bool
            If this option is `True`, the first item of the returned
            `~astropy.io.fits.HDUList` is a `~astropy.io.fits.ImageHDU`, instead
            of the default `~astropy.io.fits.PrimaryHDU`.

        Raises
        ------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        NotImplementedError
            Saving flags is not supported.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        """
    def copy(self):
        """
        Return a copy of the CCDData object.
        """
    add: Incomplete
    subtract: Incomplete
    multiply: Incomplete
    divide: Incomplete
    def _insert_in_metadata_fits_safe(self, key, value) -> None:
        """
        Insert key/value pair into metadata in a way that FITS can serialize.

        Parameters
        ----------
        key : str
            Key to be inserted in dictionary.

        value : str or None
            Value to be inserted.

        Notes
        -----
        This addresses a shortcoming of the FITS standard. There are length
        restrictions on both the ``key`` (8 characters) and ``value`` (72
        characters) in the FITS standard. There is a convention for handling
        long keywords and a convention for handling long values, but the
        two conventions cannot be used at the same time.

        This addresses that case by checking the length of the ``key`` and
        ``value`` and, if necessary, shortening the key.
        """
    known_invalid_fits_unit_strings: Incomplete

def fits_ccddata_reader(filename, hdu: int = 0, unit: Incomplete | None = None, hdu_uncertainty: str = 'UNCERT', hdu_mask: str = 'MASK', hdu_flags: Incomplete | None = None, key_uncertainty_type: str = 'UTYPE', hdu_psf: str = 'PSFIMAGE', **kwd):
    """
    Generate a CCDData object from a FITS file.

    Parameters
    ----------
    filename : str
        Name of fits file.

    hdu : int, str, tuple of (str, int), optional
        Index or other identifier of the Header Data Unit of the FITS
        file from which CCDData should be initialized. If zero and
        no data in the primary HDU, it will search for the first
        extension HDU with data. The header will be added to the primary HDU.
        Default is ``0``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used
        as the unit, if present), this argument is used for the unit.
        Default is ``None``.

    hdu_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized. If the
        extension does not exist the uncertainty of the CCDData is ``None``.
        Default is ``'UNCERT'``.

    hdu_mask : str or None, optional
        FITS extension from which the mask should be initialized. If the
        extension does not exist the mask of the CCDData is ``None``.
        Default is ``'MASK'``.

    hdu_flags : str or None, optional
        Currently not implemented.
        Default is ``None``.

    key_uncertainty_type : str, optional
        The header key name where the class name of the uncertainty  is stored
        in the hdu of the uncertainty (if any).
        Default is ``UTYPE``.

        .. versionadded:: 3.1

    hdu_psf : str or None, optional
        FITS extension from which the psf image should be initialized. If the
        extension does not exist the psf of the CCDData is `None`.

    kwd :
        Any additional keyword parameters are passed through to the FITS reader
        in :mod:`astropy.io.fits`; see Notes for additional discussion.

    Notes
    -----
    FITS files that contained scaled data (e.g. unsigned integer images) will
    be scaled and the keywords used to manage scaled data in
    :mod:`astropy.io.fits` are disabled.
    """
def fits_ccddata_writer(ccd_data, filename, hdu_mask: str = 'MASK', hdu_uncertainty: str = 'UNCERT', hdu_flags: Incomplete | None = None, key_uncertainty_type: str = 'UTYPE', as_image_hdu: bool = False, hdu_psf: str = 'PSFIMAGE', **kwd) -> None:
    """
    Write CCDData object to FITS file.

    Parameters
    ----------
    ccd_data : CCDData
        Object to write.

    filename : str
        Name of file.

    hdu_mask, hdu_uncertainty, hdu_flags, hdu_psf : str or None, optional
        If it is a string append this attribute to the HDUList as
        `~astropy.io.fits.ImageHDU` with the string as extension name.
        Flags are not supported at this time. If ``None`` this attribute
        is not appended.
        Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty,
        ``'PSFIMAGE'`` for psf, and `None` for flags.

    key_uncertainty_type : str, optional
        The header key name for the class name of the uncertainty (if any)
        that is used to store the uncertainty type in the uncertainty hdu.
        Default is ``UTYPE``.

        .. versionadded:: 3.1

    as_image_hdu : bool
        If this option is `True`, the first item of the returned
        `~astropy.io.fits.HDUList` is a `~astropy.io.fits.ImageHDU`, instead of
        the default `~astropy.io.fits.PrimaryHDU`.

    kwd :
        All additional keywords are passed to :py:mod:`astropy.io.fits`

    Raises
    ------
    ValueError
        - If ``self.mask`` is set but not a `numpy.ndarray`.
        - If ``self.uncertainty`` is set but not a
          `~astropy.nddata.StdDevUncertainty`.
        - If ``self.uncertainty`` is set but has another unit then
          ``self.data``.

    NotImplementedError
        Saving flags is not supported.
    """
