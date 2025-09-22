from .wcsapi.fitswcs import FITSWCSAPIMixin
from _typeshed import Incomplete
from astropy.utils.exceptions import AstropyWarning

__all__ = ['FITSFixedWarning', 'WCS', 'find_all_wcs', 'DistortionLookupTable', 'Sip', 'Tabprm', 'Wcsprm', 'Auxprm', 'Celprm', 'Prjprm', 'Wtbarr', 'WCSBase', 'validate', 'WcsError', 'SingularMatrixError', 'InconsistentAxisTypesError', 'InvalidTransformError', 'InvalidCoordinateError', 'InvalidPrjParametersError', 'NoSolutionError', 'InvalidSubimageSpecificationError', 'NoConvergence', 'NonseparableSubimageCoordinateSystemError', 'NoWcsKeywordsFoundError', 'InvalidTabularParametersError', 'WCSSUB_LONGITUDE', 'WCSSUB_LATITUDE', 'WCSSUB_CUBEFACE', 'WCSSUB_SPECTRAL', 'WCSSUB_STOKES', 'WCSSUB_TIME', 'WCSSUB_CELESTIAL', 'WCSHDR_IMGHEAD', 'WCSHDR_BIMGARR', 'WCSHDR_PIXLIST', 'WCSHDR_none', 'WCSHDR_all', 'WCSHDR_reject', 'WCSHDR_strict', 'WCSHDR_CROTAia', 'WCSHDR_EPOCHa', 'WCSHDR_VELREFa', 'WCSHDR_CD00i00j', 'WCSHDR_PC00i00j', 'WCSHDR_PROJPn', 'WCSHDR_CD0i_0ja', 'WCSHDR_PC0i_0ja', 'WCSHDR_PV0i_0ma', 'WCSHDR_PS0i_0ma', 'WCSHDR_RADECSYS', 'WCSHDR_VSOURCE', 'WCSHDR_DOBSn', 'WCSHDR_LONGKEY', 'WCSHDR_CNAMn', 'WCSHDR_AUXIMG', 'WCSHDR_ALLIMG', 'WCSHDO_none', 'WCSHDO_all', 'WCSHDO_safe', 'WCSHDO_DOBSn', 'WCSHDO_TPCn_ka', 'WCSHDO_PVn_ma', 'WCSHDO_CRPXna', 'WCSHDO_CNAMna', 'WCSHDO_WCSNna', 'WCSHDO_P12', 'WCSHDO_P13', 'WCSHDO_P14', 'WCSHDO_P15', 'WCSHDO_P16', 'WCSHDO_P17', 'WCSHDO_EFMT', 'WCSCOMPARE_ANCILLARY', 'WCSCOMPARE_TILING', 'WCSCOMPARE_CRPIX', 'PRJ_PVN', 'PRJ_CODES', 'PRJ_ZENITHAL', 'PRJ_CYLINDRICAL', 'PRJ_PSEUDOCYLINDRICAL', 'PRJ_CONVENTIONAL', 'PRJ_CONIC', 'PRJ_POLYCONIC', 'PRJ_QUADCUBE', 'PRJ_HEALPIX']

WCSBase: Incomplete
DistortionLookupTable: Incomplete
Sip: Incomplete
Wcsprm: Incomplete
Auxprm: Incomplete
Celprm: Incomplete
Prjprm: Incomplete
Tabprm: Incomplete
Wtbarr: Incomplete
WcsError: Incomplete
SingularMatrixError: Incomplete
InconsistentAxisTypesError: Incomplete
InvalidTransformError: Incomplete
InvalidCoordinateError: Incomplete
NoSolutionError: Incomplete
InvalidSubimageSpecificationError: Incomplete
NonseparableSubimageCoordinateSystemError: Incomplete
NoWcsKeywordsFoundError: Incomplete
InvalidTabularParametersError: Incomplete
InvalidPrjParametersError: Incomplete
WCSBase = object
Wcsprm = object
DistortionLookupTable = object
Sip = object
Tabprm = object
Wtbarr = object

class NoConvergence(Exception):
    """
    An error class used to report non-convergence and/or divergence
    of numerical methods. It is used to report errors in the
    iterative solution used by
    the :py:meth:`~astropy.wcs.WCS.all_world2pix`.

    Attributes
    ----------
    best_solution : `numpy.ndarray`
        Best solution achieved by the numerical method.

    accuracy : `numpy.ndarray`
        Accuracy of the ``best_solution``.

    niter : `int`
        Number of iterations performed by the numerical method
        to compute ``best_solution``.

    divergent : None, `numpy.ndarray`
        Indices of the points in ``best_solution`` array
        for which the solution appears to be divergent. If the
        solution does not diverge, ``divergent`` will be set to `None`.

    slow_conv : None, `numpy.ndarray`
        Indices of the solutions in ``best_solution`` array
        for which the solution failed to converge within the
        specified maximum number of iterations. If there are no
        non-converging solutions (i.e., if the required accuracy
        has been achieved for all input data points)
        then ``slow_conv`` will be set to `None`.

    """
    best_solution: Incomplete
    accuracy: Incomplete
    niter: Incomplete
    divergent: Incomplete
    slow_conv: Incomplete
    def __init__(self, *args, best_solution: Incomplete | None = None, accuracy: Incomplete | None = None, niter: Incomplete | None = None, divergent: Incomplete | None = None, slow_conv: Incomplete | None = None) -> None: ...

class FITSFixedWarning(AstropyWarning):
    """
    The warning raised when the contents of the FITS header have been
    modified to be standards compliant.
    """

class WCS(FITSWCSAPIMixin, WCSBase):
    '''WCS objects perform standard WCS transformations, and correct for
    `SIP`_ and `distortion paper`_ table-lookup transformations, based
    on the WCS keywords and supplementary data read from a FITS file.

    See also: https://docs.astropy.org/en/stable/wcs/

    Parameters
    ----------
    header : `~astropy.io.fits.Header`, `~astropy.io.fits.hdu.image.PrimaryHDU`, `~astropy.io.fits.hdu.image.ImageHDU`, str, dict-like, or None, optional
        If *header* is not provided or None, the object will be
        initialized to default values.

    fobj : `~astropy.io.fits.HDUList`, optional
        It is needed when header keywords point to a `distortion
        paper`_ lookup table stored in a different extension.

    key : str, optional
        The name of a particular WCS transform to use.  This may be
        either ``\' \'`` or ``\'A\'``-``\'Z\'`` and corresponds to the
        ``"a"`` part of the ``CTYPEia`` cards.  *key* may only be
        provided if *header* is also provided.

    minerr : float, optional
        The minimum value a distortion correction must have in order
        to be applied. If the value of ``CQERRja`` is smaller than
        *minerr*, the corresponding distortion is not applied.

    relax : bool or int, optional
        Degree of permissiveness:

        - `True` (default): Admit all recognized informal extensions
          of the WCS standard.

        - `False`: Recognize only FITS keywords defined by the
          published WCS standard.

        - `int`: a bit field selecting specific extensions to accept.
          See :ref:`astropy:relaxread` for details.

    naxis : int or sequence, optional
        Extracts specific coordinate axes using
        :meth:`~astropy.wcs.Wcsprm.sub`.  If a header is provided, and
        *naxis* is not ``None``, *naxis* will be passed to
        :meth:`~astropy.wcs.Wcsprm.sub` in order to select specific
        axes from the header.  See :meth:`~astropy.wcs.Wcsprm.sub` for
        more details about this parameter.

    keysel : sequence of str, optional
        A sequence of flags used to select the keyword types
        considered by wcslib.  When ``None``, only the standard image
        header keywords are considered (and the underlying wcspih() C
        function is called).  To use binary table image array or pixel
        list keywords, *keysel* must be set.

        Each element in the list should be one of the following
        strings:

        - \'image\': Image header keywords

        - \'binary\': Binary table image array keywords

        - \'pixel\': Pixel list keywords

        Keywords such as ``EQUIna`` or ``RFRQna`` that are common to
        binary table image arrays and pixel lists (including
        ``WCSNna`` and ``TWCSna``) are selected by both \'binary\' and
        \'pixel\'.

    colsel : sequence of int, optional
        A sequence of table column numbers used to restrict the WCS
        transformations considered to only those pertaining to the
        specified columns.  If `None`, there is no restriction.

    fix : bool, optional
        When `True` (default), call `~astropy.wcs.Wcsprm.fix` on
        the resulting object to fix any non-standard uses in the
        header.  `FITSFixedWarning` Warnings will be emitted if any
        changes were made.

    translate_units : str, optional
        Specify which potentially unsafe translations of non-standard
        unit strings to perform.  By default, performs none.  See
        `WCS.fix` for more information about this parameter.  Only
        effective when ``fix`` is `True`.

    Raises
    ------
    MemoryError
         Memory allocation failed.

    ValueError
         Invalid key.

    KeyError
         Key not found in FITS header.

    ValueError
         Lookup table distortion present in the header but *fobj* was
         not provided.

    Notes
    -----
    1. astropy.wcs supports arbitrary *n* dimensions for the core WCS
       (the transformations handled by WCSLIB).  However, the
       `distortion paper`_ lookup table and `SIP`_ distortions must be
       two dimensional.  Therefore, if you try to create a WCS object
       where the core WCS has a different number of dimensions than 2
       and that object also contains a `distortion paper`_ lookup
       table or `SIP`_ distortion, a `ValueError`
       exception will be raised.  To avoid this, consider using the
       *naxis* kwarg to select two dimensions from the core WCS.

    2. The number of coordinate axes in the transformation is not
       determined directly from the ``NAXIS`` keyword but instead from
       the highest of:

           - ``NAXIS`` keyword

           - ``WCSAXESa`` keyword

           - The highest axis number in any parameterized WCS keyword.
             The keyvalue, as well as the keyword, must be
             syntactically valid otherwise it will not be considered.

       If none of these keyword types is present, i.e. if the header
       only contains auxiliary WCS keywords for a particular
       coordinate representation, then no coordinate description is
       constructed for it.

       The number of axes, which is set as the ``naxis`` member, may
       differ for different coordinate representations of the same
       image.

    3. When the header includes duplicate keywords, in most cases the
       last encountered is used.

    4. `~astropy.wcs.Wcsprm.set` is called immediately after
       construction, so any invalid keywords or transformations will
       be raised by the constructor, not when subsequently calling a
       transformation method.

    '''
    _init_kwargs: Incomplete
    naxis: Incomplete
    _pixel_bounds: Incomplete
    def __init__(self, header: Incomplete | None = None, fobj: Incomplete | None = None, key: str = ' ', minerr: float = 0.0, relax: bool = True, naxis: Incomplete | None = None, keysel: Incomplete | None = None, colsel: Incomplete | None = None, fix: bool = True, translate_units: str = '', _do_set: bool = True) -> None: ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def copy(self):
        """
        Return a shallow copy of the object.

        Convenience method so user doesn't have to import the
        :mod:`copy` stdlib module.

        .. warning::
            Use `deepcopy` instead of `copy` unless you know why you need a
            shallow copy.
        """
    def deepcopy(self):
        """
        Return a deep copy of the object.

        Convenience method so user doesn't have to import the
        :mod:`copy` stdlib module.
        """
    def sub(self, axes: Incomplete | None = None): ...
    sip: Incomplete
    def _fix_scamp(self) -> None:
        """
        Remove SCAMP's PVi_m distortion parameters if SIP distortion parameters
        are also present. Some projects (e.g., Palomar Transient Factory)
        convert SCAMP's distortion parameters (which abuse the PVi_m cards) to
        SIP. However, wcslib gets confused by the presence of both SCAMP and
        SIP distortion parameters.

        See https://github.com/astropy/astropy/issues/299.

        SCAMP uses TAN projection exclusively. The case of CTYPE ending
        in -TAN should have been handled by ``_fix_pre2012_scamp_tpv()`` before
        calling this function.
        """
    def fix(self, translate_units: str = '', naxis: Incomplete | None = None) -> None:
        '''
        Perform the fix operations from wcslib, and warn about any
        changes it has made.

        Parameters
        ----------
        translate_units : str, optional
            Specify which potentially unsafe translations of
            non-standard unit strings to perform.  By default,
            performs none.

            Although ``"S"`` is commonly used to represent seconds,
            its translation to ``"s"`` is potentially unsafe since the
            standard recognizes ``"S"`` formally as Siemens, however
            rarely that may be used.  The same applies to ``"H"`` for
            hours (Henry), and ``"D"`` for days (Debye).

            This string controls what to do in such cases, and is
            case-insensitive.

            - If the string contains ``"s"``, translate ``"S"`` to
              ``"s"``.

            - If the string contains ``"h"``, translate ``"H"`` to
              ``"h"``.

            - If the string contains ``"d"``, translate ``"D"`` to
              ``"d"``.

            Thus ``\'\'`` doesn\'t do any unsafe translations, whereas
            ``\'shd\'`` does all of them.

        naxis : int array, optional
            Image axis lengths.  If this array is set to zero or
            ``None``, then `~astropy.wcs.Wcsprm.cylfix` will not be
            invoked.
        '''
    def calc_footprint(self, header: Incomplete | None = None, undistort: bool = True, axes: Incomplete | None = None, center: bool = True):
        """
        Calculates the footprint of the image on the sky.

        A footprint is defined as the positions of the corners of the
        image on the sky after all available distortions have been
        applied.

        Parameters
        ----------
        header : `~astropy.io.fits.Header` object, optional
            Used to get ``NAXIS1`` and ``NAXIS2``
            header and axes are mutually exclusive, alternative ways
            to provide the same information.

        undistort : bool, optional
            If `True`, take SIP and distortion lookup table into
            account

        axes : (int, int), optional
            If provided, use the given sequence as the shape of the
            image.  Otherwise, use the ``NAXIS1`` and ``NAXIS2``
            keywords from the header that was used to create this
            `WCS` object.

        center : bool, optional
            If `True` use the center of the pixel, otherwise use the corner.

        Returns
        -------
        coord : (4, 2) array of (*x*, *y*) coordinates.
            The order is clockwise starting with the bottom left corner.
        """
    def _read_det2im_kw(self, header, fobj, err: float = 0.0):
        """
        Create a `distortion paper`_ type lookup table for detector to
        image plane correction.
        """
    def _read_d2im_old_format(self, header, fobj, axiscorr): ...
    def _write_det2im(self, hdulist) -> None:
        """
        Writes a `distortion paper`_ type lookup table to the given
        `~astropy.io.fits.HDUList`.
        """
    def _read_distortion_kw(self, header, fobj, dist: str = 'CPDIS', err: float = 0.0):
        """
        Reads `distortion paper`_ table-lookup keywords and data, and
        returns a 2-tuple of `~astropy.wcs.DistortionLookupTable`
        objects.

        If no `distortion paper`_ keywords are found, ``(None, None)``
        is returned.
        """
    def _write_distortion_kw(self, hdulist, dist: str = 'CPDIS') -> None:
        """
        Write out `distortion paper`_ keywords to the given
        `~astropy.io.fits.HDUList`.
        """
    def _fix_pre2012_scamp_tpv(self, header, wcskey: str = '') -> None:
        """
        Replace -TAN with TPV (for pre-2012 SCAMP headers that use -TAN
        in CTYPE). Ignore SIP if present. This follows recommendations in
        Section 7 in
        http://web.ipac.caltech.edu/staff/shupe/reprints/SIP_to_PV_SPIE2012.pdf.

        This is to deal with pre-2012 headers that may contain TPV with a
        CTYPE that ends in '-TAN' (post-2012 they should end in '-TPV' when
        SCAMP has adopted the new TPV convention).
        """
    @staticmethod
    def _remove_sip_kw(header, del_order: bool = False) -> None:
        """
        Remove SIP information from a header.
        """
    def _read_sip_kw(self, header, wcskey: str = ''):
        """
        Reads `SIP`_ header keywords and returns a `~astropy.wcs.Sip`
        object.

        If no `SIP`_ header keywords are found, ``None`` is returned.
        """
    def _write_sip_kw(self):
        """
        Write out SIP keywords.  Returns a dictionary of key-value
        pairs.
        """
    def _denormalize_sky(self, sky): ...
    def _normalize_sky(self, sky): ...
    def _array_converter(self, func, sky, *args, ra_dec_order: bool = False):
        """
        A helper function to support reading either a pair of arrays
        or a single Nx2 array.
        """
    def all_pix2world(self, *args, **kwargs): ...
    def wcs_pix2world(self, *args, **kwargs): ...
    def _all_world2pix(self, world, origin, tolerance, maxiter, adaptive, detect_divergence, quiet): ...
    def all_world2pix(self, *args, tolerance: float = 0.0001, maxiter: int = 20, adaptive: bool = False, detect_divergence: bool = True, quiet: bool = False, **kwargs): ...
    def wcs_world2pix(self, *args, **kwargs): ...
    def pix2foc(self, *args): ...
    def p4_pix2foc(self, *args): ...
    def det2im(self, *args): ...
    def sip_pix2foc(self, *args): ...
    def sip_foc2pix(self, *args): ...
    def proj_plane_pixel_scales(self):
        '''
        Calculate pixel scales along each axis of the image pixel at
        the ``CRPIX`` location once it is projected onto the
        "plane of intermediate world coordinates" as defined in
        `Greisen & Calabretta 2002, A&A, 395, 1061 <https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1061G>`_.

        .. note::
            This method is concerned **only** about the transformation
            "image plane"->"projection plane" and **not** about the
            transformation "celestial sphere"->"projection plane"->"image plane".
            Therefore, this function ignores distortions arising due to
            non-linear nature of most projections.

        .. note::
            This method only returns sensible answers if the WCS contains
            celestial axes, i.e., the `~astropy.wcs.WCS.celestial` WCS object.

        Returns
        -------
        scale : list of `~astropy.units.Quantity`
            A vector of projection plane increments corresponding to each
            pixel side (axis).

        See Also
        --------
        astropy.wcs.utils.proj_plane_pixel_scales

        '''
    def proj_plane_pixel_area(self):
        '''
        For a **celestial** WCS (see `astropy.wcs.WCS.celestial`), returns pixel
        area of the image pixel at the ``CRPIX`` location once it is projected
        onto the "plane of intermediate world coordinates" as defined in
        `Greisen & Calabretta 2002, A&A, 395, 1061 <https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1061G>`_.

        .. note::
            This function is concerned **only** about the transformation
            "image plane"->"projection plane" and **not** about the
            transformation "celestial sphere"->"projection plane"->"image plane".
            Therefore, this function ignores distortions arising due to
            non-linear nature of most projections.

        .. note::
            This method only returns sensible answers if the WCS contains
            celestial axes, i.e., the `~astropy.wcs.WCS.celestial` WCS object.

        Returns
        -------
        area : `~astropy.units.Quantity`
            Area (in the projection plane) of the pixel at ``CRPIX`` location.

        Raises
        ------
        ValueError
            Pixel area is defined only for 2D pixels. Most likely the
            `~astropy.wcs.Wcsprm.cd` matrix of the `~astropy.wcs.WCS.celestial`
            WCS is not a square matrix of second order.

        Notes
        -----
        Depending on the application, square root of the pixel area can be used to
        represent a single pixel scale of an equivalent square pixel
        whose area is equal to the area of a generally non-square pixel.

        See Also
        --------
        astropy.wcs.utils.proj_plane_pixel_area

        '''
    def to_fits(self, relax: bool = False, key: Incomplete | None = None):
        '''
        Generate an `~astropy.io.fits.HDUList` object with all of the
        information stored in this object.  This should be logically identical
        to the input FITS file, but it will be normalized in a number of ways.

        See `to_header` for some warnings about the output produced.

        Parameters
        ----------
        relax : bool or int, optional
            Degree of permissiveness:

            - `False` (default): Write all extensions that are
              considered to be safe and recommended.

            - `True`: Write all recognized informal extensions of the
              WCS standard.

            - `int`: a bit field selecting specific extensions to
              write.  See :ref:`astropy:relaxwrite` for details.

        key : str
            The name of a particular WCS transform to use.  This may be
            either ``\' \'`` or ``\'A\'``-``\'Z\'`` and corresponds to the ``"a"``
            part of the ``CTYPEia`` cards.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        '''
    def to_header(self, relax: Incomplete | None = None, key: Incomplete | None = None):
        '''Generate an `astropy.io.fits.Header` object with the basic WCS
        and SIP information stored in this object.  This should be
        logically identical to the input FITS file, but it will be
        normalized in a number of ways.

        .. warning::

          This function does not write out FITS WCS `distortion
          paper`_ information, since that requires multiple FITS
          header data units.  To get a full representation of
          everything in this object, use `to_fits`.

        Parameters
        ----------
        relax : bool or int, optional
            Degree of permissiveness:

            - `False` (default): Write all extensions that are
              considered to be safe and recommended.

            - `True`: Write all recognized informal extensions of the
              WCS standard.

            - `int`: a bit field selecting specific extensions to
              write.  See :ref:`astropy:relaxwrite` for details.

            If the ``relax`` keyword argument is not given and any
            keywords were omitted from the output, an
            `~astropy.utils.exceptions.AstropyWarning` is displayed.
            To override this, explicitly pass a value to ``relax``.

        key : str
            The name of a particular WCS transform to use.  This may be
            either ``\' \'`` or ``\'A\'``-``\'Z\'`` and corresponds to the ``"a"``
            part of the ``CTYPEia`` cards.

        Returns
        -------
        header : `astropy.io.fits.Header`

        Notes
        -----
        The output header will almost certainly differ from the input in a
        number of respects:

          1. The output header only contains WCS-related keywords.  In
             particular, it does not contain syntactically-required
             keywords such as ``SIMPLE``, ``NAXIS``, ``BITPIX``, or
             ``END``.

          2. Deprecated (e.g. ``CROTAn``) or non-standard usage will
             be translated to standard (this is partially dependent on
             whether ``fix`` was applied).

          3. Quantities will be converted to the units used internally,
             basically SI with the addition of degrees.

          4. Floating-point quantities may be given to a different decimal
             precision.

          5. Elements of the ``PCi_j`` matrix will be written if and
             only if they differ from the unit matrix.  Thus, if the
             matrix is unity then no elements will be written.

          6. Additional keywords such as ``WCSAXES``, ``CUNITia``,
             ``LONPOLEa`` and ``LATPOLEa`` may appear.

          7. The original keycomments will be lost, although
             `to_header` tries hard to write meaningful comments.

          8. Keyword order may be changed.

        '''
    def _fix_ctype(self, header, add_sip: bool = True, log_message: bool = True):
        '''
        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            FITS header.
        add_sip : bool
            Flag indicating whether "-SIP" should be added or removed from CTYPE keywords.

            Remove "-SIP" from CTYPE when writing out a header with relax=False.
            This needs to be done outside ``to_header`` because ``to_header`` runs
            twice when ``relax=False`` and the second time ``relax`` is set to ``True``
            to display the missing keywords.

            If the user requested SIP distortion to be written out add "-SIP" to
            CTYPE if it is missing.
        '''
    def to_header_string(self, relax: Incomplete | None = None):
        """
        Identical to `to_header`, but returns a string containing the
        header cards.
        """
    def footprint_to_file(self, filename: str = 'footprint.reg', color: str = 'green', width: int = 2, coordsys: Incomplete | None = None) -> None:
        """
        Writes out a `ds9`_ style regions file. It can be loaded
        directly by `ds9`_.

        Parameters
        ----------
        filename : str, optional
            Output file name - default is ``'footprint.reg'``

        color : str, optional
            Color to use when plotting the line.

        width : int, optional
            Width of the region line.

        coordsys : str, optional
            Coordinate system. If not specified (default), the ``radesys``
            value is used. For all possible values, see
            http://ds9.si.edu/doc/ref/region.html#RegionFileFormat

        """
    _naxis: Incomplete
    def _get_naxis(self, header: Incomplete | None = None) -> None: ...
    def printwcs(self) -> None: ...
    def __repr__(self) -> str:
        """
        Return a short description. Simply porting the behavior from
        the `printwcs()` method.
        """
    def get_axis_types(self):
        '''
        Similar to `self.wcsprm.axis_types <astropy.wcs.Wcsprm.axis_types>`
        but provides the information in a more Python-friendly format.

        Returns
        -------
        result : list of dict

            Returns a list of dictionaries, one for each axis, each
            containing attributes about the type of that axis.

            Each dictionary has the following keys:

            - \'coordinate_type\':

              - None: Non-specific coordinate type.

              - \'stokes\': Stokes coordinate.

              - \'celestial\': Celestial coordinate (including ``CUBEFACE``).

              - \'spectral\': Spectral coordinate.

            - \'scale\':

              - \'linear\': Linear axis.

              - \'quantized\': Quantized axis (``STOKES``, ``CUBEFACE``).

              - \'non-linear celestial\': Non-linear celestial axis.

              - \'non-linear spectral\': Non-linear spectral axis.

              - \'logarithmic\': Logarithmic axis.

              - \'tabular\': Tabular axis.

            - \'group\'

              - Group number, e.g. lookup table number

            - \'number\'

              - For celestial axes:

                - 0: Longitude coordinate.

                - 1: Latitude coordinate.

                - 2: ``CUBEFACE`` number.

              - For lookup tables:

                - the axis number in a multidimensional table.

            ``CTYPEia`` in ``"4-3"`` form with unrecognized algorithm code will
            generate an error.
        '''
    def __reduce__(self):
        """
        Support pickling of WCS objects.  This is done by serializing
        to an in-memory FITS file and dumping that as a string.
        """
    def dropaxis(self, dropax):
        """
        Remove an axis from the WCS.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The WCS with naxis to be chopped to naxis-1
        dropax : int
            The index of the WCS to drop, counting from 0 (i.e., python convention,
            not FITS convention)

        Returns
        -------
        `~astropy.wcs.WCS`
            A new `~astropy.wcs.WCS` instance with one axis fewer
        """
    def swapaxes(self, ax0, ax1):
        """
        Swap axes in a WCS.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The WCS to have its axes swapped
        ax0 : int
        ax1 : int
            The indices of the WCS to be swapped, counting from 0 (i.e., python
            convention, not FITS convention)

        Returns
        -------
        `~astropy.wcs.WCS`
            A new `~astropy.wcs.WCS` instance with the same number of axes,
            but two swapped
        """
    def reorient_celestial_first(self):
        """
        Reorient the WCS such that the celestial axes are first, followed by
        the spectral axis, followed by any others.
        Assumes at least celestial axes are present.
        """
    def slice(self, view, numpy_order: bool = True):
        """
        Slice a WCS instance using a Numpy slice. The order of the slice should
        be reversed (as for the data) compared to the natural WCS order.

        Parameters
        ----------
        view : tuple
            A tuple containing the same number of slices as the WCS system.
            The ``step`` method, the third argument to a slice, is not
            presently supported.
        numpy_order : bool, default: True
            Use numpy order, i.e. slice the WCS so that an identical slice
            applied to a numpy array will slice the array and WCS in the same
            way. If set to `False`, the WCS will be sliced in FITS order,
            meaning the first slice will be applied to the *last* numpy index
            but the *first* WCS axis.

        Returns
        -------
        wcs_new : `~astropy.wcs.WCS`
            A new resampled WCS axis
        """
    def __getitem__(self, item): ...
    def __iter__(self): ...
    @property
    def axis_type_names(self):
        """
        World names for each coordinate axis.

        Returns
        -------
        list of str
            A list of names along each axis.
        """
    @property
    def celestial(self):
        """
        A copy of the current WCS with only the celestial axes included.
        """
    @property
    def is_celestial(self): ...
    @property
    def has_celestial(self): ...
    @property
    def spectral(self):
        """
        A copy of the current WCS with only the spectral axes included.
        """
    @property
    def is_spectral(self): ...
    @property
    def has_spectral(self): ...
    @property
    def temporal(self):
        """
        A copy of the current WCS with only the time axes included.
        """
    @property
    def is_temporal(self): ...
    @property
    def has_temporal(self): ...
    @property
    def has_distortion(self):
        """
        Returns `True` if any distortion terms are present.
        """
    @property
    def pixel_scale_matrix(self): ...
    def footprint_contains(self, coord, **kwargs):
        """
        Determines if a given SkyCoord is contained in the wcs footprint.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            The coordinate to check if it is within the wcs coordinate.
        **kwargs :
           Additional arguments to pass to `~astropy.coordinates.SkyCoord.to_pixel`

        Returns
        -------
        response : bool
           True means the WCS footprint contains the coordinate, False means it does not.
        """

def find_all_wcs(header, relax: bool = True, keysel: Incomplete | None = None, fix: bool = True, translate_units: str = '', _do_set: bool = True):
    """
    Find all the WCS transformations in the given header.

    Parameters
    ----------
    header : str or `~astropy.io.fits.Header` object.

    relax : bool or int, optional
        Degree of permissiveness:

        - `True` (default): Admit all recognized informal extensions of the
          WCS standard.

        - `False`: Recognize only FITS keywords defined by the
          published WCS standard.

        - `int`: a bit field selecting specific extensions to accept.
          See :ref:`astropy:relaxread` for details.

    keysel : sequence of str, optional
        A list of flags used to select the keyword types considered by
        wcslib.  When ``None``, only the standard image header
        keywords are considered (and the underlying wcspih() C
        function is called).  To use binary table image array or pixel
        list keywords, *keysel* must be set.

        Each element in the list should be one of the following strings:

            - 'image': Image header keywords

            - 'binary': Binary table image array keywords

            - 'pixel': Pixel list keywords

        Keywords such as ``EQUIna`` or ``RFRQna`` that are common to
        binary table image arrays and pixel lists (including
        ``WCSNna`` and ``TWCSna``) are selected by both 'binary' and
        'pixel'.

    fix : bool, optional
        When `True` (default), call `~astropy.wcs.Wcsprm.fix` on
        the resulting objects to fix any non-standard uses in the
        header.  `FITSFixedWarning` warnings will be emitted if any
        changes were made.

    translate_units : str, optional
        Specify which potentially unsafe translations of non-standard
        unit strings to perform.  By default, performs none.  See
        `WCS.fix` for more information about this parameter.  Only
        effective when ``fix`` is `True`.

    Returns
    -------
    wcses : list of `WCS`
    """
def validate(source):
    """
    Prints a WCS validation report for the given FITS file.

    Parameters
    ----------
    source : str or file-like or `~astropy.io.fits.HDUList`
        The FITS file to validate.

    Returns
    -------
    results : list subclass instance
        The result is returned as nested lists.  The first level
        corresponds to the HDUs in the given file.  The next level has
        an entry for each WCS found in that header.  The special
        subclass of list will pretty-print the results as a table when
        printed.

    """

# Names in __all__ with no definition:
#   PRJ_CODES
#   PRJ_CONIC
#   PRJ_CONVENTIONAL
#   PRJ_CYLINDRICAL
#   PRJ_HEALPIX
#   PRJ_POLYCONIC
#   PRJ_PSEUDOCYLINDRICAL
#   PRJ_PVN
#   PRJ_QUADCUBE
#   PRJ_ZENITHAL
#   WCSCOMPARE_ANCILLARY
#   WCSCOMPARE_CRPIX
#   WCSCOMPARE_TILING
#   WCSHDO_CNAMna
#   WCSHDO_CRPXna
#   WCSHDO_DOBSn
#   WCSHDO_EFMT
#   WCSHDO_P12
#   WCSHDO_P13
#   WCSHDO_P14
#   WCSHDO_P15
#   WCSHDO_P16
#   WCSHDO_P17
#   WCSHDO_PVn_ma
#   WCSHDO_TPCn_ka
#   WCSHDO_WCSNna
#   WCSHDO_all
#   WCSHDO_none
#   WCSHDO_safe
#   WCSHDR_ALLIMG
#   WCSHDR_AUXIMG
#   WCSHDR_BIMGARR
#   WCSHDR_CD00i00j
#   WCSHDR_CD0i_0ja
#   WCSHDR_CNAMn
#   WCSHDR_CROTAia
#   WCSHDR_DOBSn
#   WCSHDR_EPOCHa
#   WCSHDR_IMGHEAD
#   WCSHDR_LONGKEY
#   WCSHDR_PC00i00j
#   WCSHDR_PC0i_0ja
#   WCSHDR_PIXLIST
#   WCSHDR_PROJPn
#   WCSHDR_PS0i_0ma
#   WCSHDR_PV0i_0ma
#   WCSHDR_RADECSYS
#   WCSHDR_VELREFa
#   WCSHDR_VSOURCE
#   WCSHDR_all
#   WCSHDR_none
#   WCSHDR_reject
#   WCSHDR_strict
#   WCSSUB_CELESTIAL
#   WCSSUB_CUBEFACE
#   WCSSUB_LATITUDE
#   WCSSUB_LONGITUDE
#   WCSSUB_SPECTRAL
#   WCSSUB_STOKES
#   WCSSUB_TIME
