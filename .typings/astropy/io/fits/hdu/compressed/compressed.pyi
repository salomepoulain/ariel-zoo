from _typeshed import Incomplete
from astropy.io.fits.hdu.image import ImageHDU

__all__ = ['CompImageHDU']

class CompImageHDU(ImageHDU):
    """
    Compressed Image HDU class.
    """
    _default_name: str
    _bintable: Incomplete
    _bitpix: Incomplete
    tile_shape: Incomplete
    hcomp_scale: Incomplete
    hcomp_smooth: Incomplete
    quantize_level: Incomplete
    quantize_method: Incomplete
    dither_seed: Incomplete
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None, name: Incomplete | None = None, compression_type=..., tile_shape: Incomplete | None = None, hcomp_scale=..., hcomp_smooth=..., quantize_level=..., quantize_method=..., dither_seed=..., do_not_scale_image_data: bool = False, uint: bool = True, scale_back: Incomplete | None = None, bintable: Incomplete | None = None) -> None:
        '''
        Parameters
        ----------
        data : array, optional
            Uncompressed image data

        header : `~astropy.io.fits.Header`, optional
            Header to be associated with the image; when reading the HDU from a
            file (data=DELAYED), the header read from the file

        name : str, optional
            The ``EXTNAME`` value; if this value is `None`, then the name from
            the input image header will be used; if there is no name in the
            input image header then the default name ``COMPRESSED_IMAGE`` is
            used.

        compression_type : str, optional
            Compression algorithm: one of
            ``\'RICE_1\'``, ``\'RICE_ONE\'``, ``\'PLIO_1\'``, ``\'GZIP_1\'``,
            ``\'GZIP_2\'``, ``\'HCOMPRESS_1\'``, ``\'NOCOMPRESS\'``

        tile_shape : tuple, optional
            Compression tile shape, which should be specified using the default
            Numpy convention for array shapes (C order). The default is to
            treat each row of image as a tile.

        hcomp_scale : float, optional
            HCOMPRESS scale parameter

        hcomp_smooth : float, optional
            HCOMPRESS smooth parameter

        quantize_level : float, optional
            Floating point quantization level; see note below

        quantize_method : int, optional
            Floating point quantization dithering method; can be either
            ``NO_DITHER`` (-1; default), ``SUBTRACTIVE_DITHER_1`` (1), or
            ``SUBTRACTIVE_DITHER_2`` (2); see note below

        dither_seed : int, optional
            Random seed to use for dithering; can be either an integer in the
            range 1 to 1000 (inclusive), ``DITHER_SEED_CLOCK`` (0; default), or
            ``DITHER_SEED_CHECKSUM`` (-1); see note below

        Notes
        -----
        The astropy.io.fits package supports 2 methods of image compression:

            1) The entire FITS file may be externally compressed with the gzip
               or pkzip utility programs, producing a ``*.gz`` or ``*.zip``
               file, respectively.  When reading compressed files of this type,
               Astropy first uncompresses the entire file into a temporary file
               before performing the requested read operations.  The
               astropy.io.fits package does not support writing to these types
               of compressed files.  This type of compression is supported in
               the ``_File`` class, not in the `CompImageHDU` class.  The file
               compression type is recognized by the ``.gz`` or ``.zip`` file
               name extension.

            2) The `CompImageHDU` class supports the FITS tiled image
               compression convention in which the image is subdivided into a
               grid of rectangular tiles, and each tile of pixels is
               individually compressed.  The details of this FITS compression
               convention are described at the `FITS Support Office web site
               <https://fits.gsfc.nasa.gov/registry/tilecompression.html>`_.
               Basically, the compressed image tiles are stored in rows of a
               variable length array column in a FITS binary table.  The
               astropy.io.fits recognizes that this binary table extension
               contains an image and treats it as if it were an image
               extension.  Under this tile-compression format, FITS header
               keywords remain uncompressed.  At this time, Astropy does not
               support the ability to extract and uncompress sections of the
               image without having to uncompress the entire image.

        The astropy.io.fits package supports 3 general-purpose compression
        algorithms plus one other special-purpose compression technique that is
        designed for data masks with positive integer pixel values.  The 3
        general purpose algorithms are GZIP, Rice, and HCOMPRESS, and the
        special-purpose technique is the IRAF pixel list compression technique
        (PLIO).  The ``compression_type`` parameter defines the compression
        algorithm to be used.

        The FITS image can be subdivided into any desired rectangular grid of
        compression tiles.  With the GZIP, Rice, and PLIO algorithms, the
        default is to take each row of the image as a tile.  The HCOMPRESS
        algorithm is inherently 2-dimensional in nature, so the default in this
        case is to take 16 rows of the image per tile.  In most cases, it makes
        little difference what tiling pattern is used, so the default tiles are
        usually adequate.  In the case of very small images, it could be more
        efficient to compress the whole image as a single tile.  Note that the
        image dimensions are not required to be an integer multiple of the tile
        dimensions; if not, then the tiles at the edges of the image will be
        smaller than the other tiles.  The ``tile_shape`` parameter may be
        provided as a list of tile sizes, one for each dimension in the image.
        For example a ``tile_shape`` value of ``(100,100)`` would divide a 300 X
        300 image into 9 100 X 100 tiles.

        The 4 supported image compression algorithms are all \'lossless\' when
        applied to integer FITS images; the pixel values are preserved exactly
        with no loss of information during the compression and uncompression
        process.  In addition, the HCOMPRESS algorithm supports a \'lossy\'
        compression mode that will produce larger amount of image compression.
        This is achieved by specifying a non-zero value for the ``hcomp_scale``
        parameter.  Since the amount of compression that is achieved depends
        directly on the RMS noise in the image, it is usually more convenient
        to specify the ``hcomp_scale`` factor relative to the RMS noise.
        Setting ``hcomp_scale = 2.5`` means use a scale factor that is 2.5
        times the calculated RMS noise in the image tile.  In some cases it may
        be desirable to specify the exact scaling to be used, instead of
        specifying it relative to the calculated noise value.  This may be done
        by specifying the negative of the desired scale value (typically in the
        range -2 to -100).

        Very high compression factors (of 100 or more) can be achieved by using
        large ``hcomp_scale`` values, however, this can produce undesirable
        \'blocky\' artifacts in the compressed image.  A variation of the
        HCOMPRESS algorithm (called HSCOMPRESS) can be used in this case to
        apply a small amount of smoothing of the image when it is uncompressed
        to help cover up these artifacts.  This smoothing is purely cosmetic
        and does not cause any significant change to the image pixel values.
        Setting the ``hcomp_smooth`` parameter to 1 will engage the smoothing
        algorithm.

        Floating point FITS images (which have ``BITPIX`` = -32 or -64) usually
        contain too much \'noise\' in the least significant bits of the mantissa
        of the pixel values to be effectively compressed with any lossless
        algorithm.  Consequently, floating point images are first quantized
        into scaled integer pixel values (and thus throwing away much of the
        noise) before being compressed with the specified algorithm (either
        GZIP, RICE, or HCOMPRESS).  This technique produces much higher
        compression factors than simply using the GZIP utility to externally
        compress the whole FITS file, but it also means that the original
        floating point value pixel values are not exactly preserved.  When done
        properly, this integer scaling technique will only discard the
        insignificant noise while still preserving all the real information in
        the image.  The amount of precision that is retained in the pixel
        values is controlled by the ``quantize_level`` parameter.  Larger
        values will result in compressed images whose pixels more closely match
        the floating point pixel values, but at the same time the amount of
        compression that is achieved will be reduced.  Users should experiment
        with different values for this parameter to determine the optimal value
        that preserves all the useful information in the image, without
        needlessly preserving all the \'noise\' which will hurt the compression
        efficiency.

        The default value for the ``quantize_level`` scale factor is 16, which
        means that scaled integer pixel values will be quantized such that the
        difference between adjacent integer values will be 1/16th of the noise
        level in the image background.  An optimized algorithm is used to
        accurately estimate the noise in the image.  As an example, if the RMS
        noise in the background pixels of an image = 32.0, then the spacing
        between adjacent scaled integer pixel values will equal 2.0 by default.
        Note that the RMS noise is independently calculated for each tile of
        the image, so the resulting integer scaling factor may fluctuate
        slightly for each tile.  In some cases, it may be desirable to specify
        the exact quantization level to be used, instead of specifying it
        relative to the calculated noise value.  This may be done by specifying
        the negative of desired quantization level for the value of
        ``quantize_level``.  In the previous example, one could specify
        ``quantize_level = -2.0`` so that the quantized integer levels differ
        by 2.0.  Larger negative values for ``quantize_level`` means that the
        levels are more coarsely-spaced, and will produce higher compression
        factors.

        The quantization algorithm can also apply one of two random dithering
        methods in order to reduce bias in the measured intensity of background
        regions.  The default method, specified with the constant
        ``SUBTRACTIVE_DITHER_1`` adds dithering to the zero-point of the
        quantization array itself rather than adding noise to the actual image.
        The random noise is added on a pixel-by-pixel basis, so in order
        restore each pixel from its integer value to its floating point value
        it is necessary to replay the same sequence of random numbers for each
        pixel (see below).  The other method, ``SUBTRACTIVE_DITHER_2``, is
        exactly like the first except that before dithering any pixel with a
        floating point value of ``0.0`` is replaced with the special integer
        value ``-2147483647``.  When the image is uncompressed, pixels with
        this value are restored back to ``0.0`` exactly.  Finally, a value of
        ``NO_DITHER`` disables dithering entirely.

        As mentioned above, when using the subtractive dithering algorithm it
        is necessary to be able to generate a (pseudo-)random sequence of noise
        for each pixel, and replay that same sequence upon decompressing.  To
        facilitate this, a random seed between 1 and 10000 (inclusive) is used
        to seed a random number generator, and that seed is stored in the
        ``ZDITHER0`` keyword in the header of the compressed HDU.  In order to
        use that seed to generate the same sequence of random numbers the same
        random number generator must be used at compression and decompression
        time; for that reason the tiled image convention provides an
        implementation of a very simple pseudo-random number generator.  The
        seed itself can be provided in one of three ways, controllable by the
        ``dither_seed`` argument:  It may be specified manually, or it may be
        generated arbitrarily based on the system\'s clock
        (``DITHER_SEED_CLOCK``) or based on a checksum of the pixels in the
        image\'s first tile (``DITHER_SEED_CHECKSUM``).  The clock-based method
        is the default, and is sufficient to ensure that the value is
        reasonably "arbitrary" and that the same seed is unlikely to be
        generated sequentially.  The checksum method, on the other hand,
        ensures that the same seed is used every time for a specific image.
        This is particularly useful for software testing as it ensures that the
        same image will always use the same seed.
        '''
    def _remove_unnecessary_default_extnames(self, header) -> None:
        """Remove default EXTNAME values if they are unnecessary.

        Some data files (eg from CFHT) can have the default EXTNAME and
        an explicit value.  This method removes the default if a more
        specific header exists. It also removes any duplicate default
        values.
        """
    @classmethod
    def match_header(cls, header): ...
    @property
    def compression_type(self): ...
    _compression_type: Incomplete
    @compression_type.setter
    def compression_type(self, value) -> None: ...
    def _get_bintable_without_data(self):
        """
        Convert the current ImageHDU (excluding the actual data) to a BinTableHDU
        with the correct header.
        """
    @property
    def _data_loaded(self):
        """
        Whether the data is fully decompressed into self.data - note that is
        a little different to _data_loaded on other HDUs, but it is conceptually
        the same idea in a way.
        """
    @property
    def _data_shape(self): ...
    @property
    def compressed_data(self): ...
    def _bintable_to_image_header(self): ...
    def _add_data_to_bintable(self, bintable) -> None:
        """
        Compress the image data so that it may be written to a file.
        """
    _tmp_bintable: Incomplete
    def _prewriteto(self, checksum: bool = False, inplace: bool = False): ...
    def _writeto(self, fileobj, inplace: bool = False, copy: bool = False): ...
    def _postwriteto(self) -> None: ...
    def _close(self, closed: bool = True): ...
    def _generate_dither_seed(self, seed): ...
    @property
    def section(self):
        """
        Efficiently access a section of the image array

        This property can be used to access a section of the data without
        loading and decompressing the entire array into memory.

        The :class:`~astropy.io.fits.CompImageSection` object returned by this
        attribute is not meant to be used directly by itself. Rather, slices of
        the section return the appropriate slice of the data, and loads *only*
        that section into memory. Any valid basic Numpy index can be used to
        slice :class:`~astropy.io.fits.CompImageSection`.

        Note that accessing data using :attr:`CompImageHDU.section` will always
        load tiles one at a time from disk, and therefore when accessing a large
        fraction of the data (or slicing it in a way that would cause most tiles
        to be loaded) you may obtain better performance by using
        :attr:`CompImageHDU.data`.
        """
    def _verify(self, *args, **kwargs): ...
    @property
    def _data_offset(self): ...
    @_data_offset.setter
    def _data_offset(self, value) -> None: ...
    @property
    def _header_offset(self): ...
    @_header_offset.setter
    def _header_offset(self, value) -> None: ...
    @property
    def _data_size(self): ...
    @_data_size.setter
    def _data_size(self, value) -> None: ...
