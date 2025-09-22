__all__ = ['compress_image_data', 'decompress_image_data_section']

def decompress_image_data_section(compressed_data, compression_type, compressed_header, bintable, image_header, first_tile_index, last_tile_index):
    """
    Decompress the data in a `~astropy.io.fits.CompImageHDU`.

    Parameters
    ----------
    compressed_data : `~astropy.io.fits.FITS_rec`
        The compressed data
    compression_type : str
        The compression algorithm
    compressed_header : `~astropy.io.fits.Header`
        The header of the compressed binary table
    bintable : `~astropy.io.fits.BinTableHDU`
        The binary table HDU, used to access the raw heap data
    first_tile_index : iterable
        The indices of the first tile to decompress along each dimension
    last_tile_index : iterable
        The indices of the last tile to decompress along each dimension

    Returns
    -------
    data : `numpy.ndarray`
        The decompressed data array.
    """
def compress_image_data(image_data, compression_type, compressed_header, compressed_coldefs):
    """
    Compress the data in a `~astropy.io.fits.CompImageHDU`.

    The input HDU is expected to have a uncompressed numpy array as it's
    ``.data`` attribute.

    Parameters
    ----------
    image_data : `~numpy.ndarray`
        The image data to compress
    compression_type : str
        The compression algorithm
    compressed_header : `~astropy.io.fits.Header`
        The header of the compressed binary table
    compressed_coldefs : `~astropy.io.fits.ColDefs`
        The ColDefs object for the compressed binary table

    Returns
    -------
    nbytes : `int`
        The number of bytes of the heap.
    heap : `bytes`
        The bytes of the FITS table heap.
    """
