from _typeshed import Incomplete

__all__ = ['make_rgb']

class RGBImageMapping:
    """
    Class to map red, blue, green images into either a normalized float or
    an 8-bit image, by performing optional clipping and applying
    a scaling function to each band independently.

    Parameters
    ----------
    interval : `~astropy.visualization.BaseInterval` subclass instance or array-like, optional
        The interval object to apply to the data (either a single instance or
        an array for R, G, B). Default is
        `~astropy.visualization.ManualInterval`.
    stretch : `~astropy.visualization.BaseStretch` subclass instance, optional
        The stretch object to apply to the data. Default is
        `~astropy.visualization.LinearStretch`.

    """
    intervals: Incomplete
    stretch: Incomplete
    def __init__(self, interval=..., stretch=...) -> None: ...
    def make_rgb_image(self, image_r, image_g, image_b, output_dtype=...):
        """
        Convert 3 arrays, image_r, image_g, and image_b into a RGB image,
        either as an 8-bit per-channel or normalized image.

        The input images can be int or float, and in any range or bit-depth,
        but must have the same shape (NxM).

        Parameters
        ----------
        image_r : ndarray
            Image to map to red.
        image_g : ndarray
            Image to map to green.
        image_b : ndarray
            Image to map to blue.
        output_dtype : numpy scalar type, optional
            Image output format. Default is np.uint8.

        Returns
        -------
        RGBimage : ndarray
            RGB color image with the specified format as an NxMx3 numpy array.

        """
    def apply_mappings(self, image_r, image_g, image_b):
        """
        Apply mapping stretch and intervals to convert images image_r, image_g,
        and image_b to a triplet of normalized images.

        Parameters
        ----------
        image_r : ndarray
            Intensity of image to be mapped to red
        image_g : ndarray
            Intensity of image to be mapped to green.
        image_b : ndarray
            Intensity of image to be mapped to blue.

        Returns
        -------
        image_rgb : ndarray
            Triplet of mapped images based on the specified (per-band)
            intervals and the stretch function

        """
    def _convert_images_to_float(self, image_rgb, output_dtype):
        """
        Convert a triplet of normalized images to float.
        """
    def _convert_images_to_uint(self, image_rgb, output_dtype):
        """
        Convert a triplet of normalized images to unsigned integer images
        """

def make_rgb(image_r, image_g, image_b, interval=..., stretch=..., filename: Incomplete | None = None, output_dtype=...):
    """
    Base class to return a Red/Green/Blue color image from 3 images using
    a specified stretch and interval, for each band *independently*.

    The input images can be int or float, and in any range or bit-depth,
    but must have the same shape (NxM).

    For a more detailed look at the use of this method, see the document
    :ref:`astropy:astropy-visualization-rgb`.

    Parameters
    ----------
    image_r : ndarray
        Image to map to red.
    image_g : ndarray
        Image to map to green.
    image_b : ndarray
        Image to map to blue.
    interval : `~astropy.visualization.BaseInterval` subclass instance or array-like, optional
        The interval object to apply to the data (either a single instance or
        an array for R, G, B). Default is
        `~astropy.visualization.ManualInterval` with vmin=0.
    stretch : `~astropy.visualization.BaseStretch` subclass instance, optional
        The stretch object to apply to the data. Default is
        `~astropy.visualization.LinearStretch`.
    filename : str, optional
        Write the resulting RGB image to a file (file type determined
        from extension).
    output_dtype : numpy scalar type, optional
        Image output data type. Default is np.uint8.

    Returns
    -------
    rgb : ndarray
        RGB (either float or integer with 8-bits per channel) color image
        as an NxMx3 numpy array.

    Notes
    -----
    This procedure of clipping and then scaling is similar to the DS9
    image algorithm (see the DS9 reference guide:
    http://ds9.si.edu/doc/ref/how.html).

    """
