from .basic_rgb import RGBImageMapping
from _typeshed import Incomplete
from astropy.visualization.stretch import BaseStretch

__all__ = ['AsinhMapping', 'AsinhZScaleMapping', 'LinearMapping', 'LuptonAsinhStretch', 'LuptonAsinhZscaleStretch', 'Mapping', 'make_lupton_rgb']

class Mapping:
    """
    Baseclass to map red, blue, green intensities into uint8 values.

    Parameters
    ----------
    minimum : float or sequence(3)
        Intensity that should be mapped to black (a scalar or array for R, G, B).
    image : ndarray, optional
        An image used to calculate some parameters of some mappings.
    """
    _uint8Max: Incomplete
    minimum: Incomplete
    _image: Incomplete
    def __init__(self, minimum: Incomplete | None = None, image: Incomplete | None = None) -> None: ...
    def make_rgb_image(self, image_r, image_g, image_b):
        """
        Convert 3 arrays, image_r, image_g, and image_b into an 8-bit RGB image.

        Parameters
        ----------
        image_r : ndarray
            Image to map to red.
        image_g : ndarray
            Image to map to green.
        image_b : ndarray
            Image to map to blue.

        Returns
        -------
        RGBimage : ndarray
            RGB (integer, 8-bits per channel) color image as an NxNx3 numpy array.
        """
    def intensity(self, image_r, image_g, image_b):
        """
        Return the total intensity from the red, blue, and green intensities.
        This is a naive computation, and may be overridden by subclasses.

        Parameters
        ----------
        image_r : ndarray
            Intensity of image to be mapped to red; or total intensity if
            ``image_g`` and ``image_b`` are None.
        image_g : ndarray, optional
            Intensity of image to be mapped to green.
        image_b : ndarray, optional
            Intensity of image to be mapped to blue.

        Returns
        -------
        intensity : ndarray
            Total intensity from the red, blue and green intensities, or
            ``image_r`` if green and blue images are not provided.
        """
    def map_intensity_to_uint8(self, I):
        """
        Return an array which, when multiplied by an image, returns that image
        mapped to the range of a uint8, [0, 255] (but not converted to uint8).

        The intensity is assumed to have had minimum subtracted (as that can be
        done per-band).

        Parameters
        ----------
        I : ndarray
            Intensity to be mapped.

        Returns
        -------
        mapped_I : ndarray
            ``I`` mapped to uint8
        """
    def _convert_images_to_uint8(self, image_r, image_g, image_b):
        """
        Use the mapping to convert images image_r, image_g, and image_b to a triplet of uint8 images.
        """

class LinearMapping(Mapping):
    """
    A linear map map of red, blue, green intensities into uint8 values.

    A linear stretch from [minimum, maximum].
    If one or both are omitted use image min and/or max to set them.

    Parameters
    ----------
    minimum : float
        Intensity that should be mapped to black (a scalar or array for R, G, B).
    maximum : float
        Intensity that should be mapped to white (a scalar).
    """
    maximum: Incomplete
    _range: Incomplete
    def __init__(self, minimum: Incomplete | None = None, maximum: Incomplete | None = None, image: Incomplete | None = None) -> None: ...
    def map_intensity_to_uint8(self, I): ...

class AsinhMapping(Mapping):
    """
    A mapping for an asinh stretch (preserving colours independent of brightness).

    x = asinh(Q (I - minimum)/stretch)/Q

    This reduces to a linear stretch if Q == 0

    See https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L

    Parameters
    ----------
    minimum : float
        Intensity that should be mapped to black (a scalar or array for R, G, B).
    stretch : float
        The linear stretch of the image.
    Q : float
        The asinh softening parameter.
    """
    _slope: Incomplete
    _soften: Incomplete
    def __init__(self, minimum, stretch, Q: int = 8) -> None: ...
    def map_intensity_to_uint8(self, I): ...

class AsinhZScaleMapping(AsinhMapping):
    """
    A mapping for an asinh stretch, estimating the linear stretch by zscale.

    x = asinh(Q (I - z1)/(z2 - z1))/Q

    Parameters
    ----------
    image1 : ndarray or a list of arrays
        The image to analyse, or a list of 3 images to be converted to
        an intensity image.
    image2 : ndarray, optional
        the second image to analyse (must be specified with image3).
    image3 : ndarray, optional
        the third image to analyse (must be specified with image2).
    Q : float, optional
        The asinh softening parameter. Default is 8.
    pedestal : float or sequence(3), optional
        The value, or array of 3 values, to subtract from the images; or None.

    Notes
    -----
    pedestal, if not None, is removed from the images when calculating the
    zscale stretch, and added back into Mapping.minimum[]
    """
    _image: Incomplete
    def __init__(self, image1, image2: Incomplete | None = None, image3: Incomplete | None = None, Q: int = 8, pedestal: Incomplete | None = None) -> None: ...

class LuptonAsinhStretch(BaseStretch):
    """
    A modified asinh stretch, with some changes to the constants
    relative to `~astropy.visualization.AsinhStretch`.

    The stretch is given by:

    .. math::
        & y = {\\rm asinh}\\left(\\frac{Q * x}{stretch}\\right) *
        \\frac{frac}{{\\rm asinh}(frac * Q)} \\\\\n        & frac = 0.1

    Parameters
    ----------
    stretch : float, optional
        Linear stretch of the image. ``stretch`` must be greater than 0.
        Default is 5.

    Q : float, optional
        The asinh softening parameter. ``Q`` must be greater than 0.
        Default is 8.

    Notes
    -----
    Based on the asinh stretch presented in Lupton et al. 2004
    (https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L).

    """
    stretch: Incomplete
    Q: Incomplete
    _slope: Incomplete
    _soften: Incomplete
    def __init__(self, stretch: int = 5, Q: int = 8) -> None: ...
    def __call__(self, values, clip: bool = False, out: Incomplete | None = None): ...

class LuptonAsinhZscaleStretch(LuptonAsinhStretch):
    """
    A modified asinh stretch, where the linear stretch is calculated using
    zscale.

    The stretch is given by:

    .. math::
        & y = {\\rm asinh}\\left(\\frac{Q * x}{stretch}\\right) *
        \\frac{frac}{{\\rm asinh}(frac * Q)} \\\\\n        & frac = 0.1 \\\\\n        & stretch = z2 - z1

    Parameters
    ----------
    image1 : ndarray or array-like
        The image to analyse, or a list of 3 images to be converted to
        an intensity image.

    Q : float, optional
        The asinh softening parameter. ``Q`` must be greater than 0.
        Default is 8.

    pedestal : or array-like, optional
        The value, or array of 3 values, to subtract from the images(s)
        before determining the zscaling. Default is None (nothing subtracted).

    """
    _image: Incomplete
    def __init__(self, image, Q: int = 8, pedestal: Incomplete | None = None) -> None: ...

class RGBImageMappingLupton(RGBImageMapping):
    """
    Class to map red, blue, green images into either a normalized float or
    an 8-bit image, by performing optional clipping and applying
    a scaling function to each band in non-independent manner that depends
    on the other bands, following the scaling scheme presented in
    Lupton et al. 2004.

    Parameters
    ----------
    interval : `~astropy.visualization.BaseInterval` subclass instance or array-like, optional
        The interval object to apply to the data (either a single instance or
        an array for R, G, B). Default is
        `~astropy.visualization.ManualInterval`.
    stretch : `~astropy.visualization.BaseStretch` subclass instance
        The stretch object to apply to the data. The default is
        `~astropy.visualization.AsinhLuptonStretch`.

    """
    _pixmax: float
    def __init__(self, interval=..., stretch=...) -> None: ...
    def intensity(self, image_r, image_g, image_b):
        """
        Return the total intensity from the red, blue, and green intensities.
        This is a naive computation, and may be overridden by subclasses.

        Parameters
        ----------
        image_r : ndarray
            Intensity of image to be mapped to red; or total intensity if
            ``image_g`` and ``image_b`` are None.
        image_g : ndarray, optional
            Intensity of image to be mapped to green.
        image_b : ndarray, optional
            Intensity of image to be mapped to blue.

        Returns
        -------
        intensity : ndarray
            Total intensity from the red, blue and green intensities, or
            ``image_r`` if green and blue images are not provided.

        """
    def apply_mappings(self, image_r, image_g, image_b):
        """
        Apply mapping stretch and intervals to convert images image_r, image_g,
        and image_b to a triplet of normalized images, following the scaling
        scheme presented in Lupton et al. 2004.

        Compared to astropy's ImageNormalize which first normalizes images
        by cropping and linearly mapping onto [0.,1.] and then applies
        a specified stretch algorithm, the Lupton et al. algorithm applies
        stretching to an multi-color intensity and then computes per-band
        scaled images with bound cropping.

        This is modified here by allowing for different minimum values
        for each of the input r, g, b images, and then computing
        the intensity on the subtracted images.

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

        Notes
        -----
        The Lupton et al 2004 algorithm is computed with the following steps:

        1. Shift each band with the minimum values
        2. Compute the intensity I and stretched intensity f(I)
        3. Compute the ratio of the stretched intensity to intensity f(I)/I,
        and clip to a lower bound of 0
        4. Compute the scaled band images by multiplying with the ratio f(I)/I
        5. Clip each band to a lower bound of 0
        6. Scale down pixels where max(R,G,B)>1 by the value max(R,G,B)

        """

def make_lupton_rgb(image_r, image_g, image_b, interval: Incomplete | None = None, stretch_object: Incomplete | None = None, minimum: Incomplete | None = None, stretch: int = 5, Q: int = 8, filename: Incomplete | None = None, output_dtype=...):
    """
    Return a Red/Green/Blue color image from 3 images using interconnected
    band scaling, and an arbitrary stretch function (by default, an asinh stretch).
    The input images can be int or float, and in any range or bit-depth.

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
    stretch_object : `~astropy.visualization.BaseStretch` subclass instance, optional
        The stretch object to apply to the data. If set, the input values of
        ``minimum``, ``stretch``, and ``Q`` will be ignored.
        For the Lupton scheme, this would be an instance of
        `~astropy.visualization.LuptonAsinhStretch`, but alternatively
        `~astropy.visualization.LuptonAsinhZscaleStretch` or some other
        stretch can be used.
    minimum : float or array-like, optional
        Deprecated. Intensity that should be mapped to black (a scalar or
        array of R, G, B). If `None`, each image's minimum value is used.
        Default is None.
    stretch : float, optional
        The linear stretch of the image. Default is 5
    Q : float, optional
        The asinh softening parameter. Default is 8.
    filename : str, optional
        Write the resulting RGB image to a file (file type determined
        from extension).
    output_dtype : numpy scalar type, optional
        Image output data type. Default is np.uint8.

    Returns
    -------
    rgb : ndarray
        RGB color image as an NxNx3 numpy array, with the specified
        data type format

    """
