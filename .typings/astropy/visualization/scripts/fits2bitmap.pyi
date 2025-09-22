from _typeshed import Incomplete

__all__ = ['fits2bitmap', 'main']

def fits2bitmap(filename, ext: int = 0, out_fn: Incomplete | None = None, stretch: str = 'linear', power: float = 1.0, asinh_a: float = 0.1, vmin: Incomplete | None = None, vmax: Incomplete | None = None, min_percent: Incomplete | None = None, max_percent: Incomplete | None = None, percent: Incomplete | None = None, cmap: str = 'Greys_r'):
    """
    Create a bitmap file from a FITS image, applying a stretching
    transform between minimum and maximum cut levels and a matplotlib
    colormap.

    Parameters
    ----------
    filename : str | PathLike
        The filename of the FITS file.
    ext : int
        FITS extension name or number of the image to convert. The
        default is 0.
    out_fn : str | PathLike
        The filename of the output bitmap image. The type of bitmap is
        determined by the filename extension (e.g. '.jpg', '.png'). The
        default is a PNG file with the same name as the FITS file.
    stretch : {'linear', 'sqrt', 'power', log', 'asinh'}
        The stretching function to apply to the image. The default is
        'linear'.
    power : float, optional
        The power index for ``stretch='power'``. The default is 1.0.
    asinh_a : float, optional
        For ``stretch='asinh'``, the value where the asinh curve
        transitions from linear to logarithmic behavior, expressed as a
        fraction of the normalized image. Must be in the range between 0
        and 1. The default is 0.1.
    vmin : float, optional
        The pixel value of the minimum cut level. Data values less
        than ``vmin`` will set to ``vmin`` before stretching the
        image. The default is the image minimum. ``vmin`` overrides
        ``min_percent``.
    vmax : float, optional
        The pixel value of the maximum cut level. Data values greater
        than ``vmax`` will set to ``vmax`` before stretching the
        image. The default is the image maximum. ``vmax`` overrides
        ``max_percent``.
    min_percent : float, optional
        The percentile value used to determine the pixel value of
        minimum cut level. The default is 0.0. ``min_percent`` overrides
        ``percent``.
    max_percent : float, optional
        The percentile value used to determine the pixel value of
        maximum cut level. The default is 100.0. ``max_percent``
        overrides ``percent``.
    percent : float, optional
        The percentage of the image values used to determine the pixel
        values of the minimum and maximum cut levels. The lower cut
        level will set at the ``(100 - percent) / 2`` percentile, while
        the upper cut level will be set at the ``(100 + percent) / 2``
        percentile. The default is 100.0. ``percent`` is ignored if
        either ``min_percent`` or ``max_percent`` is input.
    cmap : str
        The matplotlib color map name. The default is 'Greys_r'.
    """
def main(args: Incomplete | None = None) -> None: ...
