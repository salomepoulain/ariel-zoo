from _typeshed import Incomplete

__all__ = ['add_beam', 'add_scalebar']

def add_beam(ax, header: Incomplete | None = None, major: Incomplete | None = None, minor: Incomplete | None = None, angle: Incomplete | None = None, corner: str = 'bottom left', frame: bool = False, borderpad: float = 0.4, pad: float = 0.5, **kwargs) -> None:
    """
    Display the beam shape and size.

    Parameters
    ----------
    ax : :class:`~astropy.visualization.wcsaxes.WCSAxes`
        WCSAxes instance in which the beam shape and size is displayed. The WCS
        must be celestial.
    header : :class:`~astropy.io.fits.Header`, optional
        Header containing the beam parameters. If specified, the ``BMAJ``,
        ``BMIN``, and ``BPA`` keywords will be searched in the FITS header
        to set the major and minor axes and the position angle on the sky.
    major : float or :class:`~astropy.units.Quantity`, optional
        Major axis of the beam in degrees or an angular quantity.
    minor : float, or :class:`~astropy.units.Quantity`, optional
        Minor axis of the beam in degrees or an angular quantity.
    angle : float or :class:`~astropy.units.Quantity`, optional
        Position angle of the beam on the sky in degrees or an angular
        quantity in the anticlockwise direction.
    corner : str, optional
        The beam location. Acceptable values are ``'left'``, ``'right'``,
        ``'top'``, 'bottom', ``'top left'``, ``'top right'``, ``'bottom left'``
        (default), and ``'bottom right'``.
    frame : bool, optional
        Whether to display a frame behind the beam (default is ``False``).
    borderpad : float, optional
        Border padding, in fraction of the font size. Default is 0.4.
    pad : float, optional
        Padding around the beam, in fraction of the font size. Default is 0.5.
    kwargs
        Additional arguments are passed to :class:`matplotlib.patches.Ellipse`.

    Notes
    -----
    This function may be inaccurate when:

    - The pixel scales at the reference pixel are different from the pixel scales
      within the image extent (e.g., when the reference pixel is well outside of
      the image extent and the projection is non-linear)
    - The pixel scales in the two directions are very different from each other
      (e.g., rectangular pixels)

    """
def add_scalebar(ax, length, label: Incomplete | None = None, corner: str = 'bottom right', frame: bool = False, borderpad: float = 0.4, pad: float = 0.5, **kwargs) -> None:
    """Add a scale bar.

    Parameters
    ----------
    ax : :class:`~astropy.visualization.wcsaxes.WCSAxes`
        WCSAxes instance in which the scale bar is displayed. The WCS must be
        celestial.
    length : float or :class:`~astropy.units.Quantity`
        The length of the scalebar in degrees or an angular quantity
    label : str, optional
        Label to place below the scale bar
    corner : str, optional
        Where to place the scale bar. Acceptable values are:, ``'left'``,
        ``'right'``, ``'top'``, ``'bottom'``, ``'top left'``, ``'top right'``,
        ``'bottom left'`` and ``'bottom right'`` (default)
    frame : bool, optional
        Whether to display a frame behind the scale bar (default is ``False``)
    borderpad : float, optional
        Border padding, in fraction of the font size. Default is 0.4.
    pad : float, optional
        Padding around the scale bar, in fraction of the font size. Default is 0.5.
    kwargs
        Additional arguments are passed to
        :class:`mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.

    Notes
    -----
    This function may be inaccurate when:

    - The pixel scales at the reference pixel are different from the pixel scales
      within the image extent (e.g., when the reference pixel is well outside of
      the image extent and the projection is non-linear)
    - The pixel scales in the two directions are very different from each other
      (e.g., rectangular pixels)

    """
