from .transforms import CurvedTransform
from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['transform_coord_meta_from_wcs', 'WCSWorld2PixelTransform', 'WCSPixel2WorldTransform', 'custom_ucd_coord_meta_mapping']

def custom_ucd_coord_meta_mapping(mapping, *, overwrite: bool = False) -> Generator[None]:
    '''
    A context manager that makes it possible to temporarily add new UCD+ to WCS coordinate
    plot metadata mappings.

    Parameters
    ----------
    mapping : dict
        A dictionary mapping a UCD to coordinate plot metadata.
        Note that custom UCD names have their "custom:" prefix stripped.
    overwrite : bool
        If `True` overwrite existing entries with ``mapping``.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from astropy.visualization.wcsaxes.wcsapi import custom_ucd_coord_meta_mapping
    >>> from astropy.wcs.wcsapi.fitswcs import custom_ctype_to_ucd_mapping
    >>> wcs = WCS(naxis=1)
    >>> wcs.wcs.ctype = ["eggs"]
    >>> wcs.wcs.cunit = ["deg"]
    >>> custom_mapping = {"eggs": "custom:pos.eggs"}
    >>> with custom_ctype_to_ucd_mapping(custom_mapping):
    ...     custom_meta = {
    ...         "pos.eggs": {
    ...             "coord_wrap": 360.0 * u.deg,
    ...             "format_unit": u.arcsec,
    ...             "coord_type": "longitude",
    ...         }
    ...     }
    ...     with custom_ucd_coord_meta_mapping(custom_meta):
    ...        fig = plt.figure()
    ...        ax = fig.add_subplot(111, projection=wcs)
    ...        ax.coords
    <CoordinatesMap with 1 world coordinates:
    <BLANKLINE>
      index       aliases           type   unit  wrap format_unit visible
                                                 deg
      ----- -------------------- --------- ---- ----- ----------- -------
          0 custom:pos.eggs eggs longitude  deg 360.0      arcsec     yes
    <BLANKLINE>
    >
    '''
def transform_coord_meta_from_wcs(wcs, frame_class, slices: Incomplete | None = None): ...

class WCSWorld2PixelTransform(CurvedTransform):
    """
    WCS transformation from world to pixel coordinates.
    """
    has_inverse: bool
    frame_in: Incomplete
    wcs: Incomplete
    invert_xy: Incomplete
    def __init__(self, wcs, invert_xy: bool = False) -> None: ...
    def __eq__(self, other): ...
    @property
    def input_dims(self): ...
    def transform(self, world): ...
    transform_non_affine = transform
    def inverted(self):
        """
        Return the inverse of the transform.
        """

class WCSPixel2WorldTransform(CurvedTransform):
    """
    WCS transformation from pixel to world coordinates.
    """
    has_inverse: bool
    wcs: Incomplete
    invert_xy: Incomplete
    frame_out: Incomplete
    def __init__(self, wcs, invert_xy: bool = False) -> None: ...
    def __eq__(self, other): ...
    @property
    def output_dims(self): ...
    def transform(self, pixel): ...
    transform_non_affine = transform
    def inverted(self):
        """
        Return the inverse of the transform.
        """
