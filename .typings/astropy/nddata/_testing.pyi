from astropy.wcs import WCS as WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS as BaseHighLevelWCS

def assert_wcs_seem_equal(wcs1, wcs2) -> None:
    """Just checks a few attributes to make sure wcs instances seem to be
    equal.
    """
def _create_wcs_simple(naxis, ctype, crpix, crval, cdelt): ...
def create_two_equal_wcs(naxis): ...
def create_two_unequal_wcs(naxis): ...
