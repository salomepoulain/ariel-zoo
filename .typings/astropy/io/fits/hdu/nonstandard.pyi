from .base import NonstandardExtHDU as NonstandardExtHDU
from .hdulist import HDUList as HDUList
from astropy.io.fits.file import _File as _File
from astropy.io.fits.header import Header as Header, _pad_length as _pad_length
from astropy.io.fits.util import fileobj_name as fileobj_name
from astropy.utils import lazyproperty as lazyproperty

class FitsHDU(NonstandardExtHDU):
    """
    A non-standard extension HDU for encapsulating entire FITS files within a
    single HDU of a container FITS file.  These HDUs have an extension (that is
    an XTENSION keyword) of FITS.

    The FITS file contained in the HDU's data can be accessed by the `hdulist`
    attribute which returns the contained FITS file as an `HDUList` object.
    """
    _extension: str
    def hdulist(self): ...
    @classmethod
    def fromfile(cls, filename, compress: bool = False):
        """
        Like `FitsHDU.fromhdulist()`, but creates a FitsHDU from a file on
        disk.

        Parameters
        ----------
        filename : str
            The path to the file to read into a FitsHDU
        compress : bool, optional
            Gzip compress the FITS file
        """
    @classmethod
    def fromhdulist(cls, hdulist, compress: bool = False):
        """
        Creates a new FitsHDU from a given HDUList object.

        Parameters
        ----------
        hdulist : HDUList
            A valid Headerlet object.
        compress : bool, optional
            Gzip compress the FITS file
        """
    @classmethod
    def match_header(cls, header): ...
    def _summary(self): ...
