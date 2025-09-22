from _typeshed import Incomplete
from astropy import __version__ as __version__, log as log
from astropy.io import fits as fits
from astropy.io.fits import CompImageHDU as CompImageHDU

DESCRIPTION: Incomplete

class ExtensionNotFoundException(Exception):
    """Raised if an HDU extension requested by the user does not exist."""

class HeaderFormatter:
    """Class to format the header(s) of a FITS file for display by the
    `fitsheader` tool; essentially a wrapper around a `HDUList` object.

    Example usage:
    fmt = HeaderFormatter('/path/to/file.fits')
    print(fmt.parse(extensions=[0, 3], keywords=['NAXIS', 'BITPIX']))

    Parameters
    ----------
    filename : str
        Path to a single FITS file.
    verbose : bool
        Verbose flag, to show more information about missing extensions,
        keywords, etc.

    Raises
    ------
    OSError
        If `filename` does not exist or cannot be read.
    """
    filename: Incomplete
    verbose: Incomplete
    _hdulist: Incomplete
    def __init__(self, filename, verbose: bool = True) -> None: ...
    def parse(self, extensions: Incomplete | None = None, keywords: Incomplete | None = None, compressed: bool = False):
        '''Returns the FITS file header(s) in a readable format.

        Parameters
        ----------
        extensions : list of int or str, optional
            Format only specific HDU(s), identified by number or name.
            The name can be composed of the "EXTNAME" or "EXTNAME,EXTVER"
            keywords.

        keywords : list of str, optional
            Keywords for which the value(s) should be returned.
            If not specified, then the entire header is returned.

        compressed : bool, optional
            If True, shows the header describing the compression, rather than
            the header obtained after decompression. (Affects FITS files
            containing `CompImageHDU` extensions only.)

        Returns
        -------
        formatted_header : str or astropy.table.Table
            Traditional 80-char wide format in the case of `HeaderFormatter`;
            an Astropy Table object in the case of `TableHeaderFormatter`.
        '''
    def _parse_internal(self, hdukeys, keywords, compressed):
        """The meat of the formatting; in a separate method to allow overriding."""
    def _get_cards(self, hdukey, keywords, compressed):
        """Returns a list of `astropy.io.fits.card.Card` objects.

        This function will return the desired header cards, taking into
        account the user's preference to see the compressed or uncompressed
        version.

        Parameters
        ----------
        hdukey : int or str
            Key of a single HDU in the HDUList.

        keywords : list of str, optional
            Keywords for which the cards should be returned.

        compressed : bool, optional
            If True, shows the header describing the compression.

        Raises
        ------
        ExtensionNotFoundException
            If the hdukey does not correspond to an extension.
        """
    def close(self) -> None: ...

class TableHeaderFormatter(HeaderFormatter):
    """Class to convert the header(s) of a FITS file into a Table object.
    The table returned by the `parse` method will contain four columns:
    filename, hdu, keyword, and value.

    Subclassed from HeaderFormatter, which contains the meat of the formatting.
    """
    def _parse_internal(self, hdukeys, keywords, compressed):
        """Method called by the parse method in the parent class."""

def print_headers_traditional(args) -> None:
    """Prints FITS header(s) using the traditional 80-char format.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed from the command-line as defined below.
    """
def print_headers_as_table(args):
    """Prints FITS header(s) in a machine-readable table format.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed from the command-line as defined below.
    """
def print_headers_as_comparison(args):
    """Prints FITS header(s) with keywords as columns.

    This follows the dfits+fitsort format.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed from the command-line as defined below.
    """
def main(args: Incomplete | None = None) -> None:
    """This is the main function called by the `fitsheader` script."""
