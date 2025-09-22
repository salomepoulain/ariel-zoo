from _typeshed import Incomplete
from astropy import __version__ as __version__
from astropy.io import fits as fits

log: Incomplete
DESCRIPTION: Incomplete

def handle_options(args): ...
def setup_logging() -> None: ...
def verify_checksums(filename):
    """
    Prints a message if any HDU in `filename` has a bad checksum or datasum.
    """
def verify_compliance(filename):
    """Check for FITS standard compliance."""
def update(filename) -> None:
    """
    Sets the ``CHECKSUM`` and ``DATASUM`` keywords for each HDU of `filename`.

    Also updates fixes standards violations if possible and requested.
    """
def process_file(filename):
    """
    Handle a single .fits file,  returning the count of checksum and compliance
    errors.
    """
def main(args: Incomplete | None = None):
    """
    Processes command line parameters into options and files,  then checks
    or update FITS DATASUM and CHECKSUM keywords for the specified files.
    """
