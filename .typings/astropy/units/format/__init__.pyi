from .base import Base as Base
from .cds import CDS as CDS
from .console import Console as Console
from .fits import FITS as FITS
from .generic import Generic as Generic
from .latex import Latex as Latex, LatexInline as LatexInline
from .ogip import OGIP as OGIP
from .unicode_format import Unicode as Unicode
from .vounit import VOUnit as VOUnit
from _typeshed import Incomplete

__all__ = ['Base', 'Generic', 'CDS', 'Console', 'FITS', 'Latex', 'LatexInline', 'OGIP', 'Unicode', 'VOUnit', 'get_format']

def get_format(format: Incomplete | None = None):
    """
    Get a formatter by name.

    Parameters
    ----------
    format : str or `astropy.units.format.Base` subclass
        The name of the format, or the formatter class itself.

    Returns
    -------
    format : `astropy.units.format.Base` subclass
        The requested formatter.
    """
