from .card import *
from .column import *
from .convenience import *
from .diff import *
from .hdu import *
from .fitsrec import FITS_rec as FITS_rec, FITS_record as FITS_record
from .hdu.groups import GroupData as GroupData
from .hdu.hdulist import fitsopen as open
from .hdu.image import Section as Section
from .header import Header as Header
from .verify import VerifyError as VerifyError
from _typeshed import Incomplete
from astropy import config as _config

__all__ = ['Conf', 'conf', 'Card', 'Undefined', 'Column', 'ColDefs', 'Delayed', 'getheader', 'getdata', 'getval', 'setval', 'delval', 'writeto', 'append', 'update', 'info', 'tabledump', 'tableload', 'table_to_hdu', 'printdiff', 'HDUList', 'PrimaryHDU', 'ImageHDU', 'TableHDU', 'BinTableHDU', 'GroupsHDU', 'GroupData', 'Group', 'CompImageHDU', 'FitsHDU', 'StreamingHDU', 'register_hdu', 'unregister_hdu', 'DELAYED', 'BITPIX2DTYPE', 'DTYPE2BITPIX', 'FITS_record', 'FITS_rec', 'GroupData', 'open', 'Section', 'Header', 'VerifyError', 'conf']

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.io.fits`.
    """
    enable_record_valued_keyword_cards: Incomplete
    extension_name_case_sensitive: Incomplete
    strip_header_whitespace: Incomplete
    use_memmap: Incomplete
    lazy_load_hdus: Incomplete
    enable_uint: Incomplete

conf: Incomplete

# Names in __all__ with no definition:
#   BITPIX2DTYPE
#   BinTableHDU
#   Card
#   ColDefs
#   Column
#   CompImageHDU
#   DELAYED
#   DTYPE2BITPIX
#   Delayed
#   FitsHDU
#   Group
#   GroupsHDU
#   HDUList
#   ImageHDU
#   PrimaryHDU
#   StreamingHDU
#   TableHDU
#   Undefined
#   append
#   delval
#   getdata
#   getheader
#   getval
#   info
#   printdiff
#   register_hdu
#   setval
#   table_to_hdu
#   tabledump
#   tableload
#   unregister_hdu
#   update
#   writeto
