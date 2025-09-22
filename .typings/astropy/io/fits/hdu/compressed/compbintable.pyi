from astropy.io.fits.hdu.table import BinTableHDU

__all__ = ['_CompBinTableHDU']

class _CompBinTableHDU(BinTableHDU):
    _load_variable_length_data: bool
    _manages_own_heap: bool
    @classmethod
    def match_header(cls, header): ...
