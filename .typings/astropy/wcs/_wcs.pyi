from astropy.wcs import Auxprm as Auxprm, Celprm as Celprm, DistortionLookupTable as DistortionLookupTable, Prjprm as Prjprm, Sip as Sip, Tabprm as Tabprm, Wcsprm as Wcsprm, Wtbarr as Wtbarr, _Wcs as _Wcs
from typing import Any

PRJ_CODES: list
PRJ_CONIC: int
PRJ_CONVENTIONAL: int
PRJ_CYLINDRICAL: int
PRJ_HEALPIX: int
PRJ_POLYCONIC: int
PRJ_PSEUDOCYLINDRICAL: int
PRJ_PVN: int
PRJ_QUADCUBE: int
PRJ_ZENITHAL: int
WCSCOMPARE_ANCILLARY: int
WCSCOMPARE_CRPIX: int
WCSCOMPARE_TILING: int
WCSHDO_CNAMna: int
WCSHDO_CRPXna: int
WCSHDO_DOBSn: int
WCSHDO_EFMT: int
WCSHDO_P12: int
WCSHDO_P13: int
WCSHDO_P14: int
WCSHDO_P15: int
WCSHDO_P16: int
WCSHDO_P17: int
WCSHDO_PVn_ma: int
WCSHDO_TPCn_ka: int
WCSHDO_WCSNna: int
WCSHDO_all: int
WCSHDO_none: int
WCSHDO_safe: int
WCSHDR_ALLIMG: int
WCSHDR_AUXIMG: int
WCSHDR_BIMGARR: int
WCSHDR_CD00i00j: int
WCSHDR_CD0i_0ja: int
WCSHDR_CNAMn: int
WCSHDR_CROTAia: int
WCSHDR_DOBSn: int
WCSHDR_EPOCHa: int
WCSHDR_IMGHEAD: int
WCSHDR_LONGKEY: int
WCSHDR_PC00i00j: int
WCSHDR_PC0i_0ja: int
WCSHDR_PIXLIST: int
WCSHDR_PROJPn: int
WCSHDR_PS0i_0ma: int
WCSHDR_PV0i_0ma: int
WCSHDR_RADECSYS: int
WCSHDR_VELREFa: int
WCSHDR_VSOURCE: int
WCSHDR_all: int
WCSHDR_none: int
WCSHDR_reject: int
WCSHDR_strict: int
WCSSUB_CELESTIAL: int
WCSSUB_CUBEFACE: int
WCSSUB_LATITUDE: int
WCSSUB_LONGITUDE: int
WCSSUB_SPECTRAL: int
WCSSUB_STOKES: int
WCSSUB_TIME: int
_ASTROPY_WCS_API: PyCapsule
__version__: str

class InconsistentAxisTypesError(WcsError): ...

class InvalidCoordinateError(WcsError): ...

class InvalidPrjParametersError(WcsError): ...

class InvalidSubimageSpecificationError(WcsError): ...

class InvalidTabularParametersError(WcsError): ...

class InvalidTransformError(WcsError): ...

class NoSolutionError(WcsError): ...

class NoWcsKeywordsFoundError(WcsError): ...

class NonseparableSubimageCoordinateSystemError(WcsError): ...

class SingularMatrixError(WcsError): ...

class WcsError(ValueError): ...

def _sanity_check(*args, **kwargs): ...
def find_all_wcs(relax=..., keysel=...) -> Any:
    """find_all_wcs(relax=0, keysel=0)

    Find all WCS transformations in the header.

    Parameters
    ----------

    header : str
        The raw FITS header data.

    relax : bool or int
        Degree of permissiveness:

        - `False`: Recognize only FITS keywords defined by the published
          WCS standard.

        - `True`: Admit all recognized informal extensions of the WCS
          standard.

        - `int`: a bit field selecting specific extensions to accept.  See
          :ref:`astropy:relaxread` for details.

    keysel : sequence of flags
        Used to restrict the keyword types considered:

        - ``WCSHDR_IMGHEAD``: Image header keywords.

        - ``WCSHDR_BIMGARR``: Binary table image array.

        - ``WCSHDR_PIXLIST``: Pixel list keywords.

        If zero, there is no restriction.  If -1, `wcspih` is called,
        rather than `wcstbh`.

    Returns
    -------
    wcs_list : list of `~astropy.wcs.Wcsprm`
    """
def set_wtbarr_fitsio_callback(*args, **kwargs): ...
