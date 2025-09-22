from .formats import *
from .core import *
from _typeshed import Incomplete
from astropy import config as _config
from astropy.utils.masked import Masked as Masked

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.time`.
    """
    use_fast_parser: Incomplete
    masked_array_type: Incomplete
    _MASKED_CLASSES: Incomplete
    @property
    def _masked_cls(self):
        '''The masked class set by ``masked_array_type``.

        This is |Masked| for "astropy", `numpy.ma.MaskedArray` for "numpy".
        '''

conf: Incomplete
