from .client import *
from .constants import *
from .errors import *
from .hub import *
from .hub_proxy import *
from .integrated_client import *
from .utils import *
from _typeshed import Incomplete
from astropy import config as _config

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.samp`.
    """
    use_internet: Incomplete
    n_retries: Incomplete

conf: Incomplete
