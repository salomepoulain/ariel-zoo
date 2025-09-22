from .wcs import *
from . import utils as utils
from .wcs import InvalidTabularParametersError as InvalidTabularParametersError

def get_include():
    """
    Get the path to astropy.wcs's C header files.
    """
