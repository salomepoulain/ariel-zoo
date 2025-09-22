from .exceptions import IOWarning as IOWarning, UnimplementedWarning as UnimplementedWarning, VOTableChangeWarning as VOTableChangeWarning, VOTableSpecError as VOTableSpecError, VOTableSpecWarning as VOTableSpecWarning, VOWarning as VOWarning
from .table import from_table as from_table, is_votable as is_votable, parse as parse, parse_single_table as parse_single_table, validate as validate, writeto as writeto
from _typeshed import Incomplete
from astropy import config as _config

__all__ = ['Conf', 'conf', 'parse', 'parse_single_table', 'validate', 'from_table', 'is_votable', 'writeto', 'VOWarning', 'VOTableChangeWarning', 'VOTableSpecWarning', 'UnimplementedWarning', 'IOWarning', 'VOTableSpecError']

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.io.votable`.
    """
    verify: Incomplete

conf: Incomplete
