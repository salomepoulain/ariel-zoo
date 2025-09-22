from _typeshed import Incomplete
from astropy.table import QTable
from collections.abc import Generator

__all__ = ['BaseTimeSeries', 'autocheck_required_columns']

def autocheck_required_columns(cls):
    """
    This is a decorator that ensures that the table contains specific
    methods indicated by the _required_columns attribute. The aim is to
    decorate all methods that might affect the columns in the table and check
    for consistency after the methods have been run.
    """

class BaseTimeSeries(QTable):
    _required_columns: Incomplete
    _required_columns_enabled: bool
    _required_columns_relax: bool
    def _check_required_columns(self): ...
    def _delay_required_column_checks(self) -> Generator[None]: ...
