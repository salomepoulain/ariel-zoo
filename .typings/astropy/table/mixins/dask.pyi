import dask.array as da
from _typeshed import Incomplete
from astropy.utils.data_info import ParentDtypeInfo

__all__ = ['as_dask_column']

class DaskInfo(ParentDtypeInfo):
    @staticmethod
    def default_format(val): ...

class DaskColumn(da.Array):
    info: Incomplete
    def copy(self): ...
    def __getitem__(self, item): ...
    def insert(self, obj, values, axis: int = 0): ...

def as_dask_column(array, info: Incomplete | None = None): ...
