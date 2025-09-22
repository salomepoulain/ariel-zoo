import numpy as np
from _typeshed import Incomplete
from astropy.utils.data_info import ParentDtypeInfo as ParentDtypeInfo

class NdarrayMixinInfo(ParentDtypeInfo):
    _represent_as_dict_primary_data: str
    def _represent_as_dict(self):
        """Represent Column as a dict that can be serialized."""
    def _construct_from_dict(self, map):
        """Construct Column from ``map``."""

class NdarrayMixin(np.ndarray):
    """
    Mixin column class to allow storage of arbitrary numpy
    ndarrays within a Table.  This is a subclass of numpy.ndarray
    and has the same initialization options as ``np.array()``.
    """
    info: Incomplete
    def __new__(cls, obj, *args, **kwargs): ...
    def __array_finalize__(self, obj) -> None: ...
    def __reduce__(self): ...
    def __setstate__(self, state) -> None: ...
