from _typeshed import Incomplete

__all__ = ['FlagCollection']

class FlagCollection(dict):
    """
    The purpose of this class is to provide a dictionary for
    containing arrays of flags for the `NDData` class. Flags should be
    stored in Numpy arrays that have the same dimensions as the parent
    data, so the `FlagCollection` class adds shape checking to a
    dictionary.

    The `FlagCollection` should be initialized like a
    dict, but with the addition of a ``shape=``
    keyword argument used to pass the NDData shape.
    """
    shape: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def __setitem__(self, item, value, **kwargs) -> None: ...
