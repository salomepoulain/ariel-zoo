from _typeshed import Incomplete

class HomogeneousList(list):
    """
    A subclass of list that contains only elements of a given type or
    types.  If an item that is not of the specified type is added to
    the list, a `TypeError` is raised.
    """
    _types: Incomplete
    def __init__(self, types, values=[]) -> None:
        """
        Parameters
        ----------
        types : sequence of types
            The types to accept.

        values : sequence, optional
            An initial set of values.
        """
    def _assert(self, x) -> None: ...
    def __iadd__(self, other): ...
    def __setitem__(self, idx, value) -> None: ...
    def append(self, x): ...
    def insert(self, i, x): ...
    def extend(self, x) -> None: ...
