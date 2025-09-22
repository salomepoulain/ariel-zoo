from _typeshed import Incomplete
from astropy.utils.compat.optional_deps import HAS_SORTEDCONTAINERS as HAS_SORTEDCONTAINERS

class Node:
    __slots__: Incomplete
    key: Incomplete
    value: Incomplete
    def __init__(self, key, value) -> None: ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    __hash__: Incomplete
    def __repr__(self) -> str: ...

class SCEngine:
    """
    Fast tree-based implementation for indexing, using the
    ``sortedcontainers`` package.

    Parameters
    ----------
    data : Table
        Sorted columns of the original table
    row_index : Column object
        Row numbers corresponding to data columns
    unique : bool
        Whether the values of the index must be unique.
        Defaults to False.
    """
    _nodes: Incomplete
    _unique: Incomplete
    def __init__(self, data, row_index, unique: bool = False) -> None: ...
    def add(self, key, value) -> None:
        """
        Add a key, value pair.
        """
    def find(self, key):
        """
        Find rows corresponding to the given key.
        """
    def remove(self, key, data: Incomplete | None = None):
        """
        Remove data from the given key.
        """
    def shift_left(self, row) -> None:
        """
        Decrement rows larger than the given row.
        """
    def shift_right(self, row) -> None:
        """
        Increment rows greater than or equal to the given row.
        """
    def items(self):
        """
        Return a list of key, data tuples.
        """
    def sort(self) -> None:
        """
        Make row order align with key order.
        """
    def sorted_data(self):
        """
        Return a list of rows in order sorted by key.
        """
    def range(self, lower, upper, bounds=(True, True)):
        """
        Return row values in the given range.
        """
    def replace_rows(self, row_map) -> None:
        """
        Replace rows with the values in row_map.
        """
    def __repr__(self) -> str: ...
