from _typeshed import Incomplete

def _searchsorted(array, val, side: str = 'left'):
    """
    Call np.searchsorted or use a custom binary
    search if necessary.
    """

class SortedArray:
    """
    Implements a sorted array container using
    a list of numpy arrays.

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
    data: Incomplete
    row_index: Incomplete
    num_cols: Incomplete
    unique: Incomplete
    def __init__(self, data, row_index, unique: bool = False) -> None: ...
    @property
    def cols(self): ...
    def add(self, key, row) -> None:
        """
        Add a new entry to the sorted array.

        Parameters
        ----------
        key : tuple
            Column values at the given row
        row : int
            Row number
        """
    def _get_key_slice(self, i, begin, end):
        """
        Retrieve the ith slice of the sorted array
        from begin to end.
        """
    def find_pos(self, key, data, exact: bool = False):
        """
        Return the index of the largest key in data greater than or
        equal to the given key, data pair.

        Parameters
        ----------
        key : tuple
            Column key
        data : int
            Row number
        exact : bool
            If True, return the index of the given key in data
            or -1 if the key is not present.
        """
    def find(self, key):
        """
        Find all rows matching the given key.

        Parameters
        ----------
        key : tuple
            Column values

        Returns
        -------
        matching_rows : list
            List of rows matching the input key
        """
    def range(self, lower, upper, bounds):
        """
        Find values in the given range.

        Parameters
        ----------
        lower : tuple
            Lower search bound
        upper : tuple
            Upper search bound
        bounds : (2,) tuple of bool
            Indicates whether the search should be inclusive or
            exclusive with respect to the endpoints. The first
            argument corresponds to an inclusive lower bound,
            and the second argument to an inclusive upper bound.
        """
    def remove(self, key, data):
        """
        Remove the given entry from the sorted array.

        Parameters
        ----------
        key : tuple
            Column values
        data : int
            Row number

        Returns
        -------
        successful : bool
            Whether the entry was successfully removed
        """
    def shift_left(self, row) -> None:
        """
        Decrement all row numbers greater than the input row.

        Parameters
        ----------
        row : int
            Input row number
        """
    def shift_right(self, row) -> None:
        """
        Increment all row numbers greater than or equal to the input row.

        Parameters
        ----------
        row : int
            Input row number
        """
    def replace_rows(self, row_map) -> None:
        """
        Replace all rows with the values they map to in the
        given dictionary. Any rows not present as keys in
        the dictionary will have their entries deleted.

        Parameters
        ----------
        row_map : dict
            Mapping of row numbers to new row numbers
        """
    def items(self):
        """
        Retrieve all array items as a list of pairs of the form
        [(key, [row 1, row 2, ...]), ...].
        """
    def sort(self) -> None:
        """
        Make row order align with key order.
        """
    def sorted_data(self):
        """
        Return rows in sorted order.
        """
    def __getitem__(self, item):
        """
        Return a sliced reference to this sorted array.

        Parameters
        ----------
        item : slice
            Slice to use for referencing
        """
    def __repr__(self) -> str: ...
