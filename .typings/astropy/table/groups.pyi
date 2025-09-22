from _typeshed import Incomplete

__all__ = ['TableGroups', 'ColumnGroups']

class BaseGroups:
    """
    A class to represent groups within a table of heterogeneous data.

      - ``keys``: key values corresponding to each group
      - ``indices``: index values in parent table or column corresponding to group boundaries
      - ``aggregate()``: method to create new table by aggregating within groups
    """
    @property
    def parent(self): ...
    _iter_index: int
    def __iter__(self): ...
    def next(self): ...
    __next__ = next
    def __getitem__(self, item): ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

class ColumnGroups(BaseGroups):
    parent_column: Incomplete
    parent_table: Incomplete
    _indices: Incomplete
    _keys: Incomplete
    def __init__(self, parent_column, indices: Incomplete | None = None, keys: Incomplete | None = None) -> None: ...
    @property
    def indices(self): ...
    @property
    def keys(self): ...
    def aggregate(self, func): ...
    def filter(self, func):
        """
        Filter groups in the Column based on evaluating function ``func`` on each
        group sub-table.

        The function which is passed to this method must accept one argument:

        - ``column`` : `Column` object

        It must then return either `True` or `False`.  As an example, the following
        will select all column groups with only positive values::

          def all_positive(column):
              if np.any(column < 0):
                  return False
              return True

        Parameters
        ----------
        func : function
            Filter function

        Returns
        -------
        out : Column
            New column with the aggregated rows.
        """

class TableGroups(BaseGroups):
    parent_table: Incomplete
    _indices: Incomplete
    _keys: Incomplete
    def __init__(self, parent_table, indices: Incomplete | None = None, keys: Incomplete | None = None) -> None: ...
    @property
    def key_colnames(self):
        """
        Return the names of columns in the parent table that were used for grouping.
        """
    @property
    def indices(self): ...
    def aggregate(self, func):
        """
        Aggregate each group in the Table into a single row by applying the reduction
        function ``func`` to group values in each column.

        Parameters
        ----------
        func : function
            Function that reduces an array of values to a single value

        Returns
        -------
        out : Table
            New table with the aggregated rows.
        """
    def filter(self, func):
        """
        Filter groups in the Table based on evaluating function ``func`` on each
        group sub-table.

        The function which is passed to this method must accept two arguments:

        - ``table`` : `Table` object
        - ``key_colnames`` : tuple of column names in ``table`` used as keys for grouping

        It must then return either `True` or `False`.  As an example, the following
        will select all table groups with only positive values in the non-key columns::

          def all_positive(table, key_colnames):
              colnames = [name for name in table.colnames if name not in key_colnames]
              for colname in colnames:
                  if np.any(table[colname] < 0):
                      return False
              return True

        Parameters
        ----------
        func : function
            Filter function

        Returns
        -------
        out : Table
            New table with the aggregated rows.
        """
    @property
    def keys(self): ...
