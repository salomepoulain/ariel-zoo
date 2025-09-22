from _typeshed import Incomplete

__all__ = ['join', 'setdiff', 'hstack', 'vstack', 'unique', 'join_skycoord', 'join_distance']

def join_skycoord(distance, distance_func: str = 'search_around_sky'):
    """Helper function to join on SkyCoord columns using distance matching.

    This function is intended for use in ``table.join()`` to allow performing a
    table join where the key columns are both ``SkyCoord`` objects, matched by
    computing the distance between points and accepting values below
    ``distance``.

    The distance cross-matching is done using either
    `~astropy.coordinates.search_around_sky` or
    `~astropy.coordinates.search_around_3d`, depending on the value of
    ``distance_func``.  The default is ``'search_around_sky'``.

    One can also provide a function object for ``distance_func``, in which case
    it must be a function that follows the same input and output API as
    `~astropy.coordinates.search_around_sky`. In this case the function will
    be called with ``(skycoord1, skycoord2, distance)`` as arguments.

    Parameters
    ----------
    distance : `~astropy.units.Quantity` ['angle', 'length']
        Maximum distance between points to be considered a join match.
        Must have angular or distance units.
    distance_func : str or function
        Specifies the function for performing the cross-match based on
        ``distance``. If supplied as a string this specifies the name of a
        function in `astropy.coordinates`. If supplied as a function then that
        function is called directly.

    Returns
    -------
    join_func : function
        Function that accepts two ``SkyCoord`` columns (col1, col2) and returns
        the tuple (ids1, ids2) of pair-matched unique identifiers.

    Examples
    --------
    This example shows an inner join of two ``SkyCoord`` columns, taking any
    sources within 0.2 deg to be a match.  Note the new ``sc_id`` column which
    is added and provides a unique source identifier for the matches.

      >>> from astropy.coordinates import SkyCoord
      >>> import astropy.units as u
      >>> from astropy.table import Table, join_skycoord
      >>> from astropy import table

      >>> sc1 = SkyCoord([0, 1, 1.1, 2], [0, 0, 0, 0], unit='deg')
      >>> sc2 = SkyCoord([0.5, 1.05, 2.1], [0, 0, 0], unit='deg')

      >>> join_func = join_skycoord(0.2 * u.deg)
      >>> join_func(sc1, sc2)  # Associate each coordinate with unique source ID
      (array([3, 1, 1, 2]), array([4, 1, 2]))

      >>> t1 = Table([sc1], names=['sc'])
      >>> t2 = Table([sc2], names=['sc'])
      >>> t12 = table.join(t1, t2, join_funcs={'sc': join_skycoord(0.2 * u.deg)})
      >>> print(t12)  # Note new `sc_id` column with the IDs from join_func()
      sc_id   sc_1    sc_2
            deg,deg deg,deg
      ----- ------- --------
          1 1.0,0.0 1.05,0.0
          1 1.1,0.0 1.05,0.0
          2 2.0,0.0  2.1,0.0

    """
def join_distance(distance, kdtree_args: Incomplete | None = None, query_args: Incomplete | None = None):
    '''Helper function to join table columns using distance matching.

    This function is intended for use in ``table.join()`` to allow performing
    a table join where the key columns are matched by computing the distance
    between points and accepting values below ``distance``. This numerical
    "fuzzy" match can apply to 1-D or 2-D columns, where in the latter case
    the distance is a vector distance.

    The distance cross-matching is done using `scipy.spatial.KDTree`. If
    necessary you can tweak the default behavior by providing ``dict`` values
    for the ``kdtree_args`` or ``query_args``.

    Parameters
    ----------
    distance : float or `~astropy.units.Quantity` [\'length\']
        Maximum distance between points to be considered a join match
    kdtree_args : dict, None
        Optional extra args for `~scipy.spatial.KDTree`
    query_args : dict, None
        Optional extra args for `~scipy.spatial.KDTree.query_ball_tree`

    Returns
    -------
    join_func : function
        Function that accepts (skycoord1, skycoord2) and returns the tuple
        (ids1, ids2) of pair-matched unique identifiers.

    Examples
    --------
      >>> from astropy.table import Table, join_distance
      >>> from astropy import table

      >>> c1 = [0, 1, 1.1, 2]
      >>> c2 = [0.5, 1.05, 2.1]

      >>> t1 = Table([c1], names=[\'col\'])
      >>> t2 = Table([c2], names=[\'col\'])
      >>> t12 = table.join(t1, t2, join_type=\'outer\', join_funcs={\'col\': join_distance(0.2)})
      >>> print(t12)
      col_id col_1 col_2
      ------ ----- -----
           1   1.0  1.05
           1   1.1  1.05
           2   2.0   2.1
           3   0.0    --
           4    --   0.5

    '''
def join(left, right, keys: Incomplete | None = None, join_type: str = 'inner', *, keys_left: Incomplete | None = None, keys_right: Incomplete | None = None, keep_order: bool = False, uniq_col_name: str = '{col_name}_{table_name}', table_names=['1', '2'], metadata_conflicts: str = 'warn', join_funcs: Incomplete | None = None):
    '''
    Perform a join of the left table with the right table on specified keys.

    Parameters
    ----------
    left : `~astropy.table.Table`-like object
        Left side table in the join. If not a Table, will call ``Table(left)``
    right : `~astropy.table.Table`-like object
        Right side table in the join. If not a Table, will call ``Table(right)``
    keys : str or list of str
        Name(s) of column(s) used to match rows of left and right tables.
        Default is to use all columns which are common to both tables.
    join_type : str
        Join type (\'inner\' | \'outer\' | \'left\' | \'right\' | \'cartesian\'), default is \'inner\'
    keys_left : str or list of str or list of column-like, optional
        Left column(s) used to match rows instead of ``keys`` arg. This can be
        be a single left table column name or list of column names, or a list of
        column-like values with the same lengths as the left table.
    keys_right : str or list of str or list of column-like, optional
        Same as ``keys_left``, but for the right side of the join.
    keep_order: bool, optional
        By default, rows are sorted by the join keys. If True, preserve the order of
        rows from the left table for "inner" or "left" joins, or from the right table
        for "right" joins. For other join types this argument is ignored except that a
        warning is issued if ``keep_order=True``.
    uniq_col_name : str or None
        String generate a unique output column name in case of a conflict.
        The default is \'{col_name}_{table_name}\'.
    table_names : list of str or None
        Two-element list of table names used when generating unique output
        column names.  The default is [\'1\', \'2\'].
    metadata_conflicts : str
        How to proceed with metadata conflicts. This should be one of:
            * ``\'silent\'``: silently pick the last conflicting meta-data value
            * ``\'warn\'``: pick the last conflicting meta-data value, but emit a warning (default)
            * ``\'error\'``: raise an exception.
    join_funcs : dict, None
        Dict of functions to use for matching the corresponding key column(s).
        See `~astropy.table.join_skycoord` for an example and details.

    Returns
    -------
    joined_table : `~astropy.table.Table` object
        New table containing the result of the join operation.
    '''
def setdiff(table1, table2, keys: Incomplete | None = None):
    """
    Take a set difference of table rows.

    The row set difference will contain all rows in ``table1`` that are not
    present in ``table2``. If the keys parameter is not defined, all columns in
    ``table1`` will be included in the output table.

    Parameters
    ----------
    table1 : `~astropy.table.Table`
        ``table1`` is on the left side of the set difference.
    table2 : `~astropy.table.Table`
        ``table2`` is on the right side of the set difference.
    keys : str or list of str
        Name(s) of column(s) used to match rows of left and right tables.
        Default is to use all columns in ``table1``.

    Returns
    -------
    diff_table : `~astropy.table.Table`
        New table containing the set difference between tables. If the set
        difference is none, an empty table will be returned.

    Examples
    --------
    To get a set difference between two tables::

      >>> from astropy.table import setdiff, Table
      >>> t1 = Table({'a': [1, 4, 9], 'b': ['c', 'd', 'f']}, names=('a', 'b'))
      >>> t2 = Table({'a': [1, 5, 9], 'b': ['c', 'b', 'f']}, names=('a', 'b'))
      >>> print(t1)
       a   b
      --- ---
        1   c
        4   d
        9   f
      >>> print(t2)
       a   b
      --- ---
        1   c
        5   b
        9   f
      >>> print(setdiff(t1, t2))
       a   b
      --- ---
        4   d

      >>> print(setdiff(t2, t1))
       a   b
      --- ---
        5   b
    """
def vstack(tables, join_type: str = 'outer', metadata_conflicts: str = 'warn'):
    """
    Stack tables vertically (along rows).

    A ``join_type`` of 'exact' means that the tables must all have exactly
    the same column names (though the order can vary).  If ``join_type``
    is 'inner' then the intersection of common columns will be the output.
    A value of 'outer' (default) means the output will have the union of
    all columns, with table values being masked where no common values are
    available.

    Parameters
    ----------
    tables : `~astropy.table.Table` or `~astropy.table.Row` or list thereof
        Table(s) to stack along rows (vertically) with the current table
    join_type : str
        Join type ('inner' | 'exact' | 'outer'), default is 'outer'
    metadata_conflicts : str
        How to proceed with metadata conflicts. This should be one of:
            * ``'silent'``: silently pick the last conflicting meta-data value
            * ``'warn'``: pick the last conflicting meta-data value, but emit a warning (default)
            * ``'error'``: raise an exception.

    Returns
    -------
    stacked_table : `~astropy.table.Table` object
        New table containing the stacked data from the input tables.

    Examples
    --------
    To stack two tables along rows do::

      >>> from astropy.table import vstack, Table
      >>> t1 = Table({'a': [1, 2], 'b': [3, 4]}, names=('a', 'b'))
      >>> t2 = Table({'a': [5, 6], 'b': [7, 8]}, names=('a', 'b'))
      >>> print(t1)
       a   b
      --- ---
        1   3
        2   4
      >>> print(t2)
       a   b
      --- ---
        5   7
        6   8
      >>> print(vstack([t1, t2]))
       a   b
      --- ---
        1   3
        2   4
        5   7
        6   8
    """
def hstack(tables, join_type: str = 'outer', uniq_col_name: str = '{col_name}_{table_name}', table_names: Incomplete | None = None, metadata_conflicts: str = 'warn'):
    """
    Stack tables along columns (horizontally).

    A ``join_type`` of 'exact' means that the tables must all
    have exactly the same number of rows.  If ``join_type`` is 'inner' then
    the intersection of rows will be the output.  A value of 'outer' (default)
    means the output will have the union of all rows, with table values being
    masked where no common values are available.

    Parameters
    ----------
    tables : `~astropy.table.Table` or `~astropy.table.Row` or list thereof
        Tables to stack along columns (horizontally) with the current table
    join_type : str
        Join type ('inner' | 'exact' | 'outer'), default is 'outer'
    uniq_col_name : str or None
        String generate a unique output column name in case of a conflict.
        The default is '{col_name}_{table_name}'.
    table_names : list of str or None
        Two-element list of table names used when generating unique output
        column names.  The default is ['1', '2', ..].
    metadata_conflicts : str
        How to proceed with metadata conflicts. This should be one of:
            * ``'silent'``: silently pick the last conflicting meta-data value
            * ``'warn'``: pick the last conflicting meta-data value,
              but emit a warning (default)
            * ``'error'``: raise an exception.

    Returns
    -------
    stacked_table : `~astropy.table.Table` object
        New table containing the stacked data from the input tables.

    See Also
    --------
    Table.add_columns, Table.replace_column, Table.update

    Examples
    --------
    To stack two tables horizontally (along columns) do::

      >>> from astropy.table import Table, hstack
      >>> t1 = Table({'a': [1, 2], 'b': [3, 4]}, names=('a', 'b'))
      >>> t2 = Table({'c': [5, 6], 'd': [7, 8]}, names=('c', 'd'))
      >>> print(t1)
       a   b
      --- ---
        1   3
        2   4
      >>> print(t2)
       c   d
      --- ---
        5   7
        6   8
      >>> print(hstack([t1, t2]))
       a   b   c   d
      --- --- --- ---
        1   3   5   7
        2   4   6   8
    """
def unique(input_table, keys: Incomplete | None = None, silent: bool = False, keep: str = 'first'):
    """
    Return a new table with unique rows, sorted by ``keys``.

    Parameters
    ----------
    input_table : table-like
    keys : str or list of str
        Name(s) of column(s) used to create unique rows.
        Default is to use all columns.
    keep : {'first', 'last', 'none'}
        Whether to keep the first or last row for each set of
        duplicates. If 'none', all rows that are duplicate are
        removed, leaving only rows that are already unique in
        the input.
        Default is 'first'.
    silent : bool
        If `True`, masked value column(s) are silently removed from
        ``keys``. If `False`, an exception is raised when ``keys``
        contains masked value column(s).
        Default is `False`.

    Returns
    -------
    unique_table : `~astropy.table.Table` object
        New table containing only the unique rows of ``input_table``.

    Examples
    --------
    >>> from astropy.table import unique, Table
    >>> import numpy as np
    >>> table = Table(data=[[1,2,3,2,3,3],
    ... [2,3,4,5,4,6],
    ... [3,4,5,6,7,8]],
    ... names=['col1', 'col2', 'col3'],
    ... dtype=[np.int32, np.int32, np.int32])
    >>> table
    <Table length=6>
     col1  col2  col3
    int32 int32 int32
    ----- ----- -----
        1     2     3
        2     3     4
        3     4     5
        2     5     6
        3     4     7
        3     6     8
    >>> unique(table, keys='col1')
    <Table length=3>
     col1  col2  col3
    int32 int32 int32
    ----- ----- -----
        1     2     3
        2     3     4
        3     4     5
    >>> unique(table, keys=['col1'], keep='last')
    <Table length=3>
     col1  col2  col3
    int32 int32 int32
    ----- ----- -----
        1     2     3
        2     5     6
        3     6     8
    >>> unique(table, keys=['col1', 'col2'])
    <Table length=5>
     col1  col2  col3
    int32 int32 int32
    ----- ----- -----
        1     2     3
        2     3     4
        2     5     6
        3     4     5
        3     6     8
    >>> unique(table, keys=['col1', 'col2'], keep='none')
    <Table length=4>
     col1  col2  col3
    int32 int32 int32
    ----- ----- -----
        1     2     3
        2     3     4
        2     5     6
        3     6     8
    >>> unique(table, keys=['col1'], keep='none')
    <Table length=1>
     col1  col2  col3
    int32 int32 int32
    ----- ----- -----
        1     2     3

    """
