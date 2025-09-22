from _typeshed import Incomplete
from astropy.utils.compat.optional_deps import HAS_PYARROW as HAS_PYARROW
from astropy.utils.exceptions import AstropyUserWarning as AstropyUserWarning
from astropy.utils.misc import NOT_OVERWRITING_MSG as NOT_OVERWRITING_MSG

PARQUET_SIGNATURE: bytes
__all__: Incomplete

def parquet_identify(origin, filepath, fileobj, *args, **kwargs):
    """Checks if input is in the Parquet format.

    Parameters
    ----------
    origin : Any
    filepath : str or None
    fileobj : `~pyarrow.NativeFile` or None
    *args, **kwargs

    Returns
    -------
    is_parquet : bool
        True if 'fileobj' is not None and is a pyarrow file, or if
        'filepath' is a string ending with '.parquet' or '.parq'.
        False otherwise.
    """
def read_table_parquet(input, include_names: Incomplete | None = None, exclude_names: Incomplete | None = None, schema_only: bool = False, filters: Incomplete | None = None):
    """
    Read a Table object from a Parquet file.

    This requires `pyarrow <https://arrow.apache.org/docs/python/>`_
    to be installed.

    The ``filters`` parameter consists of predicates that are expressed
    in disjunctive normal form (DNF), like ``[[('x', '=', 0), ...], ...]``.
    DNF allows arbitrary boolean logical combinations of single column
    predicates. The innermost tuples each describe a single column predicate.
    The list of inner predicates is interpreted as a conjunction (AND),
    forming a more selective and multiple column predicate. Finally, the most
    outer list combines these filters as a disjunction (OR).

    Predicates may also be passed as List[Tuple]. This form is interpreted
    as a single conjunction. To express OR in predicates, one must
    use the (preferred) List[List[Tuple]] notation.

    Each tuple has format: (``key``, ``op``, ``value``) and compares the
    ``key`` with the ``value``.
    The supported ``op`` are:  ``=`` or ``==``, ``!=``, ``<``, ``>``, ``<=``,
    ``>=``, ``in`` and ``not in``. If the ``op`` is ``in`` or ``not in``, the
    ``value`` must be a collection such as a ``list``, a ``set`` or a
    ``tuple``.

    For example:

    .. code-block:: python

        ('x', '=', 0)
        ('y', 'in', ['a', 'b', 'c'])
        ('z', 'not in', {'a','b'})

    Parameters
    ----------
    input : str or path-like or file-like object
        If a string or path-like object, the filename to read the table from.
        If a file-like object, the stream to read data.
    include_names : list [str], optional
        List of names to include in output. If not supplied, then
        include all columns.
    exclude_names : list [str], optional
        List of names to exclude from output (applied after ``include_names``).
        If not supplied then no columns are excluded.
    schema_only : bool, optional
        Only read the schema/metadata with table information.
    filters : list [tuple] or list [list [tuple] ] or None, optional
        Rows which do not match the filter predicate will be removed from
        scanned data.  See `pyarrow.parquet.read_table()` for details.

    Returns
    -------
    table : `~astropy.table.Table`
        Table will have zero rows and only metadata information
        if schema_only is True.
    """
def write_table_parquet(table, output, overwrite: bool = False) -> None:
    """
    Write a Table object to a Parquet file.

    The parquet writer supports tables with regular columns, fixed-size array
    columns, and variable-length array columns (provided all arrays have the
    same type).

    This requires `pyarrow <https://arrow.apache.org/docs/python/>`_
    to be installed.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Data table that is to be written to output.
    output : str or path-like
        The filename to write the table to.
    overwrite : bool, optional
        Whether to overwrite any existing file without warning. Default `False`.

    Notes
    -----
    Tables written with array columns (fixed-size or variable-length) cannot
    be read with pandas.

    Raises
    ------
    ValueError
        If one of the columns has a mixed-type variable-length array, or
        if it is a zero-length table and any of the columns are variable-length
        arrays.
    """
def _get_names(_dict):
    """Recursively find the names in a serialized column dictionary.

    Parameters
    ----------
    _dict : `dict`
        Dictionary from astropy __serialized_columns__

    Returns
    -------
    all_names : `list` [`str`]
        All the column names mentioned in _dict and sub-dicts.
    """
def register_parquet() -> None:
    """
    Register Parquet with Unified I/O.
    """
def get_pyarrow(): ...
def write_parquet_votable(table, output, *, metadata: Incomplete | None = None, overwrite: bool = False, overwrite_metadata: bool = False) -> None:
    """
    Writes a Parquet file with a VOT (XML) metadata table included.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Data table that is to be written to output.
    output : str or path-like
        The filename to write the table to.
    metadata : dict
        Nested dictionary (keys = column names; sub-keys = meta keys) for each
        of the columns containing a dictionary with metadata. Existing metadata
        takes precedent, use ``overwrite_metadata`` to ensure this dictionary is
        being used in all cases.
    overwrite : bool, optional
        If `True`, overwrite the output file if it exists. Raises an
        ``OSError`` if ``False`` and the output file exists. Default is `False`.
    overwrite_metadata : bool, optional
        If `True`, overwrite existing column metadata. Default is `False`.
    """
def read_parquet_votable(filename):
    """
    Reads a Parquet file with a VOT (XML) metadata table included.

    Parameters
    ----------
    filename : str or path-like or file-like object
        If a string or path-like object, the filename to read the table from.
        If a file-like object, the stream to read data.

    Returns
    -------
    table : `~astropy.table.Table`
        A table with included votable metadata, e.g. as column units.
    """
def register_parquet_votable() -> None:
    """
    Register Parquet VOT with Unified I/O.
    """
