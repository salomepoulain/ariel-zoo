from . import from_table as from_table, parse as parse
from .tree import TableElement as TableElement, VOTableFile as VOTableFile
from _typeshed import Incomplete
from astropy.table import Table as Table
from astropy.table.column import BaseColumn as BaseColumn
from astropy.units import Quantity as Quantity
from astropy.utils.misc import NOT_OVERWRITING_MSG as NOT_OVERWRITING_MSG

def is_votable(origin, filepath, fileobj, *args, **kwargs):
    """
    Reads the header of a file to determine if it is a VOTable file.

    Parameters
    ----------
    origin : str or readable file-like
        Path or file object containing a VOTABLE_ xml file.

    Returns
    -------
    is_votable : bool
        Returns `True` if the given file is a VOTable file.
    """
def read_table_votable(input, table_id: Incomplete | None = None, use_names_over_ids: bool = False, verify: Incomplete | None = None, **kwargs):
    """
    Read a Table object from an VO table file.

    Parameters
    ----------
    input : str or `~astropy.io.votable.tree.VOTableFile` or `~astropy.io.votable.tree.TableElement`
        If a string, the filename to read the table from. If a
        :class:`~astropy.io.votable.tree.VOTableFile` or
        :class:`~astropy.io.votable.tree.TableElement` object, the object to extract
        the table from.

    table_id : str or int, optional
        The table to read in.  If a `str`, it is an ID corresponding
        to the ID of the table in the file (not all VOTable files
        assign IDs to their tables).  If an `int`, it is the index of
        the table in the file, starting at 0.

    use_names_over_ids : bool, optional
        When `True` use the ``name`` attributes of columns as the names
        of columns in the `~astropy.table.Table` instance.  Since names
        are not guaranteed to be unique, this may cause some columns
        to be renamed by appending numbers to the end.  Otherwise
        (default), use the ID attributes as the column names.

    verify : {'ignore', 'warn', 'exception'}, optional
        When ``'exception'``, raise an error when the file violates the spec,
        otherwise either issue a warning (``'warn'``) or silently continue
        (``'ignore'``). Warnings may be controlled using the standard Python
        mechanisms.  See the `warnings` module in the Python standard library
        for more information. When not provided, uses the configuration setting
        ``astropy.io.votable.verify``, which defaults to ``'ignore'``.

    **kwargs
        Additional keyword arguments are passed on to `astropy.io.votable.parse`.
    """
def write_table_votable(input, output, table_id: Incomplete | None = None, overwrite: bool = False, tabledata_format: Incomplete | None = None) -> None:
    """
    Write a Table object to an VO table file.

    Parameters
    ----------
    input : Table
        The table to write out.

    output : str
        The filename to write the table to.

    table_id : str, optional
        The table ID to use. If this is not specified, the 'ID' keyword in the
        ``meta`` object of the table will be used.

    overwrite : bool, optional
        Whether to overwrite any existing file without warning.

    tabledata_format : str, optional
        The format of table data to write.  Must be one of ``tabledata``
        (text representation), ``binary`` or ``binary2``.  Default is
        ``tabledata``.  See :ref:`astropy:votable-serialization`.
    """
def write_table_votable_parquet(input, output, column_metadata, *, overwrite: bool = False) -> None:
    '''
    This function allows writing a VOTable (XML) with PARQUET
    serialization. This functionality is currently not
    supported by Astropy (with the reason that this method
    requires writing multiple files: a VOTable/XML and
    PARQUET table). This function presents a wrapper, which
    allows to do this. The concept is simple and probably
    can be improved substantially. We first save the PARQUET
    table using Astropy functionality. Then, we create a
    VOTable with binary serialization. The latter is modified
    later to include an external reference to the create
    PARQUET table file.

    Parameters
    ----------
    input : `~astropy.table.Table`
        The table to write out.

    output : str
        The filename to write the table to.

    column_metadata : dict
        Contains the metadata for the columns such as "unit" or
        "ucd" or "utype".
        (Example: {"id": {"unit": "", "ucd": "meta.id", "utype": "none"},
                   "mass": {"unit": "solMass", "ucd": "phys.mass", "utype": "none"}})
    overwrite : bool, optional
        Whether to overwrite any existing file without warning.

    Returns
    -------
    This function creates a VOTable serialized in Parquet.
    Two files are written:
    1. The VOTable (XML file) including the column metadata and a
        ``STREAM`` tag that embeds the PARQUET table.
    2. The PARQUET table itself.

    Both files are stored at the same location. The name of the
    VOTable is ``output``, and the name of the embedded PARQUET
    file is f"{output}.parquet".
    '''
