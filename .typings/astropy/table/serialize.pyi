from .column import Column as Column, MaskedColumn as MaskedColumn
from .table import QTable as QTable, Table as Table, has_info_class as has_info_class
from _typeshed import Incomplete
from astropy.units.quantity import QuantityInfo as QuantityInfo
from astropy.utils.data_info import MixinInfo as MixinInfo
from collections import OrderedDict

__construct_mixin_classes: Incomplete

class SerializedColumnInfo(MixinInfo):
    """
    Minimal info to allow SerializedColumn to be recognized as a mixin Column.

    Used to help create a dict of columns in ColumnInfo for structured data.
    """
    def _represent_as_dict(self): ...

class SerializedColumn(dict):
    """Subclass of dict used to serialize  mixin columns.

    It is used in the representation to contain the name and possible
    other info for a mixin column or attribute (either primary data or an
    array-like attribute) that is serialized as a column in the table.

    """
    info: Incomplete
    @property
    def shape(self):
        """Minimal shape implementation to allow use as a mixin column.

        Returns the shape of the first item that has a shape at all,
        or ``()`` if none of the values has a shape attribute.
        """

def _represent_mixin_as_column(col, name, new_cols, mixin_cols, exclude_classes=()):
    '''Carry out processing needed to serialize ``col`` in an output table
    consisting purely of plain ``Column`` or ``MaskedColumn`` columns.  This
    relies on the object determine if any transformation is required and may
    depend on the ``serialize_method`` and ``serialize_context`` context
    variables.  For instance a ``MaskedColumn`` may be stored directly to
    FITS, but can also be serialized as separate data and mask columns.

    This function builds up a list of plain columns in the ``new_cols`` arg (which
    is passed as a persistent list).  This includes both plain columns from the
    original table and plain columns that represent data from serialized columns
    (e.g. ``jd1`` and ``jd2`` arrays from a ``Time`` column).

    For serialized columns the ``mixin_cols`` dict is updated with required
    attributes and information to subsequently reconstruct the table.

    Table mixin columns are always serialized and get represented by one
    or more data columns.  In earlier versions of the code *only* mixin
    columns were serialized, hence the use within this code of "mixin"
    to imply serialization.  Starting with version 3.1, the non-mixin
    ``MaskedColumn`` can also be serialized.
    '''
def represent_mixins_as_columns(tbl, exclude_classes=()):
    """Represent input Table ``tbl`` using only `~astropy.table.Column`
    or  `~astropy.table.MaskedColumn` objects.

    This function represents any mixin columns like `~astropy.time.Time` in
    ``tbl`` to one or more plain ``~astropy.table.Column`` objects and returns
    a new Table.  A single mixin column may be split into multiple column
    components as needed for fully representing the column.  This includes the
    possibility of recursive splitting, as shown in the example below.  The
    new column names are formed as ``<column_name>.<component>``, e.g.
    ``sc.ra`` for a `~astropy.coordinates.SkyCoord` column named ``sc``.

    In addition to splitting columns, this function updates the table ``meta``
    dictionary to include a dict named ``__serialized_columns__`` which provides
    additional information needed to construct the original mixin columns from
    the split columns.

    This function is used by astropy I/O when writing tables to ECSV, FITS,
    HDF5 formats.

    Note that if the table does not include any mixin columns then the original
    table is returned with no update to ``meta``.

    Parameters
    ----------
    tbl : `~astropy.table.Table` or subclass
        Table to represent mixins as Columns
    exclude_classes : tuple of class
        Exclude any mixin columns which are instannces of any classes in the tuple

    Returns
    -------
    tbl : `~astropy.table.Table`
        New Table with updated columns, or else the original input ``tbl``

    Examples
    --------
    >>> from astropy.table import Table, represent_mixins_as_columns
    >>> from astropy.time import Time
    >>> from astropy.coordinates import SkyCoord

    >>> x = [100.0, 200.0]
    >>> obstime = Time([1999.0, 2000.0], format='jyear')
    >>> sc = SkyCoord([1, 2], [3, 4], unit='deg', obstime=obstime)
    >>> tbl = Table([sc, x], names=['sc', 'x'])
    >>> represent_mixins_as_columns(tbl)
    <Table length=2>
     sc.ra   sc.dec sc.obstime.jd1 sc.obstime.jd2    x
      deg     deg
    float64 float64    float64        float64     float64
    ------- ------- -------------- -------------- -------
        1.0     3.0      2451180.0          -0.25   100.0
        2.0     4.0      2451545.0            0.0   200.0

    """
def _construct_mixin_from_obj_attrs_and_info(obj_attrs, info): ...

class _TableLite(OrderedDict):
    """
    Minimal table-like object for _construct_mixin_from_columns.  This allows
    manipulating the object like a Table but without the actual overhead
    for a full Table.

    More pressing, there is an issue with constructing MaskedColumn, where the
    encoded Column components (data, mask) are turned into a MaskedColumn.
    When this happens in a real table then all other columns are immediately
    Masked and a warning is issued. This is not desirable.
    """
    def add_column(self, col, index: int = 0) -> None: ...
    @property
    def colnames(self): ...
    def itercols(self): ...

def _construct_mixin_from_columns(new_name, obj_attrs, out): ...
def _construct_mixins_from_columns(tbl): ...
