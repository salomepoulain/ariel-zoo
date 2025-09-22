from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['data_info_factory', 'dtype_info_name', 'BaseColumnInfo', 'DataInfo', 'MixinInfo', 'ParentDtypeInfo']

def dtype_info_name(dtype):
    """Return a human-oriented string name of the ``dtype`` arg.
    This can be use by astropy methods that present type information about
    a data object.

    The output is mostly equivalent to ``dtype.name`` which takes the form
    <type_name>[B] where <type_name> is like ``int`` or ``bool`` and [B] is an
    optional number of bits which gets included only for numeric types.

    The output is shown below for ``bytes`` and ``str`` types, with <N> being
    the number of characters. This representation corresponds to the Python
    type that matches the dtype::

      Numpy          S<N>      U<N>
      Python      bytes<N>   str<N>

    Parameters
    ----------
    dtype : str, `~numpy.dtype`, type
        Input as an object that can be converted via :class:`numpy.dtype`.

    Returns
    -------
    dtype_info_name : str
        String name of ``dtype``
    """
def data_info_factory(names, funcs):
    """
    Factory to create a function that can be used as an ``option``
    for outputting data object summary information.

    Examples
    --------
    >>> from astropy.utils.data_info import data_info_factory
    >>> from astropy.table import Column
    >>> c = Column([4., 3., 2., 1.])
    >>> mystats = data_info_factory(names=['min', 'median', 'max'],
    ...                             funcs=[np.min, np.median, np.max])
    >>> c.info(option=mystats)
    min = 1
    median = 2.5
    max = 4
    n_bad = 0
    length = 4

    Parameters
    ----------
    names : list
        List of information attribute names
    funcs : list
        List of functions that compute the corresponding information attribute

    Returns
    -------
    func : function
        Function that can be used as a data info option
    """

class InfoAttribute:
    attr: Incomplete
    default: Incomplete
    def __init__(self, attr, default: Incomplete | None = None) -> None: ...
    def __get__(self, instance, owner_cls): ...
    def __set__(self, instance, value) -> None: ...

class ParentAttribute:
    attr: Incomplete
    def __init__(self, attr) -> None: ...
    def __get__(self, instance, owner_cls): ...
    def __set__(self, instance, value) -> None: ...

class DataInfoMeta(type):
    def __new__(mcls, name, bases, dct): ...
    def __init__(cls, name, bases, dct) -> None: ...

class DataInfo(metaclass=DataInfoMeta):
    """
    Descriptor that data classes use to add an ``info`` attribute for storing
    data attributes in a uniform and portable way.  Note that it *must* be
    called ``info`` so that the DataInfo() object can be stored in the
    ``instance`` using the ``info`` key.  Because owner_cls.x is a descriptor,
    Python doesn't use __dict__['x'] normally, and the descriptor can safely
    store stuff there.  Thanks to
    https://nbviewer.jupyter.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb
    for this trick that works for non-hashable classes.

    Parameters
    ----------
    bound : bool
        If True this is a descriptor attribute in a class definition, else it
        is a DataInfo() object that is bound to a data object instance. Default is False.
    """
    _stats: Incomplete
    attrs_from_parent: Incomplete
    attr_names: Incomplete
    _attr_defaults: Incomplete
    _attrs_no_copy: Incomplete
    _info_summary_attrs: Incomplete
    __slots__: Incomplete
    _represent_as_dict_attrs: Incomplete
    _construct_from_dict_args: Incomplete
    _represent_as_dict_primary_data: Incomplete
    _attrs: Incomplete
    def __init__(self, bound: bool = False) -> None: ...
    @property
    def _parent(self): ...
    _parent_cls: Incomplete
    def __get__(self, instance, owner_cls): ...
    def __set__(self, instance, value) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def _represent_as_dict(self, attrs: Incomplete | None = None):
        """Get the values for the parent ``attrs`` and return as a dict. This
        ignores any attributes that are None.  In the context of serializing
        the supported core astropy classes this conversion will succeed and
        results in more succinct and less python-specific YAML.
        By default, uses '_represent_as_dict_attrs'.
        """
    def _construct_from_dict(self, map): ...
    info_summary_attributes: Incomplete
    info_summary_stats: Incomplete
    def __call__(self, option: str = 'attributes', out: str = ''):
        """
        Write summary information about data object to the ``out`` filehandle.
        By default this prints to standard output via sys.stdout.

        The ``option`` argument specifies what type of information
        to include.  This can be a string, a function, or a list of
        strings or functions.  Built-in options are:

        - ``attributes``: data object attributes like ``dtype`` and ``format``
        - ``stats``: basic statistics: min, mean, and max

        If a function is specified then that function will be called with the
        data object as its single argument.  The function must return an
        OrderedDict containing the information attributes.

        If a list is provided then the information attributes will be
        appended for each of the options, in order.

        Examples
        --------
        >>> from astropy.table import Column
        >>> c = Column([1, 2], unit='m', dtype='int32')
        >>> c.info()
        dtype = int32
        unit = m
        class = Column
        n_bad = 0
        length = 2

        >>> c.info(['attributes', 'stats'])
        dtype = int32
        unit = m
        class = Column
        mean = 1.5
        std = 0.5
        min = 1
        max = 2
        n_bad = 0
        length = 2

        Parameters
        ----------
        option : str, callable, list of (str or callable)
            Info option, defaults to 'attributes'.
        out : file-like, None
            Output destination, defaults to sys.stdout.  If None then the
            OrderedDict with information attributes is returned

        Returns
        -------
        info : `~collections.OrderedDict` or None
            `~collections.OrderedDict` if out==None else None
        """
    def __repr__(self) -> str: ...

class BaseColumnInfo(DataInfo):
    """Base info class for anything that can be a column in an astropy Table.

    There are at least two classes that inherit from this:

      ColumnInfo: for native astropy Column / MaskedColumn objects
      MixinInfo: for mixin column objects

    Note that this class is defined here so that mixins can use it
    without importing the table package.
    """
    attr_names: Incomplete
    _attrs_no_copy: Incomplete
    _serialize_context: Incomplete
    __slots__: Incomplete
    @property
    def parent_table(self): ...
    @parent_table.setter
    def parent_table(self, parent_table) -> None: ...
    _format_funcs: Incomplete
    def __init__(self, bound: bool = False) -> None: ...
    def __set__(self, instance, value) -> None: ...
    def iter_str_vals(self) -> Generator[Incomplete, Incomplete]:
        """
        This is a mixin-safe version of Column.iter_str_vals.
        """
    @property
    def indices(self): ...
    @indices.setter
    def indices(self, indices) -> None: ...
    def adjust_indices(self, index, value, col_len) -> None:
        """
        Adjust info indices after column modification.

        Parameters
        ----------
        index : slice, int, list, or ndarray
            Element(s) of column to modify. This parameter can
            be a single row number, a list of row numbers, an
            ndarray of row numbers, a boolean ndarray (a mask),
            or a column slice.
        value : int, list, or ndarray
            New value(s) to insert
        col_len : int
            Length of the column
        """
    def slice_indices(self, col_slice, item, col_len):
        """
        Given a sliced object, modify its indices
        to correctly represent the slice.

        Parameters
        ----------
        col_slice : `~astropy.table.Column` or mixin
            Sliced object. If not a column, it must be a valid mixin, see
            https://docs.astropy.org/en/stable/table/mixin_columns.html
        item : slice, list, or ndarray
            Slice used to create col_slice
        col_len : int
            Length of original object
        """
    @staticmethod
    def merge_cols_attributes(cols, metadata_conflicts, name, attrs):
        """
        Utility method to merge and validate the attributes ``attrs`` for the
        input table columns ``cols``.

        Note that ``dtype`` and ``shape`` attributes are handled specially.
        These should not be passed in ``attrs`` but will always be in the
        returned dict of merged attributes.

        Parameters
        ----------
        cols : list
            List of input Table column objects
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name
        attrs : list
            List of attribute names to be merged

        Returns
        -------
        attrs : dict
            Of merged attributes.

        """
    def get_sortable_arrays(self) -> None:
        """
        Return a list of arrays which can be lexically sorted to represent
        the order of the parent column.

        The base method raises NotImplementedError and must be overridden.

        Returns
        -------
        arrays : list of ndarray
        """

class MixinInfo(BaseColumnInfo):
    @property
    def name(self): ...
    @name.setter
    def name(self, name) -> None: ...
    @property
    def groups(self): ...

class ParentDtypeInfo(MixinInfo):
    """Mixin that gets info.dtype from parent."""
    attrs_from_parent: Incomplete
