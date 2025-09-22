from _typeshed import Incomplete

__all__ = ['read_table_hdf5', 'write_table_hdf5']

def read_table_hdf5(input, path: Incomplete | None = None, character_as_bytes: bool = True):
    """
    Read a Table object from an HDF5 file.

    This requires `h5py <http://www.h5py.org/>`_ to be installed. If more than one
    table is present in the HDF5 file or group, the first table is read in and
    a warning is displayed.

    Parameters
    ----------
    input : str or :class:`h5py.File` or :class:`h5py.Group` or
        :class:`h5py.Dataset` If a string, the filename to read the table from.
        If an h5py object, either the file or the group object to read the
        table from.
    path : str
        The path from which to read the table inside the HDF5 file.
        This should be relative to the input file or group.
    character_as_bytes : bool
        If `True` then Table columns are left as bytes.
        If `False` then Table columns are converted to unicode.
    """
def write_table_hdf5(table, output, path: Incomplete | None = None, compression: bool = False, append: bool = False, overwrite: bool = False, serialize_meta: bool = False, **create_dataset_kwargs):
    """
    Write a Table object to an HDF5 file.

    This requires `h5py <http://www.h5py.org/>`_ to be installed.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Data table that is to be written to file.
    output : str or os.PathLike[str] or :class:`h5py.File` or :class:`h5py.Group`
        If a string, the filename to write the table to. If an h5py object,
        either the file or the group object to write the table to.
    path : str
        The path to which to write the table inside the HDF5 file.
        This should be relative to the input file or group.
        If not specified, defaults to ``__astropy_table__``.
    compression : bool or str or int
        Whether to compress the table inside the HDF5 file. If set to `True`,
        ``'gzip'`` compression is used. If a string is specified, it should be
        one of ``'gzip'``, ``'szip'``, or ``'lzf'``. If an integer is
        specified (in the range 0-9), ``'gzip'`` compression is used, and the
        integer denotes the compression level.
    append : bool
        Whether to append the table to an existing HDF5 file.
    overwrite : bool
        Whether to overwrite any existing file without warning.
        If ``append=True`` and ``overwrite=True`` then only the dataset will be
        replaced; the file/group will not be overwritten.
    serialize_meta : bool
        Whether to serialize rich table meta-data when writing the HDF5 file, in
        particular such data required to write and read back mixin columns like
        ``Time``, ``SkyCoord``, or ``Quantity`` to the file.
    **create_dataset_kwargs
        Additional keyword arguments are passed to
        ``h5py.File.create_dataset()`` or ``h5py.Group.create_dataset()``.
    """
