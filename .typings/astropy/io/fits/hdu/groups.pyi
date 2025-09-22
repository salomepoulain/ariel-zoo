from .base import DELAYED as DELAYED, DTYPE2BITPIX as DTYPE2BITPIX
from .image import PrimaryHDU as PrimaryHDU
from .table import _TableLikeHDU as _TableLikeHDU
from _typeshed import Incomplete
from astropy.io.fits.column import ColDefs as ColDefs, Column as Column, FITS2NUMPY as FITS2NUMPY
from astropy.io.fits.fitsrec import FITS_rec as FITS_rec, FITS_record as FITS_record
from astropy.io.fits.util import _is_int as _is_int, _is_pseudo_integer as _is_pseudo_integer, _pseudo_zero as _pseudo_zero
from astropy.utils import lazyproperty as lazyproperty

class Group(FITS_record):
    """
    One group of the random group data.
    """
    def __init__(self, input, row: int = 0, start: Incomplete | None = None, end: Incomplete | None = None, step: Incomplete | None = None, base: Incomplete | None = None) -> None: ...
    @property
    def parnames(self): ...
    @property
    def data(self): ...
    def _unique(self): ...
    def par(self, parname):
        """
        Get the group parameter value.
        """
    def setpar(self, parname, value) -> None:
        """
        Set the group parameter value.
        """

class GroupData(FITS_rec):
    """
    Random groups data object.

    Allows structured access to FITS Group data in a manner analogous
    to tables.
    """
    _record_type = Group
    _data_field: Incomplete
    _coldefs: Incomplete
    parnames: Incomplete
    def __new__(cls, input: Incomplete | None = None, bitpix: Incomplete | None = None, pardata: Incomplete | None = None, parnames=[], bscale: Incomplete | None = None, bzero: Incomplete | None = None, parbscales: Incomplete | None = None, parbzeros: Incomplete | None = None):
        """
        Parameters
        ----------
        input : array or FITS_rec instance
            input data, either the group data itself (a
            `numpy.ndarray`) or a record array (`FITS_rec`) which will
            contain both group parameter info and the data.  The rest
            of the arguments are used only for the first case.

        bitpix : int
            data type as expressed in FITS ``BITPIX`` value (8, 16, 32,
            64, -32, or -64)

        pardata : sequence of array
            parameter data, as a list of (numeric) arrays.

        parnames : sequence of str
            list of parameter names.

        bscale : int
            ``BSCALE`` of the data

        bzero : int
            ``BZERO`` of the data

        parbscales : sequence of int
            list of bscales for the parameters

        parbzeros : sequence of int
            list of bzeros for the parameters
        """
    def __array_finalize__(self, obj) -> None: ...
    def __getitem__(self, key): ...
    @property
    def data(self):
        """
        The raw group data represented as a multi-dimensional `numpy.ndarray`
        array.
        """
    def _unique(self): ...
    def par(self, parname):
        """
        Get the group parameter values.
        """

class GroupsHDU(PrimaryHDU, _TableLikeHDU):
    """
    FITS Random Groups HDU class.

    See the :ref:`astropy:random-groups` section in the Astropy documentation
    for more details on working with this type of HDU.
    """
    _bitpix2tform: Incomplete
    _data_type = GroupData
    _data_field: str
    _axes: Incomplete
    def __init__(self, data: Incomplete | None = None, header: Incomplete | None = None) -> None: ...
    @classmethod
    def match_header(cls, header): ...
    def data(self):
        """
        The data of a random group FITS file will be like a binary table's
        data.
        """
    def parnames(self):
        """The names of the group parameters as described by the header."""
    def columns(self): ...
    @property
    def _nrows(self): ...
    def _theap(self): ...
    @property
    def is_image(self): ...
    @property
    def size(self):
        """
        Returns the size (in bytes) of the HDU's data part.
        """
    def update_header(self) -> None: ...
    def _writedata_internal(self, fileobj):
        """
        Basically copy/pasted from `_ImageBaseHDU._writedata_internal()`, but
        we have to get the data's byte order a different way...

        TODO: Might be nice to store some indication of the data's byte order
        as an attribute or function so that we don't have to do this.
        """
    def _verify(self, option: str = 'warn'): ...
    def _calculate_datasum(self):
        """
        Calculate the value for the ``DATASUM`` card in the HDU.
        """
    def _summary(self): ...

def _par_indices(names):
    """
    Given a list of objects, returns a mapping of objects in that list to the
    index or indices at which that object was found in the list.
    """
def _unique_parnames(names):
    """
    Given a list of parnames, including possible duplicates, returns a new list
    of parnames with duplicates prepended by one or more underscores to make
    them unique.  This is also case insensitive.
    """
