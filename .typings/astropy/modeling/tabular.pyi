from .core import Model
from _typeshed import Incomplete

__all__ = ['tabular_model', 'Tabular1D', 'Tabular2D']

class _Tabular(Model):
    '''
    Returns an interpolated lookup table value.

    Parameters
    ----------
    points : tuple of ndarray of float, optional
        The points defining the regular grid in n dimensions.
        ndarray must have shapes (m1, ), ..., (mn, ),
    lookup_table : array-like
        The data on a regular grid in n dimensions.
        Must have shapes (m1, ..., mn, ...)
    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest", and "splinef2d". "splinef2d" is only supported for
        2-dimensional data. Default is "linear".
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then ``fill_value`` is used.
    fill_value : float or `~astropy.units.Quantity`, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d". If Quantity is given, it will be converted to the unit of
        ``lookup_table``, if applicable.

    Returns
    -------
    value : ndarray
        Interpolated values at input coordinates.

    Raises
    ------
    ImportError
        Scipy is not installed.

    Notes
    -----
    Uses `scipy.interpolate.interpn`.

    '''
    linear: bool
    fittable: bool
    standard_broadcasting: bool
    _is_dynamic: bool
    _id: int
    outputs: Incomplete
    points: Incomplete
    lookup_table: Incomplete
    bounds_error: Incomplete
    method: Incomplete
    fill_value: Incomplete
    def __init__(self, points: Incomplete | None = None, lookup_table: Incomplete | None = None, method: str = 'linear', bounds_error: bool = True, fill_value=..., **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def input_units(self): ...
    @property
    def return_units(self): ...
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(points_low, points_high)``.

        Examples
        --------
        >>> from astropy.modeling.models import Tabular1D, Tabular2D
        >>> t1 = Tabular1D(points=[1, 2, 3], lookup_table=[10, 20, 30])
        >>> t1.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=1, upper=3)
            }
            model=Tabular1D(inputs=('x',))
            order='C'
        )
        >>> t2 = Tabular2D(points=[[1, 2, 3], [2, 3, 4]],
        ...                lookup_table=[[10, 20, 30], [20, 30, 40]])
        >>> t2.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=1, upper=3)
                y: Interval(lower=2, upper=4)
            }
            model=Tabular2D(inputs=('x', 'y'))
            order='C'
        )

        """
    def evaluate(self, *inputs):
        """
        Return the interpolated values at the input coordinates.

        Parameters
        ----------
        inputs : list of scalar or list of ndarray
            Input coordinates. The number of inputs must be equal
            to the dimensions of the lookup table.
        """
    @property
    def inverse(self): ...

def tabular_model(dim, name: Incomplete | None = None):
    """
    Make a ``Tabular`` model where ``n_inputs`` is
    based on the dimension of the lookup_table.

    This model has to be further initialized and when evaluated
    returns the interpolated values.

    Parameters
    ----------
    dim : int
        Dimensions of the lookup table.
    name : str
        Name for the class.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import tabular_model
    >>> tab = tabular_model(2, name='Tabular2D')
    >>> print(tab)
    <class 'astropy.modeling.tabular.Tabular2D'>
    Name: Tabular2D
    N_inputs: 2
    N_outputs: 1

    Setting ``fill_value`` to `None` allows extrapolation.

    >>> points = ([1, 2, 3], [1, 2, 3])
    >>> table = np.array([[3., 0., 0.],
    ...                   [0., 2., 0.],
    ...                   [0., 0., 0.]])
    >>> model = tab(points, lookup_table=table, name='my_table',
    ...             bounds_error=False, fill_value=None, method='nearest')
    >>> xinterp = [0, 1, 1.5, 2.72, 3.14]
    >>> model(xinterp, xinterp)  # doctest: +FLOAT_CMP
    array([3., 3., 3., 0., 0.])
    """

Tabular1D: Incomplete
Tabular2D: Incomplete
