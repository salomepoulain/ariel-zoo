from _typeshed import Incomplete

__all__ = ['Kernel', 'Kernel1D', 'Kernel2D', 'kernel_arithmetics']

class Kernel:
    """
    Convolution kernel base class.

    Parameters
    ----------
    array : ndarray
        Kernel array.
    """
    _separable: bool
    _is_bool: bool
    _model: Incomplete
    __array_ufunc__: Incomplete
    _array: Incomplete
    def __init__(self, array) -> None: ...
    @property
    def truncation(self):
        """
        Absolute deviation of the sum of the kernel array values from
        one.
        """
    @property
    def is_bool(self):
        """
        Indicates if kernel is bool.

        If the kernel is bool the multiplication in the convolution could
        be omitted, to increase the performance.
        """
    @property
    def model(self):
        """
        Kernel response model.
        """
    @property
    def dimension(self):
        """
        Kernel dimension.
        """
    @property
    def center(self):
        """
        Index of the kernel center.
        """
    _kernel_sum: Incomplete
    def normalize(self, mode: str = 'integral') -> None:
        """
        Normalize the filter kernel.

        Parameters
        ----------
        mode : {'integral', 'peak'}
            One of the following modes:
                * 'integral' (default)
                    Kernel is normalized such that its integral = 1.
                * 'peak'
                    Kernel is normalized such that its peak = 1.
        """
    @property
    def shape(self):
        """
        Shape of the kernel array.
        """
    @property
    def separable(self):
        """
        Indicates if the filter kernel is separable.

        A 2D filter is separable, when its filter array can be written as the
        outer product of two 1D arrays.

        If a filter kernel is separable, higher dimension convolutions will be
        performed by applying the 1D filter array consecutively on every dimension.
        This is significantly faster, than using a filter array with the same
        dimension.
        """
    @property
    def array(self):
        """
        Filter kernel array.
        """
    def __add__(self, kernel):
        """
        Add two filter kernels.
        """
    def __sub__(self, kernel):
        """
        Subtract two filter kernels.
        """
    def __mul__(self, value):
        """
        Multiply kernel with number or convolve two kernels.
        """
    def __rmul__(self, value):
        """
        Multiply kernel with number or convolve two kernels.
        """
    def __array__(self, dtype: Incomplete | None = None, copy: Incomplete | None = None):
        """
        Array representation of the kernel.
        """

class Kernel1D(Kernel):
    """
    Base class for 1D filter kernels.

    Parameters
    ----------
    model : `~astropy.modeling.FittableModel`
        Model to be evaluated.
    x_size : int or None, optional
        Size of the kernel array. Default = ⌊8*width+1⌋.
        Only used if ``array`` is None.
    array : ndarray or None, optional
        Kernel array.
    width : number
        Width of the filter kernel.
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by linearly interpolating
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin.
    factor : number, optional
        Factor of oversampling. Default factor = 10.
    """
    def __init__(self, model: Incomplete | None = None, x_size: Incomplete | None = None, array: Incomplete | None = None, **kwargs) -> None: ...

class Kernel2D(Kernel):
    """
    Base class for 2D filter kernels.

    Parameters
    ----------
    model : `~astropy.modeling.FittableModel`
        Model to be evaluated.
    x_size : int, optional
        Size in x direction of the kernel array. Default = ⌊8*width + 1⌋.
        Only used if ``array`` is None.
    y_size : int, optional
        Size in y direction of the kernel array. Default = ⌊8*width + 1⌋.
        Only used if ``array`` is None,
    array : ndarray or None, optional
        Kernel array. Default is None.
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by performing a bilinear interpolation
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin.
    width : number
        Width of the filter kernel.
    factor : number, optional
        Factor of oversampling. Default factor = 10.
    """
    def __init__(self, model: Incomplete | None = None, x_size: Incomplete | None = None, y_size: Incomplete | None = None, array: Incomplete | None = None, **kwargs) -> None: ...

def kernel_arithmetics(kernel, value, operation):
    """
    Add, subtract or multiply two kernels.

    Parameters
    ----------
    kernel : `astropy.convolution.Kernel`
        Kernel instance.
    value : `astropy.convolution.Kernel`, float, or int
        Value to operate with.
    operation : {'add', 'sub', 'mul'}
        One of the following operations:
            * 'add'
                Add two kernels
            * 'sub'
                Subtract two kernels
            * 'mul'
                Multiply kernel with number or convolve two kernels.
    """
