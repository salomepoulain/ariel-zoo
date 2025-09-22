from _typeshed import Incomplete

__all__ = ['parallel_fit_dask']

class ParameterContainer:
    """
    This is an array container intended to be passed to dask's ``from_array``.

    The initial parameter values need to be broadcast up to the data shape so
    that map_blocks can then iterate over both the data and parameters. We
    need to control the final chunking so that it matches the data. However,
    rather than use dask to do the broadcasting and rechunking, which creates a
    complex graph and results in high memory usage, this class can be used
    instead to do all the broadcasting on-the-fly with Numpy as needed and keeps
    the dask graph simple.
    """
    _values: Incomplete
    shape: Incomplete
    ndim: Incomplete
    dtype: Incomplete
    _iterating_shape: Incomplete
    _iterating_axes: Incomplete
    def __init__(self, values, iterating_shape, iterating_axes, data_shape) -> None: ...
    def __getitem__(self, item): ...

def parallel_fit_dask(*, model, fitter, data, data_unit: Incomplete | None = None, weights: Incomplete | None = None, mask: Incomplete | None = None, fitting_axes: Incomplete | None = None, world: Incomplete | None = None, chunk_n_max: Incomplete | None = None, diagnostics: Incomplete | None = None, diagnostics_path: Incomplete | None = None, diagnostics_callable: Incomplete | None = None, scheduler: Incomplete | None = None, fitter_kwargs: Incomplete | None = None, preserve_native_chunks: bool = False, equivalencies: Incomplete | None = None):
    """
    Fit a model in parallel to an N-dimensional dataset.

    Axes in the N-dimensional dataset are considered to be either 'fitting
    axes' or 'iterating axes'. As a specific example, if fitting a
    spectral cube with two celestial and one spectral axis, then if fitting a
    1D model to each spectrum in the cube, the spectral axis would be a fitting
    axis and the celestial axes would be iterating axes.

    Parameters
    ----------
    model : :class:`astropy.modeling.Model`
        The model to fit, specifying the initial parameter values. The shape
        of the parameters should be broadcastable to the shape of the iterating
        axes.
    fitter : :class:`astropy.modeling.fitting.Fitter`
        The fitter to use in the fitting process.
    data : `numpy.ndarray` or `dask.array.core.Array`
        The N-dimensional data to fit.
    data_units : `astropy.units.Unit`
        Units for the data array, for when the data array is not a ``Quantity``
        instance.
    weights : `numpy.ndarray`, `dask.array.core.Array` or `astropy.nddata.NDUncertainty`
        The weights to use in the fitting. See the documentation for specific
        fitters for more information about the meaning of weights.
        If passed as a `.NDUncertainty` object it will be converted to a
        `.StdDevUncertainty` and then passed to the fitter as 1 over that.
    mask : `numpy.ndarray`
        A boolean mask to be applied to the data.
    fitting_axes : int or tuple
        The axes to keep for the fitting (other axes will be sliced/iterated over)
    world : `None` or tuple or APE-14-WCS
        This can be specified either as a tuple of world coordinates for each
        fitting axis, or as WCS for the whole cube. If specified as a tuple,
        the values in the tuple can be either 1D arrays, or can be given as
        N-dimensional arrays with shape broadcastable to the data shape. If
        specified as a WCS, the WCS can have any dimensionality so long as it
        matches the data. If not specified, the fitting is carried out in pixel
        coordinates.
    chunk_n_max : int
        Maximum number of fits to include in a chunk. If this is made too
        large, then the workload will not be split properly over processes, and
        if it is too small it may be inefficient. If not specified, this will
        default to 500.
    diagnostics : { None | 'error' | 'error+warn' | 'all' }, optional
        Whether to output diagnostic information for fits. This can be either
        `None` (nothing), ``'error'`` (output information for fits that raised
        exceptions), or ``'all'`` (output information for all fits).
    diagnostics_path : str, optional
        If ``diagnostics`` is not `None`, this should be the path to a folder in
        which a folder will be made for each fit that is output.
    diagnostics_callable : callable
        By default, any warnings or errors are output to ``diagnostics_path``.
        However, you can also specify a callable that can e.g. make a plot or
        write out information in a custom format. The callable should take the
        following arguments: the path to the subfolder of ``diagnostics_path``
        for the specific index being fit, a list of the coordinates passed to
        the fitter, the data array, the weights array (or `None` if no weights
        are being used), the model that was fit (or `None` if the fit errored),
        and a dictionary of other keyword arguments passed to the fitter.
    scheduler : str, optional
        If not specified, a local multi-processing scheduler will be
        used. If ``'default'``, whatever is the current default scheduler will be
        used. You can also set this to anything that would be passed to
        ``array.compute(scheduler=...)``
    fitter_kwargs : None or dict
        Keyword arguments to pass to the fitting when it is called.
    preserve_native_chunks : bool, optional
        If `True`, the native data chunks will be used, although an error will
        be raised if this chunk size does not include the whole fitting axes.
    """
