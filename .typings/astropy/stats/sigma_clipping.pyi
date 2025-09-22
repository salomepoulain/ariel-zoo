import numpy as np
from _typeshed import Incomplete
from astropy.stats.funcs import mad_std
from collections.abc import Callable
from numpy.typing import ArrayLike, NDArray
from typing import Literal

__all__ = ['SigmaClip', 'sigma_clip', 'SigmaClippedStats', 'sigma_clipped_stats']

class SigmaClip:
    """
    Class to perform sigma clipping.

    The data will be iterated over, each time rejecting values that are
    less or more than a specified number of standard deviations from a
    center value.

    Clipped (rejected) pixels are those where::

        data < center - (sigma_lower * std)
        data > center + (sigma_upper * std)

    where::

        center = cenfunc(data [, axis=])
        std = stdfunc(data [, axis=])

    Invalid data values (i.e., NaN or inf) are automatically clipped.

    For a functional interface to sigma clipping, see
    :func:`sigma_clip`.

    .. note::
        `scipy.stats.sigmaclip` provides a subset of the functionality
        in this class. Also, its input data cannot be a masked array
        and it does not handle data that contains invalid values (i.e.,
        NaN or inf). Also note that it uses the mean as the centering
        function. The equivalent settings to `scipy.stats.sigmaclip`
        are::

            sigclip = SigmaClip(sigma=4., cenfunc='mean', maxiters=None)
            sigclip(data, axis=None, masked=False, return_bounds=True)

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower
        and upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

    sigma_lower : float or None, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    sigma_upper : float or None, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    maxiters : int or None, optional
        The maximum number of sigma-clipping iterations to perform or
        `None` to clip until convergence is achieved (i.e., iterate
        until the last iteration clips nothing). If convergence is
        achieved prior to ``maxiters`` iterations, the clipping
        iterations will stop. The default is 5.

    cenfunc : {'median', 'mean'} or callable, optional
        The statistic or callable function/object used to compute
        the center value for the clipping. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'median'``.

    stdfunc : {'std', 'mad_std'} or callable, optional
        The statistic or callable function/object used to compute the
        standard deviation about the center value. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'std'``.

    grow : float or `False`, optional
        Radius within which to mask the neighbouring pixels of those
        that fall outwith the clipping limits (only applied along
        ``axis``, if specified). As an example, for a 2D image a value
        of 1 will mask the nearest pixels in a cross pattern around each
        deviant pixel, while 1.5 will also reject the nearest diagonal
        neighbours and so on.

    See Also
    --------
    sigma_clip, sigma_clipped_stats, SigmaClippedStats

    Notes
    -----
    The best performance will typically be obtained by setting
    ``cenfunc`` and ``stdfunc`` to one of the built-in functions
    specified as a string. If one of the options is set to a string
    while the other has a custom callable, you may in some cases see
    better performance if you have the `bottleneck`_ package installed.
    To preserve accuracy, bottleneck is only used for float64 computations.

    .. _bottleneck:  https://github.com/pydata/bottleneck

    Examples
    --------
    This example uses a data array of random variates from a Gaussian
    distribution. We clip all points that are more than 2 sample
    standard deviations from the median. The result is a masked array,
    where the mask is `True` for clipped data::

        >>> from astropy.stats import SigmaClip
        >>> from numpy.random import randn
        >>> randvar = randn(10000)
        >>> sigclip = SigmaClip(sigma=2, maxiters=5)
        >>> filtered_data = sigclip(randvar)

    This example clips all points that are more than 3 sigma relative
    to the sample *mean*, clips until convergence, returns an unmasked
    `~numpy.ndarray`, and modifies the data in-place::

        >>> from astropy.stats import SigmaClip
        >>> from numpy.random import randn
        >>> from numpy import mean
        >>> randvar = randn(10000)
        >>> sigclip = SigmaClip(sigma=3, maxiters=None, cenfunc='mean')
        >>> filtered_data = sigclip(randvar, masked=False, copy=False)

    This example sigma clips along one axis::

        >>> from astropy.stats import SigmaClip
        >>> from numpy.random import normal
        >>> from numpy import arange, diag, ones
        >>> data = arange(5) + normal(0., 0.05, (5, 5)) + diag(ones(5))
        >>> sigclip = SigmaClip(sigma=2.3)
        >>> filtered_data = sigclip(data, axis=0)

    Note that along the other axis, no points would be clipped, as the
    standard deviation is higher.
    """
    sigma: Incomplete
    sigma_lower: Incomplete
    sigma_upper: Incomplete
    maxiters: Incomplete
    cenfunc: Incomplete
    stdfunc: Incomplete
    _cenfunc_parsed: Incomplete
    _stdfunc_parsed: Incomplete
    _min_value: Incomplete
    _max_value: Incomplete
    _niterations: int
    grow: Incomplete
    _binary_dilation: Incomplete
    def __init__(self, sigma: float = 3.0, sigma_lower: float | None = None, sigma_upper: float | None = None, maxiters: int | None = 5, cenfunc: Literal['median', 'mean'] | Callable = 'median', stdfunc: Literal['std', 'mad_std'] | Callable = 'std', grow: float | Literal[False] | None = False) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def _parse_cenfunc(cenfunc: Literal['median', 'mean'] | Callable | None) -> Callable | None: ...
    @staticmethod
    def _parse_stdfunc(stdfunc: Literal['std', 'mad_std'] | Callable | None) -> Callable | None: ...
    def _compute_bounds(self, data: ArrayLike, axis: int | tuple[int, ...] | None = None) -> None: ...
    def _sigmaclip_fast(self, data: ArrayLike, axis: int | tuple[int, ...] | None = None, masked: bool | None = True, return_bounds: bool | None = False, copy: bool | None = True) -> NDArray | np.ma.MaskedArray | tuple[NDArray | np.ma.MaskedArray, float, float] | tuple[NDArray | np.ma.MaskedArray, NDArray, NDArray]:
        """
        Fast C implementation for simple use cases.
        """
    def _sigmaclip_noaxis(self, data: ArrayLike, masked: bool | None = True, return_bounds: bool | None = False, copy: bool | None = True) -> NDArray | np.ma.MaskedArray | tuple[NDArray | np.ma.MaskedArray, float, float]:
        """
        Sigma clip when ``axis`` is None and ``grow`` is not >0.

        In this simple case, we remove clipped elements from the
        flattened array during each iteration.
        """
    def _sigmaclip_withaxis(self, data: ArrayLike, axis: int | tuple[int, ...] | None = None, masked: bool | None = True, return_bounds: bool | None = False, copy: bool | None = True) -> NDArray | np.ma.MaskedArray | tuple[NDArray | np.ma.MaskedArray, float, float] | tuple[NDArray | np.ma.MaskedArray, NDArray, NDArray]:
        """
        Sigma clip the data when ``axis`` or ``grow`` is specified.

        In this case, we replace clipped values with NaNs as placeholder
        values.
        """
    def __call__(self, data: ArrayLike, axis: int | tuple[int, ...] | None = None, masked: bool | None = True, return_bounds: bool | None = False, copy: bool | None = True) -> NDArray | np.ma.MaskedArray | tuple[NDArray | np.ma.MaskedArray, float, float] | tuple[NDArray | np.ma.MaskedArray, NDArray, NDArray]:
        """
        Perform sigma clipping on the provided data.

        Parameters
        ----------
        data : array-like or `~numpy.ma.MaskedArray`
            The data to be sigma clipped.

        axis : None or int or tuple of int, optional
            The axis or axes along which to sigma clip the data. If
            `None`, then the flattened data will be used. ``axis`` is
            passed to the ``cenfunc`` and ``stdfunc``. The default is
            `None`.

        masked : bool, optional
            If `True`, then a `~numpy.ma.MaskedArray` is returned, where
            the mask is `True` for clipped values. If `False`, then a
            `~numpy.ndarray` is returned. The default is `True`.

        return_bounds : bool, optional
            If `True`, then the minimum and maximum clipping bounds are
            also returned.

        copy : bool, optional
            If `True`, then the ``data`` array will be copied. If
            `False` and ``masked=True``, then the returned masked array
            data will contain the same array as the input ``data`` (if
            ``data`` is a `~numpy.ndarray` or `~numpy.ma.MaskedArray`).
            If `False` and ``masked=False``, the input data is modified
            in-place. The default is `True`.

        Returns
        -------
        result : array-like
            If ``masked=True``, then a `~numpy.ma.MaskedArray` is
            returned, where the mask is `True` for clipped values and
            where the input mask was `True`.

            If ``masked=False``, then a `~numpy.ndarray` is returned.

            If ``return_bounds=True``, then in addition to the masked
            array or array above, the minimum and maximum clipping
            bounds are returned.

            If ``masked=False`` and ``axis=None``, then the output
            array is a flattened 1D `~numpy.ndarray` where the clipped
            values have been removed. If ``return_bounds=True`` then the
            returned minimum and maximum thresholds are scalars.

            If ``masked=False`` and ``axis`` is specified, then the
            output `~numpy.ndarray` will have the same shape as the
            input ``data`` and contain ``np.nan`` where values were
            clipped. In this case, integer-type ``data`` arrays will
            be converted to `~numpy.float32`. If the input ``data``
            was a masked array, then the output `~numpy.ndarray` will
            also contain ``np.nan`` where the input mask was `True`. If
            ``return_bounds=True`` then the returned minimum and maximum
            clipping thresholds will be be `~numpy.ndarray`\\s.
        """

def sigma_clip(data: ArrayLike, sigma: float = 3.0, sigma_lower: float | None = None, sigma_upper: float | None = None, maxiters: int | None = 5, cenfunc: Literal['median', 'mean'] | Callable = 'median', stdfunc: Literal['std', 'mad_std'] | Callable = 'std', axis: int | tuple[int, ...] | None = None, masked: bool | None = True, return_bounds: bool | None = False, copy: bool | None = True, grow: float | Literal[False] | None = False) -> ArrayLike | tuple[ArrayLike, float, float] | tuple[ArrayLike, ...]:
    """
    Perform sigma-clipping on the provided data.

    The data will be iterated over, each time rejecting values that are
    less or more than a specified number of standard deviations from a
    center value.

    Clipped (rejected) pixels are those where::

        data < center - (sigma_lower * std)
        data > center + (sigma_upper * std)

    where::

        center = cenfunc(data [, axis=])
        std = stdfunc(data [, axis=])

    Invalid data values (i.e., NaN or inf) are automatically clipped.

    For an object-oriented interface to sigma clipping, see
    :class:`SigmaClip`.

    .. note::
        `scipy.stats.sigmaclip` provides a subset of the functionality
        in this class. Also, its input data cannot be a masked array
        and it does not handle data that contains invalid values (i.e.,
        NaN or inf). Also note that it uses the mean as the centering
        function. The equivalent settings to `scipy.stats.sigmaclip`
        are::

            sigma_clip(sigma=4., cenfunc='mean', maxiters=None, axis=None,
            ...        masked=False, return_bounds=True)

    Parameters
    ----------
    data : array-like or `~numpy.ma.MaskedArray`
        The data to be sigma clipped.

    sigma : float, optional
        The number of standard deviations to use for both the lower
        and upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

    sigma_lower : float or None, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    sigma_upper : float or None, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    maxiters : int or None, optional
        The maximum number of sigma-clipping iterations to perform or
        `None` to clip until convergence is achieved (i.e., iterate
        until the last iteration clips nothing). If convergence is
        achieved prior to ``maxiters`` iterations, the clipping
        iterations will stop. The default is 5.

    cenfunc : {'median', 'mean'} or callable, optional
        The statistic or callable function/object used to compute
        the center value for the clipping. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'median'``.

    stdfunc : {'std', 'mad_std'} or callable, optional
        The statistic or callable function/object used to compute the
        standard deviation about the center value. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'std'``.

    axis : None or int or tuple of int, optional
        The axis or axes along which to sigma clip the data. If `None`,
        then the flattened data will be used. ``axis`` is passed to the
        ``cenfunc`` and ``stdfunc``. The default is `None`.

    masked : bool, optional
        If `True`, then a `~numpy.ma.MaskedArray` is returned, where
        the mask is `True` for clipped values. If `False`, then a
        `~numpy.ndarray` is returned. The default is `True`.

    return_bounds : bool, optional
        If `True`, then the minimum and maximum clipping bounds are also
        returned.

    copy : bool, optional
        If `True`, then the ``data`` array will be copied. If `False`
        and ``masked=True``, then the returned masked array data will
        contain the same array as the input ``data`` (if ``data`` is a
        `~numpy.ndarray` or `~numpy.ma.MaskedArray`). If `False` and
        ``masked=False``, the input data is modified in-place. The
        default is `True`.

    grow : float or `False`, optional
        Radius within which to mask the neighbouring pixels of those
        that fall outwith the clipping limits (only applied along
        ``axis``, if specified). As an example, for a 2D image a value
        of 1 will mask the nearest pixels in a cross pattern around each
        deviant pixel, while 1.5 will also reject the nearest diagonal
        neighbours and so on.

    Returns
    -------
    result : array-like
        If ``masked=True``, then a `~numpy.ma.MaskedArray` is returned,
        where the mask is `True` for clipped values and where the input
        mask was `True`.

        If ``masked=False``, then a `~numpy.ndarray` is returned.

        If ``return_bounds=True``, then in addition to the masked array
        or array above, the minimum and maximum clipping bounds are
        returned.

        If ``masked=False`` and ``axis=None``, then the output array
        is a flattened 1D `~numpy.ndarray` where the clipped values
        have been removed. If ``return_bounds=True`` then the returned
        minimum and maximum thresholds are scalars.

        If ``masked=False`` and ``axis`` is specified, then the
        output `~numpy.ndarray` will have the same shape as the input
        ``data`` and contain ``np.nan`` where values were clipped. In
        this case, integer-type ``data`` arrays will be converted to
        `~numpy.float32`. If the input ``data`` was a masked array,
        then the output `~numpy.ndarray` will also contain ``np.nan``
        where the input mask was `True`. If ``return_bounds=True`` then
        the returned minimum and maximum clipping thresholds will be
        `~numpy.ndarray`\\s.

    See Also
    --------
    SigmaClip, sigma_clipped_stats, SigmaClippedStats

    Notes
    -----
    The best performance will typically be obtained by setting
    ``cenfunc`` and ``stdfunc`` to one of the built-in functions
    specified as a string. If one of the options is set to a string
    while the other has a custom callable, you may in some cases see
    better performance if you have the `bottleneck`_ package installed.
    To preserve accuracy, bottleneck is only used for float64 computations.

    .. _bottleneck:  https://github.com/pydata/bottleneck

    Examples
    --------
    This example uses a data array of random variates from a Gaussian
    distribution. We clip all points that are more than 2 sample
    standard deviations from the median. The result is a masked array,
    where the mask is `True` for clipped data::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> randvar = randn(10000)
        >>> filtered_data = sigma_clip(randvar, sigma=2, maxiters=5)

    This example clips all points that are more than 3 sigma relative
    to the sample *mean*, clips until convergence, returns an unmasked
    `~numpy.ndarray`, and does not copy the data::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> from numpy import mean
        >>> randvar = randn(10000)
        >>> filtered_data = sigma_clip(randvar, sigma=3, maxiters=None,
        ...                            cenfunc=mean, masked=False, copy=False)

    This example sigma clips along one axis::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import normal
        >>> from numpy import arange, diag, ones
        >>> data = arange(5) + normal(0., 0.05, (5, 5)) + diag(ones(5))
        >>> filtered_data = sigma_clip(data, sigma=2.3, axis=0)

    Note that along the other axis, no points would be clipped, as the
    standard deviation is higher.
    """

class SigmaClippedStats:
    """
    Class to calculate sigma-clipped statistics on the provided data.

    Parameters
    ----------
    data : array-like or `~numpy.ma.MaskedArray`
        Data array or object that can be converted to an array.

    mask : `numpy.ndarray` (bool), optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are excluded when computing the statistics.

    mask_value : float, optional
        A data value (e.g., ``0.0``) that is ignored when computing the
        statistics. ``mask_value`` will be masked in addition to any
        input ``mask``.

    sigma : float, optional
        The number of standard deviations to use for both the lower
        and upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

    sigma_lower : float or None, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    sigma_upper : float or None, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    maxiters : int or None, optional
        The maximum number of sigma-clipping iterations to perform or
        `None` to clip until convergence is achieved (i.e., iterate
        until the last iteration clips nothing). If convergence is
        achieved prior to ``maxiters`` iterations, the clipping
        iterations will stop. The default is 5.

    cenfunc : {'median', 'mean'} or callable, optional
        The statistic or callable function/object used to compute
        the center value for the clipping. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'median'``.

    stdfunc : {'std', 'mad_std'} or callable, optional
        The statistic or callable function/object used to compute the
        standard deviation about the center value. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'std'``.

    axis : None or int or tuple of int, optional
        The axis or axes along which to sigma clip the data. If `None`,
        then the flattened data will be used. ``axis`` is passed to the
        ``cenfunc`` and ``stdfunc``. The default is `None`.

    grow : float or `False`, optional
        Radius within which to mask the neighbouring pixels of those
        that fall outwith the clipping limits (only applied along
        ``axis``, if specified). As an example, for a 2D image a value
        of 1 will mask the nearest pixels in a cross pattern around each
        deviant pixel, while 1.5 will also reject the nearest diagonal
        neighbours and so on.

    Notes
    -----
    The best performance will typically be obtained by setting
    ``cenfunc`` and ``stdfunc`` to one of the built-in functions
    specified as a string. If one of the options is set to a string
    while the other has a custom callable, you may in some cases
    see better performance if you have the `bottleneck`_ package
    installed. To preserve accuracy, bottleneck is only used for float64
    computations.

    .. _bottleneck:  https://github.com/pydata/bottleneck

    See Also
    --------
    sigma_clipped_stats, SigmaClip, sigma_clip
    """
    data: Incomplete
    axis: Incomplete
    def __init__(self, data: ArrayLike, *, mask: NDArray | None = None, mask_value: float | None = None, sigma: float = 3.0, sigma_lower: float | None = None, sigma_upper: float | None = None, maxiters: int = 5, cenfunc: Literal['median', 'mean'] | Callable = 'median', stdfunc: Literal['std', 'mad_std'] | Callable = 'std', axis: int | tuple[int, ...] | None = None, grow: float | Literal[False] | None = False) -> None: ...
    def min(self) -> float | NDArray:
        """
        Calculate the minimum of the data.

        NaN values are ignored.

        Returns
        -------
        min : float or `~numpy.ndarray`
            The minimum of the data.
        """
    def max(self) -> float | NDArray:
        """
        Calculate the maximum of the data.

        NaN values are ignored.

        Returns
        -------
        max : float or `~numpy.ndarray`
            The maximum of the data.
        """
    def sum(self) -> float | NDArray:
        """
        Calculate the sum of the data.

        NaN values are ignored.

        Returns
        -------
        sum : float or `~numpy.ndarray`
            The sum of the data.
        """
    def mean(self) -> float | NDArray:
        """
        Calculate the mean of the data.

        NaN values are ignored.

        Returns
        -------
        mean : float or `~numpy.ndarray`
            The mean of the data.
        """
    def median(self) -> float | NDArray:
        """
        Calculate the median of the data.

        NaN values are ignored.

        Returns
        -------
        median : float or `~numpy.ndarray`
            The median of the data.
        """
    def mode(self, median_factor: float = 3.0, mean_factor: float = 2.0) -> float | NDArray:
        """
        Calculate the mode of the data using a estimator of the form
        ``(median_factor * median) - (mean_factor * mean)``.

        NaN values are ignored.

        Parameters
        ----------
        median_factor : float, optional
            The multiplicative factor for the data median. Defaults to 3.

        mean_factor : float, optional
            The multiplicative factor for the data mean. Defaults to 2.

        Returns
        -------
        mode : float or `~numpy.ndarray`
            The estimated mode of the data.
        """
    def std(self, ddof: int = 0) -> float | NDArray:
        """
        Calculate the standard deviation of the data.

        NaN values are ignored.

        Parameters
        ----------
        ddof : int, optional
            The delta degrees of freedom for the standard deviation
            calculation. The divisor used in the calculation is ``N -
            ddof``, where ``N`` represents the number of elements. For
            a population standard deviation where you have data for the
            entire population, use ``ddof=0``. For a sample standard
            deviation where you have a sample of the population, use
            ``ddof=1``. The default is 0.

        Returns
        -------
        std : float or `~numpy.ndarray`
            The standard deviation of the data.
        """
    def var(self, ddof: int = 0) -> float | NDArray:
        """
        Calculate the variance of the data.

        NaN values are ignored.

        Parameters
        ----------
        ddof : int, optional
            The delta degrees of freedom. The divisor used in the
            calculation is ``N - ddof``, where ``N`` represents the
            number of elements. For a population variance where you have
            data for the entire population, use ``ddof=0``. For a sample
            variance where you have a sample of the population, use
            ``ddof=1``. The default is 0.

        Returns
        -------
        var : float or `~numpy.ndarray`
            The variance of the data.
        """
    def biweight_location(self, c: float = 6.0, M: float | None = None) -> float | NDArray:
        """
        Calculate the biweight location of the data.

        NaN values are ignored.

        Parameters
        ----------
        c : float, optional
            Tuning constant for the biweight estimator. Default value is
            6.0.

        M : float or None, optional
            Initial guess for the biweight location. Default value is
            `None`.

        Returns
        -------
        biweight_location : float or `~numpy.ndarray`
            The biweight location of the data.
        """
    def biweight_scale(self, c: float = 6.0, M: float | None = None) -> float | NDArray:
        """
        Calculate the biweight scale of the data.

        NaN values are ignored.

        Parameters
        ----------
        c : float, optional
            Tuning constant for the biweight estimator. Default value is
            6.0.

        M : float or None, optional
            Initial guess for the biweight location. Default value is
            `None`.

        Returns
        -------
        biweight_scale : float or `~numpy.ndarray`
            The biweight scale of the data.
        """
    def mad_std(self) -> float | NDArray:
        """
        Calculate the median absolute deviation (MAD) based standard
        deviation of the data.

        NaN values are ignored.

        Returns
        -------
        mad_std : float or `~numpy.ndarray`
            The MAD-based standard deviation of the data.
        """

def sigma_clipped_stats(data: ArrayLike, mask: NDArray | None = None, mask_value: float | None = None, sigma: float = 3.0, sigma_lower: float | None = None, sigma_upper: float | None = None, maxiters: int | None = 5, cenfunc: Literal['median', 'mean'] | Callable = 'median', stdfunc: Literal['std', 'mad_std'] | Callable = 'std', std_ddof: int = 0, axis: int | tuple[int, ...] | None = None, grow: float | Literal[False] | None = False) -> tuple[float, float, float]:
    """
    Calculate sigma-clipped statistics on the provided data.

    Parameters
    ----------
    data : array-like or `~numpy.ma.MaskedArray`
        Data array or object that can be converted to an array.

    mask : `numpy.ndarray` (bool), optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are excluded when computing the statistics.

    mask_value : float, optional
        A data value (e.g., ``0.0``) that is ignored when computing the
        statistics. ``mask_value`` will be masked in addition to any
        input ``mask``.

    sigma : float, optional
        The number of standard deviations to use for both the lower
        and upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

    sigma_lower : float or None, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    sigma_upper : float or None, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    maxiters : int or None, optional
        The maximum number of sigma-clipping iterations to perform or
        `None` to clip until convergence is achieved (i.e., iterate
        until the last iteration clips nothing). If convergence is
        achieved prior to ``maxiters`` iterations, the clipping
        iterations will stop. The default is 5.

    cenfunc : {'median', 'mean'} or callable, optional
        The statistic or callable function/object used to compute
        the center value for the clipping. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'median'``.

    stdfunc : {'std', 'mad_std'} or callable, optional
        The statistic or callable function/object used to compute the
        standard deviation about the center value. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'std'``.

    std_ddof : int, optional
        The delta degrees of freedom for the standard deviation
        calculation. The divisor used in the calculation is ``N -
        std_ddof``, where ``N`` represents the number of elements. For a
        population standard deviation where you have data for the entire
        population, use ``std_ddof=0``. For a sample standard deviation
        where you have a sample of the population, use ``std_ddof=1``.
        The default is 0.

    axis : None or int or tuple of int, optional
        The axis or axes along which to sigma clip the data. If `None`,
        then the flattened data will be used. ``axis`` is passed to the
        ``cenfunc`` and ``stdfunc``. The default is `None`.

    grow : float or `False`, optional
        Radius within which to mask the neighbouring pixels of those
        that fall outwith the clipping limits (only applied along
        ``axis``, if specified). As an example, for a 2D image a value
        of 1 will mask the nearest pixels in a cross pattern around each
        deviant pixel, while 1.5 will also reject the nearest diagonal
        neighbours and so on.

    Notes
    -----
    The best performance will typically be obtained by setting
    ``cenfunc`` and ``stdfunc`` to one of the built-in functions
    specified as a string. If one of the options is set to a string
    while the other has a custom callable, you may in some cases see
    better performance if you have the `bottleneck`_ package installed.
    To preserve accuracy, bottleneck is only used for float64 computations.

    .. _bottleneck:  https://github.com/pydata/bottleneck

    Returns
    -------
    mean, median, stddev : float
        The mean, median, and standard deviation of the sigma-clipped
        data.

    See Also
    --------
    SigmaClippedStats, SigmaClip, sigma_clip
    """
