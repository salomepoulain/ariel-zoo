import abc
from .transform import BaseTransform
from _typeshed import Incomplete

__all__ = ['BaseInterval', 'ManualInterval', 'MinMaxInterval', 'AsymmetricPercentileInterval', 'PercentileInterval', 'ZScaleInterval']

class BaseInterval(BaseTransform, metaclass=abc.ABCMeta):
    """
    Base class for the interval classes, which, when called with an
    array of values, return an interval computed following different
    algorithms.
    """
    @abc.abstractmethod
    def get_limits(self, values):
        """
        Return the minimum and maximum value in the interval based on
        the values provided.

        Parameters
        ----------
        values : ndarray
            The image values.

        Returns
        -------
        vmin, vmax : float
            The mininium and maximum image value in the interval.
        """
    def __call__(self, values, clip: bool = True, out: Incomplete | None = None):
        """
        Transform values using this interval.

        Parameters
        ----------
        values : array-like
            The input values.
        clip : bool, optional
            If `True` (default), values outside the [0:1] range are
            clipped to the [0:1] range.
        out : ndarray, optional
            If specified, the output values will be placed in this array
            (typically used for in-place calculations).

        Returns
        -------
        result : ndarray
            The transformed values.
        """

class ManualInterval(BaseInterval):
    """
    Interval based on user-specified values.

    Parameters
    ----------
    vmin : float, optional
        The minimum value in the scaling.  Defaults to the image
        minimum (ignoring NaNs)
    vmax : float, optional
        The maximum value in the scaling.  Defaults to the image
        maximum (ignoring NaNs)
    """
    vmin: Incomplete
    vmax: Incomplete
    def __init__(self, vmin: Incomplete | None = None, vmax: Incomplete | None = None) -> None: ...
    def get_limits(self, values): ...

class MinMaxInterval(BaseInterval):
    """
    Interval based on the minimum and maximum values in the data.
    """
    def get_limits(self, values): ...

class AsymmetricPercentileInterval(BaseInterval):
    """
    Interval based on a keeping a specified fraction of pixels (can be
    asymmetric).

    Parameters
    ----------
    lower_percentile : float or None
        The lower percentile below which to ignore pixels. If None, then
        defaults to 0.
    upper_percentile : float or None
        The upper percentile above which to ignore pixels. If None, then
        defaults to 100.
    n_samples : int, optional
        Maximum number of values to use. If this is specified, and there
        are more values in the dataset as this, then values are randomly
        sampled from the array (with replacement).
    """
    lower_percentile: Incomplete
    upper_percentile: Incomplete
    n_samples: Incomplete
    def __init__(self, lower_percentile: Incomplete | None = None, upper_percentile: Incomplete | None = None, n_samples: Incomplete | None = None) -> None: ...
    def get_limits(self, values): ...

class PercentileInterval(AsymmetricPercentileInterval):
    """
    Interval based on a keeping a specified fraction of pixels.

    Parameters
    ----------
    percentile : float
        The fraction of pixels to keep. The same fraction of pixels is
        eliminated from both ends.
    n_samples : int, optional
        Maximum number of values to use. If this is specified, and there
        are more values in the dataset as this, then values are randomly
        sampled from the array (with replacement).
    """
    def __init__(self, percentile, n_samples: Incomplete | None = None) -> None: ...

class ZScaleInterval(BaseInterval):
    """
    Interval based on IRAF's zscale.

    Original implementation:
    https://github.com/spacetelescope/stsci.numdisplay/blob/master/lib/stsci/numdisplay/zscale.py

    Licensed under a 3-clause BSD style license (see AURA_LICENSE.rst).

    Parameters
    ----------
    n_samples : int, optional
        The number of points in the array to sample for determining
        scaling factors.  Defaults to 1000.

        .. versionchanged:: 7.0
            ``nsamples`` parameter is removed.

    contrast : float, optional
        The scaling factor (between 0 and 1) for determining the minimum
        and maximum value.  Larger values decrease the difference
        between the minimum and maximum values used for display.
        Defaults to 0.25.
    max_reject : float, optional
        If more than ``max_reject * npixels`` pixels are rejected, then
        the returned values are the minimum and maximum of the data.
        Defaults to 0.5.
    min_npixels : int, optional
        If there are less than ``min_npixels`` pixels remaining after
        the pixel rejection, then the returned values are the minimum
        and maximum of the data.  Defaults to 5.
    krej : float, optional
        The number of sigma used for the rejection. Defaults to 2.5.
    max_iterations : int, optional
        The maximum number of iterations for the rejection. Defaults to
        5.
    """
    n_samples: Incomplete
    contrast: Incomplete
    max_reject: Incomplete
    min_npixels: Incomplete
    krej: Incomplete
    max_iterations: Incomplete
    def __init__(self, n_samples: int = 1000, contrast: float = 0.25, max_reject: float = 0.5, min_npixels: int = 5, krej: float = 2.5, max_iterations: int = 5) -> None: ...
    def get_limits(self, values): ...
