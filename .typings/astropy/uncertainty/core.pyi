import numpy as np
from _typeshed import Incomplete

__all__ = ['Distribution']

class Distribution:
    """A scalar value or array values with associated uncertainty distribution.

    This object will take its exact type from whatever the ``samples``
    argument is. In general this is expected to be ``NdarrayDistribution`` for
    |ndarray| input, and, e.g., ``QuantityDistribution`` for a subclass such
    as |Quantity|. But anything compatible with `numpy.asanyarray` is possible
    (generally producing ``NdarrayDistribution``).

    See also: https://docs.astropy.org/en/stable/uncertainty/

    Parameters
    ----------
    samples : array-like
        The distribution, with sampling along the *trailing* axis. If 1D, the sole
        dimension is used as the sampling axis (i.e., it is a scalar distribution).
        If an |ndarray| or subclass, the data will not be copied unless it is not
        possible to take a view (generally, only when the strides of the last axis
        are negative).

    """
    _generated_subclasses: Incomplete
    def __new__(cls, samples): ...
    @classmethod
    def _get_distribution_cls(cls, samples_cls): ...
    @classmethod
    def _get_distribution_dtype(cls, dtype, n_samples, itemsize: Incomplete | None = None): ...
    @property
    def distribution(self): ...
    @property
    def dtype(self): ...
    @dtype.setter
    def dtype(self, dtype) -> None: ...
    def astype(self, dtype, *args, **kwargs): ...
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...
    def __array_function__(self, function, types, args, kwargs): ...
    def _result_as_distribution(self, result, out, ncore_out: Incomplete | None = None, axis: Incomplete | None = None):
        """Turn result into a distribution.

        If no output is given, it will create a Distribution from the array,
        If an output is given, it should be fine as is.

        Parameters
        ----------
        result : ndarray or tuple thereof
            Array(s) which need to be turned into Distribution.
        out : Distribution, tuple of Distribution or None
            Possible output |Distribution|. Should be `None` or a tuple if result
            is a tuple.
        ncore_out: int or tuple thereof
            The number of core dimensions for the output array for a gufunc.  This
            is used to determine which axis should be used for the samples.
        axis: int or None
            The axis a gufunc operated on.  Used only if ``ncore_out`` is given.

        Returns
        -------
        out : Distribution
        """
    def _not_implemented_or_raise(self, function, types): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    @property
    def n_samples(self):
        """
        The number of samples of this distribution.  A single `int`.
        """
    def pdf_mean(self, dtype: Incomplete | None = None, out: Incomplete | None = None):
        """
        The mean of this distribution.

        Arguments are as for `numpy.mean`.
        """
    def pdf_std(self, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0):
        """
        The standard deviation of this distribution.

        Arguments are as for `numpy.std`.
        """
    def pdf_var(self, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0):
        """
        The variance of this distribution.

        Arguments are as for `numpy.var`.
        """
    def pdf_median(self, out: Incomplete | None = None):
        """
        The median of this distribution.

        Parameters
        ----------
        out : array, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output,
            but the type (of the output) will be cast if necessary.
        """
    def pdf_mad(self, out: Incomplete | None = None):
        """
        The median absolute deviation of this distribution.

        Parameters
        ----------
        out : array, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output,
            but the type (of the output) will be cast if necessary.
        """
    def pdf_smad(self, out: Incomplete | None = None):
        """
        The median absolute deviation of this distribution rescaled to match the
        standard deviation for a normal distribution.

        Parameters
        ----------
        out : array, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output,
            but the type (of the output) will be cast if necessary.
        """
    def pdf_percentiles(self, percentile, **kwargs):
        """
        Compute percentiles of this Distribution.

        Parameters
        ----------
        percentile : float or array of float or `~astropy.units.Quantity`
            The desired percentiles of the distribution (i.e., on [0,100]).
            `~astropy.units.Quantity` will be converted to percent, meaning
            that a ``dimensionless_unscaled`` `~astropy.units.Quantity` will
            be interpreted as a quantile.

        Additional keywords are passed into `numpy.percentile`.

        Returns
        -------
        percentiles : `~astropy.units.Quantity` ['dimensionless']
            The ``fracs`` percentiles of this distribution.
        """
    def pdf_histogram(self, **kwargs):
        """
        Compute histogram over the samples in the distribution.

        Parameters
        ----------
        All keyword arguments are passed into `astropy.stats.histogram`. Note
        That some of these options may not be valid for some multidimensional
        distributions.

        Returns
        -------
        hist : array
            The values of the histogram. Trailing dimension is the histogram
            dimension.
        bin_edges : array of dtype float
            Return the bin edges ``(length(hist)+1)``. Trailing dimension is the
            bin histogram dimension.
        """

class ScalarDistribution(Distribution, np.void):
    """Scalar distribution.

    This class mostly exists to make `~numpy.array2print` possible for
    all subclasses.  It is a scalar element, still with n_samples samples.
    """

class ArrayDistribution(Distribution, np.ndarray):
    _samples_cls = np.ndarray
    def view(self, dtype: Incomplete | None = None, type: Incomplete | None = None):
        """New view of array with the same data.

        Like `~numpy.ndarray.view` except that the result will always be a new
        `~astropy.uncertainty.Distribution` instance.  If the requested
        ``type`` is a `~astropy.uncertainty.Distribution`, then no change in
        ``dtype`` is allowed.

        """
    @property
    def distribution(self): ...
    def __getitem__(self, item): ...
    def __setitem__(self, item, value) -> None: ...

class _DistributionRepr:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_latex_(self): ...

class NdarrayDistribution(_DistributionRepr, ArrayDistribution): ...
