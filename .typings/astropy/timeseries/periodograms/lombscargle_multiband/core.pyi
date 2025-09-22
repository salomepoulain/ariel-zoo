from _typeshed import Incomplete
from astropy.timeseries.periodograms.lombscargle import LombScargle

__all__ = ['LombScargleMultiband']

class LombScargleMultiband(LombScargle):
    '''Compute the Lomb-Scargle Periodogram.

    This implementation is based on code presented in [1]_ and [2]_;
    if you use this functionality in an academic application, citation of
    those works would be appreciated.

    Parameters
    ----------
    t : array-like or `~astropy.units.Quantity` [\'time\']
        sequence of observation times
    y : array-like or `~astropy.units.Quantity`
        sequence of observations associated with times t
    bands : array-like
        sequence of passband labels associated with times t, each unique label
        defines a single band of data.
    dy : float, array-like, or `~astropy.units.Quantity`, optional
        error or sequence of observational errors associated with times t
    normalization : {\'standard\', \'model\', \'log\', \'psd\'}, optional
        Normalization to use for the periodogram.
    nterms_base : int, optional
        number of frequency terms to use for the base model common to all bands.
        In the case of the fast algorithm, this parameter is passed along to
        the single band LombScargle method as the ``nterms`` parameter.
    nterms_band : int, optional
        number of frequency terms to use for the residuals between the base
        model and each individual band
    reg_base : float or None (default = None)
        amount of regularization to use on the base model parameters
    reg_band : float or None (default = 1E-6)
        amount of regularization to use on the band model parameters
    regularize_by_trace : bool (default = True)
        if True, then regularization is expressed in units of the trace of
        the normal matrix
    center_data : bool, optional
        if True, pre-center the data by subtracting the weighted mean
        of the input data.
    fit_mean : bool, optional
        if True, include a constant offset as part of the model at each
        frequency. This can lead to more accurate results, especially in the
        case of incomplete phase coverage. Only applicable to the "fast" method

    References
    ----------
    .. [1] Vanderplas, J., Connolly, A. Ivezic, Z. & Gray, A. *Introduction to
        astroML: Machine learning for astrophysics*. Proceedings of the
        Conference on Intelligent Data Understanding (2012)
    .. [2] VanderPlas, J. & Ivezic, Z. *Periodograms for Multiband Astronomical
        Time Series*. ApJ 812.1:18 (2015)
    '''
    available_methods: Incomplete
    t: Incomplete
    _tstart: Incomplete
    normalization: Incomplete
    nterms_base: Incomplete
    nterms_band: Incomplete
    reg_base: Incomplete
    reg_band: Incomplete
    regularize_by_trace: Incomplete
    center_data: Incomplete
    fit_mean: Incomplete
    nterms: Incomplete
    def __init__(self, t, y, bands, dy: Incomplete | None = None, normalization: str = 'standard', nterms_base: int = 1, nterms_band: int = 1, reg_base: Incomplete | None = None, reg_band: float = 1e-06, regularize_by_trace: bool = True, center_data: bool = True, fit_mean: bool = True) -> None: ...
    @classmethod
    def from_timeseries(cls, timeseries, signal_column: Incomplete | None = None, uncertainty_column: Incomplete | None = None, band_labels: Incomplete | None = None, **kwargs):
        """
        Initialize a multiband periodogram from a time series object.

        If a binned time series is passed, the time at the center of the bins is
        used. Also note that this method automatically gets rid of NaN/undefined
        values when initializing the periodogram.

        Parameters
        ----------
        signal_column : list
            The names of columns containing the signal values to use.
        uncertainty_column : list, optional
            The names of columns containing the errors on the signal.
        band_labels : list, optional
            The labels for each band, matched by index. If none, uses the
            labels of ``signal_column`` as band names.
        **kwargs
            Additional keyword arguments are passed to the initializer for this
            periodogram class.
        """
    def _validate_inputs(self, t, y, bands, dy): ...
    def autofrequency(self, samples_per_peak: int = 5, nyquist_factor: int = 5, minimum_frequency: Incomplete | None = None, maximum_frequency: Incomplete | None = None, return_freq_limits: bool = False):
        '''Determine a suitable frequency grid for data.
        Note that this assumes the peak width is driven by the observational
        baseline, which is generally a good assumption when the baseline is
        much larger than the oscillation period.
        If you are searching for periods longer than the baseline of your
        observations, this may not perform well.
        Even with a large baseline, be aware that the maximum frequency
        returned is based on the concept of "average Nyquist frequency", which
        may not be useful for irregularly-sampled data. The maximum frequency
        can be adjusted via the nyquist_factor argument, or through the
        maximum_frequency argument.

        Parameters
        ----------
        samples_per_peak : float, optional
            The approximate number of desired samples across the typical peak
        nyquist_factor : float, optional
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        minimum_frequency : float, optional
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float, optional
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.
        return_freq_limits : bool, optional
            if True, return only the frequency limits rather than the full
            frequency grid.

        Returns
        -------
        frequency : ndarray or `~astropy.units.Quantity` [\'frequency\']
            The heuristically-determined optimal frequency bin
        '''
    def autopower(self, method: str = 'flexible', sb_method: str = 'auto', normalization: str = 'standard', samples_per_peak: int = 5, nyquist_factor: int = 5, minimum_frequency: Incomplete | None = None, maximum_frequency: Incomplete | None = None):
        """Compute Lomb-Scargle power at automatically-determined frequencies.

        Parameters
        ----------
        method : str, optional
            specify the multi-band lomb scargle implementation to use. Options are:

            - 'flexible': Constructs a common model, and an offset model per individual
                band. Applies regularization to the resulting model to constrain
                complexity.
            - 'fast': Passes each band individually through LombScargle (single-band),
                combines periodograms at the end by weight. Speed depends on single-band
                method chosen in 'sb_method'.

        sb_method : str, optional
            specify the single-band lomb scargle implementation to use, only in
            the case of using the 'fast' multiband method. Options are:

            - 'auto': choose the best method based on the input
            - 'fast': use the O[N log N] fast method. Note that this requires
              evenly-spaced frequencies: by default this will be checked unless
              ``assume_regular_frequency`` is set to True.
            - 'slow': use the O[N^2] pure-python implementation
            - 'cython': use the O[N^2] cython implementation. This is slightly
              faster than method='slow', but much more memory efficient.
            - 'chi2': use the O[N^2] chi2/linear-fitting implementation
            - 'fastchi2': use the O[N log N] chi2 implementation. Note that this
              requires evenly-spaced frequencies: by default this will be checked
              unless ``assume_regular_frequency`` is set to True.
            - 'scipy': use ``scipy.signal.lombscargle``, which is an O[N^2]
              implementation written in C. Note that this does not support
              heteroskedastic errors.

        normalization : {'standard', 'model', 'log', 'psd'}, optional
            If specified, override the normalization specified at instantiation.
        samples_per_peak : float, optional
            The approximate number of desired samples across the typical peak
        nyquist_factor : float, optional
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        minimum_frequency : float or `~astropy.units.Quantity` ['frequency'], optional
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline. Should be `~astropy.units.Quantity`
            if inputs to LombScargle are `~astropy.units.Quantity`.
        maximum_frequency : float or `~astropy.units.Quantity` ['frequency'], optional
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency. Should be `~astropy.units.Quantity`
            if inputs to LombScargle are `~astropy.units.Quantity`.

        Returns
        -------
        frequency, power : ndarray
            The frequency and Lomb-Scargle power
        """
    def power(self, frequency, method: str = 'flexible', sb_method: str = 'auto', normalization: str = 'standard'):
        """Compute the Lomb-Scargle power at the given frequencies.

        Parameters
        ----------
        frequency : array-like or `~astropy.units.Quantity` ['frequency']
            frequencies (not angular frequencies) at which to evaluate the
            periodogram. Note that in order to use method='fast', frequencies
            must be regularly-spaced.
        method : str, optional
            specify the multi-band lomb scargle implementation to use. Options are:

            - 'flexible': Constructs a common model, and an offset model per individual
                band. Applies regularization to the resulting model to constrain
                complexity.
            - 'fast': Passes each band individually through LombScargle (single-band),
                combines periodograms at the end by weight. Speed depends on single-band
                method chosen in 'sb_method'.
        sb_method : str, optional
            specify the single-band lomb scargle implementation to use, only in
            the case of using the 'fast' multiband method. Options can be found
            in `~astropy.timeseries.LombScargle`.
        normalization : {'standard', 'model', 'log', 'psd'}, optional
            If specified, override the normalization specified at instantiation.

        Returns
        -------
        power : ndarray
            The Lomb-Scargle power at the specified frequency
        """
    def design_matrix(self, frequency, t_fit: Incomplete | None = None, bands_fit: Incomplete | None = None):
        """Compute the design matrix for a given frequency

        Parameters
        ----------
        frequency : float
            the frequency for the model
        t_fit : array-like, `~astropy.units.Quantity`, or `~astropy.time.Time` (optional)
            Times (length ``n_samples``) at which to compute the model.
            If not specified, then the times and uncertainties of the input
            data are used.
        bands_fit : array-like, or str
            Bands to use in fitting, must be subset of bands in input data.

        Returns
        -------
        ndarray
            The design matrix for the model at the given frequency.
            This should have a shape of (``len(t)``, ``n_parameters``).

        See Also
        --------
        model
        model_parameters
        offset
        """
    def model(self, t, frequency, bands_fit: Incomplete | None = None):
        """Compute the Lomb-Scargle model at the given frequency.

        The model at a particular frequency is a linear model:
        model = offset + dot(design_matrix, model_parameters)

        Parameters
        ----------
        t : array-like or `~astropy.units.Quantity` ['time']
            Times (length ``n_samples``) at which to compute the model.
        frequency : float
            the frequency for the model
        bands_fit : list or array-like
            the unique bands to fit in the model

        Returns
        -------
        y : np.ndarray
            The model fit corresponding to the input times.
            Will have shape (``n_bands``,``n_samples``).

        See Also
        --------
        design_matrix
        offset
        model_parameters
        """
    def offset(self, t_fit: Incomplete | None = None, bands_fit: Incomplete | None = None):
        """Return the offset array of the model

        The offset array of the model is the (weighted) mean of the y values in each band.
        Note that if self.center_data is False, the offset is 0 by definition.

        Parameters
        ----------
        t_fit : array-like, `~astropy.units.Quantity`, or `~astropy.time.Time` (optional)
            Times (length ``n_samples``) at which to compute the model.
            If not specified, then the times and uncertainties of the input
            data are used.
        bands_fit : array-like, or str
            Bands to use in fitting, must be subset of bands in input data.

        Returns
        -------
        offset : array

        See Also
        --------
        design_matrix
        model
        model_parameters
        """
    def model_parameters(self, frequency, units: bool = True):
        """Compute the best-fit model parameters at the given frequency.

        The model described by these parameters is:

        .. math::

            y(t; f, \\vec{\\theta}) = \\theta_0 + \\sum_{n=1}^{\\tt nterms_base} [\\theta_{2n-1}\\sin(2\\pi n f t) + \\theta_{2n}\\cos(2\\pi n f t)]
            + \\theta_0^{(k)} + \\sum_{n=1}^{\\tt nterms_band} [\\theta_{2n-1}^{(k)}\\sin(2\\pi n f t) +

        where :math:`\\vec{\\theta}` is the array of parameters returned by this function.

        Parameters
        ----------
        frequency : float
            the frequency for the model
        units : bool
            If True (default), return design matrix with data units.

        Returns
        -------
        theta : np.ndarray (n_parameters,)
            The best-fit model parameters at the given frequency.

        See Also
        --------
        design_matrix
        model
        offset
        """
    def false_alarm_probability(self) -> None:
        """Not Implemented"""
    def false_alarm_level(self) -> None:
        """Not Implemented"""
    def distribution(self) -> None:
        """Not Implemented"""
