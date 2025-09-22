from _typeshed import Incomplete
from collections.abc import KeysView
from numpy.typing import ArrayLike, NDArray
from typing import Literal

__all__ = ['FitnessFunc', 'Events', 'RegularEvents', 'PointMeasures', 'bayesian_blocks']

def bayesian_blocks(t: ArrayLike, x: ArrayLike | None = None, sigma: ArrayLike | float | None = None, fitness: Literal['events', 'regular_events', 'measures'] | FitnessFunc = 'events', **kwargs) -> NDArray[float]:
    """Compute optimal segmentation of data with Scargle's Bayesian Blocks.

    This is a flexible implementation of the Bayesian Blocks algorithm
    described in Scargle 2013 [1]_.

    Parameters
    ----------
    t : array-like
        data times (one dimensional, length N)
    x : array-like, optional
        data values
    sigma : array-like or float, optional
        data errors
    fitness : str or object
        the fitness function to use for the model.
        If a string, the following options are supported:

        - 'events' : binned or unbinned event data.  Arguments are ``gamma``,
          which gives the slope of the prior on the number of bins, or
          ``ncp_prior``, which is :math:`-\\ln({\\tt gamma})`.
        - 'regular_events' : non-overlapping events measured at multiples of a
          fundamental tick rate, ``dt``, which must be specified as an
          additional argument.  Extra arguments are ``p0``, which gives the
          false alarm probability to compute the prior, or ``gamma``, which
          gives the slope of the prior on the number of bins, or ``ncp_prior``,
          which is :math:`-\\ln({\\tt gamma})`.
        - 'measures' : fitness for a measured sequence with Gaussian errors.
          Extra arguments are ``p0``, which gives the false alarm probability
          to compute the prior, or ``gamma``, which gives the slope of the
          prior on the number of bins, or ``ncp_prior``, which is
          :math:`-\\ln({\\tt gamma})`.

        In all three cases, if more than one of ``p0``, ``gamma``, and
        ``ncp_prior`` is chosen, ``ncp_prior`` takes precedence over ``gamma``
        which takes precedence over ``p0``.

        Alternatively, the fitness parameter can be an instance of
        :class:`FitnessFunc` or a subclass thereof.

    **kwargs :
        any additional keyword arguments will be passed to the specified
        :class:`FitnessFunc` derived class.

    Returns
    -------
    edges : ndarray
        array containing the (N+1) edges defining the N bins

    Examples
    --------
    .. testsetup::

        >>> np.random.seed(12345)

    Event data:

    >>> t = np.random.normal(size=100)
    >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)

    Event data with repeats:

    >>> t = np.random.normal(size=100)
    >>> t[80:] = t[:20]
    >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)

    Regular event data:

    >>> dt = 0.05
    >>> t = dt * np.arange(1000)
    >>> x = np.zeros(len(t))
    >>> x[np.random.randint(0, len(t), len(t) // 10)] = 1
    >>> edges = bayesian_blocks(t, x, fitness='regular_events', dt=dt)

    Measured point data with errors:

    >>> t = 100 * np.random.random(100)
    >>> x = np.exp(-0.5 * (t - 50) ** 2)
    >>> sigma = 0.1
    >>> x_obs = np.random.normal(x, sigma)
    >>> edges = bayesian_blocks(t, x_obs, sigma, fitness='measures')

    References
    ----------
    .. [1] Scargle, J et al. (2013)
       https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S

    .. [2] Bellman, R.E., Dreyfus, S.E., 1962. Applied Dynamic
       Programming. Princeton University Press, Princeton.
       https://press.princeton.edu/books/hardcover/9780691651873/applied-dynamic-programming

    .. [3] Bellman, R., Roth, R., 1969. Curve fitting by segmented
       straight lines. J. Amer. Statist. Assoc. 64, 1079–1084.
       https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501038

    See Also
    --------
    astropy.stats.histogram : compute a histogram using bayesian blocks
    """

class FitnessFunc:
    """Base class for bayesian blocks fitness functions.

    Derived classes should overload the following method:

    ``fitness(self, **kwargs)``:
      Compute the fitness given a set of named arguments.
      Arguments accepted by fitness must be among ``[T_k, N_k, a_k, b_k, c_k]``
      (See [1]_ for details on the meaning of these parameters).

    Additionally, other methods may be overloaded as well:

    ``__init__(self, **kwargs)``:
      Initialize the fitness function with any parameters beyond the normal
      ``p0`` and ``gamma``.

    ``validate_input(self, t, x, sigma)``:
      Enable specific checks of the input data (``t``, ``x``, ``sigma``)
      to be performed prior to the fit.

    ``compute_ncp_prior(self, N)``: If ``ncp_prior`` is not defined explicitly,
      this function is called in order to define it before fitting. This may be
      calculated from ``gamma``, ``p0``, or whatever method you choose.

    ``p0_prior(self, N)``:
      Specify the form of the prior given the false-alarm probability ``p0``
      (See [1]_ for details).

    For examples of implemented fitness functions, see :class:`Events`,
    :class:`RegularEvents`, and :class:`PointMeasures`.

    References
    ----------
    .. [1] Scargle, J et al. (2013)
       https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
    """
    p0: Incomplete
    gamma: Incomplete
    ncp_prior: Incomplete
    def __init__(self, p0: float = 0.05, gamma: float | None = None, ncp_prior: float | None = None) -> None: ...
    def validate_input(self, t: ArrayLike, x: ArrayLike | None = None, sigma: float | ArrayLike | None = None) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
        """Validate inputs to the model.

        Parameters
        ----------
        t : array-like
            times of observations
        x : array-like, optional
            values observed at each time
        sigma : float or array-like, optional
            errors in values x

        Returns
        -------
        t, x, sigma : array-like, float
            validated and perhaps modified versions of inputs
        """
    def fitness(self, **kwargs) -> None: ...
    def p0_prior(self, N: int) -> float:
        '''Empirical prior, parametrized by the false alarm probability ``p0``.

        See eq. 21 in Scargle (2013).

        Note that there was an error in this equation in the original Scargle
        paper (the "log" was missing). The following corrected form is taken
        from https://arxiv.org/abs/1304.2818
        '''
    @property
    def _fitness_args(self) -> KeysView[str]: ...
    def compute_ncp_prior(self, N: int) -> float:
        """
        If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``
        or ``p0``.
        """
    def fit(self, t: ArrayLike, x: ArrayLike | None = None, sigma: ArrayLike | float | None = None) -> NDArray[float]:
        """Fit the Bayesian Blocks model given the specified fitness function.

        Parameters
        ----------
        t : array-like
            data times (one dimensional, length N)
        x : array-like, optional
            data values
        sigma : array-like or float, optional
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """

class Events(FitnessFunc):
    """Bayesian blocks fitness for binned or unbinned events.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\\rm blocks}` (see eq. 21 of Scargle 2013). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \\sim {\\tt gamma}^{N_{\\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\\tt ncp\\_prior} = -\\ln({\\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """
    def fitness(self, N_k: NDArray[float], T_k: NDArray[float]) -> NDArray[float]: ...
    def validate_input(self, t: ArrayLike, x: ArrayLike | None, sigma: float | ArrayLike | None) -> tuple[NDArray[float], NDArray[float], NDArray[float]]: ...

class RegularEvents(FitnessFunc):
    '''Bayesian blocks fitness for regular events.

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\\rm
        blocks}` (see eq. 21 of Scargle 2013). If gamma is specified, p0 is
        ignored.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \\sim {\\tt gamma}^{N_{\\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\\tt ncp\\_prior} = -\\ln({\\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    '''
    dt: Incomplete
    def __init__(self, dt: float, p0: float = 0.05, gamma: float | None = None, ncp_prior: float | None = None) -> None: ...
    def validate_input(self, t: ArrayLike, x: ArrayLike | None = None, sigma: float | ArrayLike | None = None) -> tuple[NDArray[float], NDArray[float], NDArray[float]]: ...
    def fitness(self, T_k: NDArray[float], N_k: NDArray[float]) -> NDArray[float]: ...

class PointMeasures(FitnessFunc):
    """Bayesian blocks fitness for point measures.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\\rm
        blocks}` (see eq. 21 of Scargle 2013). If gamma is specified, p0 is
        ignored.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \\sim {\\tt gamma}^{N_{\\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\\tt ncp\\_prior} = -\\ln({\\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """
    def __init__(self, p0: float = 0.05, gamma: float | None = None, ncp_prior: float | None = None) -> None: ...
    def fitness(self, a_k: NDArray[float], b_k: ArrayLike) -> NDArray[float]: ...
    def validate_input(self, t: ArrayLike, x: ArrayLike | None, sigma: float | ArrayLike | None) -> tuple[NDArray[float], NDArray[float], NDArray[float]]: ...
