import abc
from ._fitting_parallel import parallel_fit_dask as parallel_fit_dask
from .spline import SplineExactKnotsFitter as SplineExactKnotsFitter, SplineInterpolateFitter as SplineInterpolateFitter, SplineSmoothingFitter as SplineSmoothingFitter, SplineSplrepFitter as SplineSplrepFitter
from _typeshed import Incomplete

__all__ = ['LinearLSQFitter', 'LevMarLSQFitter', 'TRFLSQFitter', 'DogBoxLSQFitter', 'LMLSQFitter', 'FittingWithOutlierRemoval', 'SLSQPLSQFitter', 'SimplexLSQFitter', 'JointFitter', 'Fitter', 'ModelLinearityError', 'ModelsError', 'SplineExactKnotsFitter', 'SplineInterpolateFitter', 'SplineSmoothingFitter', 'SplineSplrepFitter', 'parallel_fit_dask']

class NonFiniteValueError(RuntimeError):
    """
    Error raised when attempting to a non-finite value.
    """

class Covariance:
    """Class for covariance matrix calculated by fitter."""
    cov_matrix: Incomplete
    param_names: Incomplete
    def __init__(self, cov_matrix, param_names) -> None: ...
    def pprint(self, max_lines, round_val): ...
    def __repr__(self) -> str: ...
    def __getitem__(self, params): ...

class StandardDeviations:
    """Class for fitting uncertainties."""
    param_names: Incomplete
    stds: Incomplete
    def __init__(self, cov_matrix, param_names) -> None: ...
    def _calc_stds(self, cov_matrix): ...
    def pprint(self, max_lines, round_val): ...
    def __repr__(self) -> str: ...
    def __getitem__(self, param): ...

class ModelsError(Exception):
    """Base class for model exceptions."""
class ModelLinearityError(ModelsError):
    """Raised when a non-linear model is passed to a linear fitter."""
class UnsupportedConstraintError(ModelsError, ValueError):
    """
    Raised when a fitter does not support a type of constraint.
    """

class _FitterMeta(abc.ABCMeta):
    """
    Currently just provides a registry for all Fitter classes.
    """
    registry: Incomplete
    def __new__(mcls, name, bases, members): ...

class Fitter(metaclass=_FitterMeta):
    """
    Base class for all fitters.

    Parameters
    ----------
    optimizer : callable
        A callable implementing an optimization algorithm
    statistic : callable
        Statistic function

    """
    supported_constraints: Incomplete
    _opt_method: Incomplete
    _stat_method: Incomplete
    def __init__(self, optimizer, statistic) -> None: ...
    def objective_function(self, fps, *args):
        """
        Function to minimize.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [other_args], [input coordinates]]
            other_args may include weights or any other quantities specific for
            a statistic

        Notes
        -----
        The list of arguments (args) is set in the `__call__` method.
        Fitters may overwrite this method, e.g. when statistic functions
        require other arguments.

        """
    @staticmethod
    def _add_fitting_uncertainties(*args) -> None:
        """
        When available, calculate and sets the parameter covariance matrix
        (model.cov_matrix) and standard deviations (model.stds).
        """
    @abc.abstractmethod
    def __call__(self):
        """
        This method performs the actual fitting and modifies the parameter list
        of a model.
        Fitter subclasses should implement this method.
        """

class LinearLSQFitter(metaclass=_FitterMeta):
    """
    A class performing a linear least square fitting.
    Uses `numpy.linalg.lstsq` to do the fitting.
    Given a model and data, fits the model to the data and changes the
    model's parameters. Keeps a dictionary of auxiliary fitting information.

    Notes
    -----
    Note that currently LinearLSQFitter does not support compound models.
    """
    supported_constraints: Incomplete
    supports_masked_input: bool
    fit_info: Incomplete
    _calc_uncertainties: Incomplete
    def __init__(self, calc_uncertainties: bool = False) -> None: ...
    @staticmethod
    def _is_invertible(m):
        """Check if inverse of matrix can be obtained."""
    def _add_fitting_uncertainties(self, model, a, n_coeff, x, y, z: Incomplete | None = None, resids: Incomplete | None = None):
        """
        Calculate and parameter covariance matrix and standard deviations
        and set `cov_matrix` and `stds` attributes.
        """
    @staticmethod
    def _deriv_with_constraints(model, param_indices, x: Incomplete | None = None, y: Incomplete | None = None): ...
    def _map_domain_window(self, model, x, y: Incomplete | None = None):
        """
        Maps domain into window for a polynomial model which has these
        attributes.
        """
    def __call__(self, model, x, y, z: Incomplete | None = None, weights: Incomplete | None = None, rcond: Incomplete | None = None, *, inplace: bool = False):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
            Input coordinates
        y : array-like
            Input coordinates
        z : array-like, optional
            Input coordinates.
            If the dependent (``y`` or ``z``) coordinate values are provided
            as a `numpy.ma.MaskedArray`, any masked points are ignored when
            fitting. Note that model set fitting is significantly slower when
            there are masked points (not just an empty mask), as the matrix
            equation has to be solved for each model separately when their
            coordinate grids differ.
        weights : array, optional
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        rcond :  float, optional
            Cut-off ratio for small singular values of ``a``.
            Singular values are set to zero if they are smaller than ``rcond``
            times the largest singular value of ``a``.
        equivalencies : list or None, optional, keyword-only
            List of *additional* equivalencies that are should be applied in
            case x, y and/or z have units. Default is None.
        inplace : bool, optional
            If `False` (the default), a copy of the model with the fitted
            parameters set will be returned. If `True`, the returned model will
            be the same instance as the model passed in, and the parameter
            values will be changed inplace.

        Returns
        -------
        fitted_model : `~astropy.modeling.FittableModel`
            If ``inplace`` is `False` (the default), this is a copy of the
            input model with parameters set by the fitter. If ``inplace`` is
            `True`, this is the same model as the input model, with parameters
            updated to be those set by the fitter.

        """

class FittingWithOutlierRemoval:
    """
    This class combines an outlier removal technique with a fitting procedure.
    Basically, given a maximum number of iterations ``niter``, outliers are
    removed and fitting is performed for each iteration, until no new outliers
    are found or ``niter`` is reached.

    Parameters
    ----------
    fitter : `Fitter`
        An instance of any Astropy fitter, i.e., LinearLSQFitter,
        LevMarLSQFitter, SLSQPLSQFitter, SimplexLSQFitter, JointFitter. For
        model set fitting, this must understand masked input data (as
        indicated by the fitter class attribute ``supports_masked_input``).
    outlier_func : callable
        A function for outlier removal.
        If this accepts an ``axis`` parameter like the `numpy` functions, the
        appropriate value will be supplied automatically when fitting model
        sets (unless overridden in ``outlier_kwargs``), to find outliers for
        each model separately; otherwise, the same filtering must be performed
        in a loop over models, which is almost an order of magnitude slower.
    niter : int, optional
        Maximum number of iterations.
    outlier_kwargs : dict, optional
        Keyword arguments for outlier_func.

    Attributes
    ----------
    fit_info : dict
        The ``fit_info`` (if any) from the last iteration of the wrapped
        ``fitter`` during the most recent fit. An entry is also added with the
        keyword ``niter`` that records the actual number of fitting iterations
        performed (as opposed to the user-specified maximum).
    """
    fitter: Incomplete
    outlier_func: Incomplete
    niter: Incomplete
    outlier_kwargs: Incomplete
    fit_info: Incomplete
    def __init__(self, fitter, outlier_func, niter: int = 3, **outlier_kwargs) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __call__(self, model, x, y, z: Incomplete | None = None, weights: Incomplete | None = None, *, inplace: bool = False, **kwargs):
        """
        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            An analytic model which will be fit to the provided data.
            This also contains the initial guess for an optimization
            algorithm.
        x : array-like
            Input coordinates.
        y : array-like
            Data measurements (1D case) or input coordinates (2D case).
        z : array-like, optional
            Data measurements (2D case).
        weights : array-like, optional
            Weights to be passed to the fitter.
        kwargs : dict, optional
            Keyword arguments to be passed to the fitter.
        inplace : bool, optional
            If `False` (the default), a copy of the model with the fitted
            parameters set will be returned. If `True`, the returned model will
            be the same instance as the model passed in, and the parameter
            values will be changed inplace.

        Returns
        -------
        fitted_model : `~astropy.modeling.FittableModel`
            If ``inplace`` is `False` (the default), this is a copy of the
            input model with parameters set by the fitter. If ``inplace`` is
            `True`, this is the same model as the input model, with parameters
            updated to be those set by the fitter.
        mask : `numpy.ndarray`
            Boolean mask array, identifying which points were used in the final
            fitting iteration (False) and which were found to be outliers or
            were masked in the input (True).
        """

class _NonLinearLSQFitter(metaclass=_FitterMeta):
    """
    Base class for Non-Linear least-squares fitters.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds : bool
        If set, the parameter bounds for a model will be enforced for each given
        parameter while fitting via a simple min/max condition.
        Default: True
    """
    supported_constraints: Incomplete
    fit_info: Incomplete
    _calc_uncertainties: Incomplete
    _use_min_max_bounds: Incomplete
    def __init__(self, calc_uncertainties: bool = False, use_min_max_bounds: bool = True) -> None: ...
    def objective_function(self, fps, *args, fit_param_indices: Incomplete | None = None):
        """
        Function to minimize.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [weights], [input coordinates]]
        fit_param_indices : list, optional
            The ``fit_param_indices`` as returned by ``model_to_fit_params``.
            This is a list of the parameter indices being fit, so excluding any
            tied or fixed parameters.  This can be passed in to the objective
            function to prevent it having to be computed on every call.
            This must be optional as not all fitters support passing kwargs to
            the objective function.
        """
    @staticmethod
    def _add_fitting_uncertainties(model, cov_matrix) -> None:
        """
        Set ``cov_matrix`` and ``stds`` attributes on model with parameter
        covariance matrix returned by ``optimize.leastsq``.
        """
    @staticmethod
    def _wrap_deriv(params, model, weights, x, y, z: Incomplete | None = None, fit_param_indices: Incomplete | None = None):
        """
        Wraps the method calculating the Jacobian of the function to account
        for model constraints.
        `scipy.optimize.leastsq` expects the function derivative to have the
        above signature (parlist, (argtuple)). In order to accommodate model
        constraints, instead of using p directly, we set the parameter list in
        this function.
        """
    def _compute_param_cov(self, model, y, init_values, cov_x, fitparams, farg, fkwarg, weights: Incomplete | None = None) -> None: ...
    def _run_fitter(self, model, farg, fkwarg, maxiter, acc, epsilon, estimate_jacobian): ...
    def _filter_non_finite(self, x, y, z: Incomplete | None = None, weights: Incomplete | None = None):
        """
        Filter out non-finite values in x, y, z.

        Returns
        -------
        x, y, z : ndarrays
            x, y, and z with non-finite values filtered out.
        """
    def __call__(self, model, x, y, z: Incomplete | None = None, weights: Incomplete | None = None, maxiter=..., acc=..., epsilon=..., estimate_jacobian: bool = False, filter_non_finite: bool = False, *, inplace: bool = False):
        '''
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
           input coordinates
        y : array
           input coordinates
        z : array, optional
           input coordinates
        weights : array, optional
            Weights for fitting. For data with Gaussian uncertainties, the weights
            should be 1/sigma.

            .. versionchanged:: 5.3
                Calculate parameter covariances while accounting for ``weights``
                as "absolute" inverse uncertainties. To recover the old behavior,
                choose ``weights=None``.

        maxiter : int
            maximum number of iterations
        acc : float
            Relative error desired in the approximate solution
        epsilon : float
            A suitable step length for the forward-difference
            approximation of the Jacobian (if model.fjac=None). If
            epsfcn is less than the machine precision, it is
            assumed that the relative errors in the functions are
            of the order of the machine precision.
        estimate_jacobian : bool
            If False (default) and if the model has a fit_deriv method,
            it will be used. Otherwise the Jacobian will be estimated.
            If True, the Jacobian will be estimated in any case.
        equivalencies : list or None, optional, keyword-only
            List of *additional* equivalencies that are should be applied in
            case x, y and/or z have units. Default is None.
        filter_non_finite : bool, optional
            Whether or not to filter data with non-finite values. Default is False
        inplace : bool, optional
            If `False` (the default), a copy of the model with the fitted
            parameters set will be returned. If `True`, the returned model will
            be the same instance as the model passed in, and the parameter
            values will be changed inplace.

        Returns
        -------
        fitted_model : `~astropy.modeling.FittableModel`
            If ``inplace`` is `False` (the default), this is a copy of the
            input model with parameters set by the fitter. If ``inplace`` is
            `True`, this is the same model as the input model, with parameters
            updated to be those set by the fitter.

        '''

class LevMarLSQFitter(_NonLinearLSQFitter):
    """
    Levenberg-Marquardt algorithm and least squares statistic.

    .. warning:

        This fitter is no longer recommended - instead you should make use of
        `LMLSQFitter` if your model does not have bounds, or one of the other
        non-linear fitters, such as `TRFLSQFitter` otherwise. For more details,
        see the main documentation page on fitting.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info : dict
        The `scipy.optimize.leastsq` result for the most recent fit (see
        notes).

    Notes
    -----
    The ``fit_info`` dictionary contains the values returned by
    `scipy.optimize.leastsq` for the most recent fit, including the values from
    the ``infodict`` dictionary it returns. See the `scipy.optimize.leastsq`
    documentation for details on the meaning of these values. Note that the
    ``x`` return value is *not* included (as it is instead the parameter values
    of the returned model).
    Additionally, one additional element of ``fit_info`` is computed whenever a
    model is fit, with the key 'param_cov'. The corresponding value is the
    covariance matrix of the parameters as a 2D numpy array.  The order of the
    matrix elements matches the order of the parameters in the fitted model
    (i.e., the same order as ``model.param_names``).

    """
    fit_info: Incomplete
    def __init__(self, calc_uncertainties: bool = False) -> None: ...
    def _run_fitter(self, model, farg, fkwarg, maxiter, acc, epsilon, estimate_jacobian): ...

class _NLLSQFitter(_NonLinearLSQFitter):
    """
    Wrapper class for `scipy.optimize.least_squares` method, which provides:
        - Trust Region Reflective
        - dogbox
        - Levenberg-Marquardt
    algorithms using the least squares statistic.

    Parameters
    ----------
    method : str
        ‘trf’ :  Trust Region Reflective algorithm, particularly suitable
            for large sparse problems with bounds. Generally robust method.
        ‘dogbox’ : dogleg algorithm with rectangular trust regions, typical
            use case is small problems with bounds. Not recommended for
            problems with rank-deficient Jacobian.
        ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK.
            Doesn’t handle bounds and sparse Jacobians. Usually the most
            efficient method for small unconstrained problems.
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds: bool
        If set, the parameter bounds for a model will be enforced for each given
        parameter while fitting via a simple min/max condition. A True setting
        will replicate how LevMarLSQFitter enforces bounds.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """
    _method: Incomplete
    def __init__(self, method, calc_uncertainties: bool = False, use_min_max_bounds: bool = False) -> None: ...
    fit_info: Incomplete
    def _run_fitter(self, model, farg, fkwarg, maxiter, acc, epsilon, estimate_jacobian): ...

class TRFLSQFitter(_NLLSQFitter):
    """
    Trust Region Reflective algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """
    def __init__(self, calc_uncertainties: bool = False, use_min_max_bounds: bool = False) -> None: ...

class DogBoxLSQFitter(_NLLSQFitter):
    """
    DogBox algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """
    def __init__(self, calc_uncertainties: bool = False, use_min_max_bounds: bool = False) -> None: ...

class LMLSQFitter(_NLLSQFitter):
    """
    `scipy.optimize.least_squares` Levenberg-Marquardt algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """
    def __init__(self, calc_uncertainties: bool = False) -> None: ...
    def __call__(self, model, x, y, z: Incomplete | None = None, weights: Incomplete | None = None, maxiter=..., acc=..., epsilon=..., estimate_jacobian: bool = False, filter_non_finite: bool = False, inplace: bool = False): ...

class SLSQPLSQFitter(Fitter):
    """
    Sequential Least Squares Programming (SLSQP) optimization algorithm and
    least squares statistic.

    Raises
    ------
    ModelLinearityError
        A linear model is passed to a nonlinear fitter

    Notes
    -----
    See also the `~astropy.modeling.optimizers.SLSQP` optimizer.

    """
    supported_constraints: Incomplete
    fit_info: Incomplete
    def __init__(self) -> None: ...
    def __call__(self, model, x, y, z: Incomplete | None = None, weights: Incomplete | None = None, *, inplace: bool = False, **kwargs):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
            input coordinates
        y : array
            input coordinates
        z : array, optional
            input coordinates
        weights : array, optional
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        inplace : bool, optional
            If `False` (the default), a copy of the model with the fitted
            parameters set will be returned. If `True`, the returned model will
            be the same instance as the model passed in, and the parameter
            values will be changed inplace.
        kwargs : dict
            optional keyword arguments to be passed to the optimizer or the statistic
        verblevel : int
            0-silent
            1-print summary upon completion,
            2-print summary after each iteration
        maxiter : int
            maximum number of iterations
        epsilon : float
            the step size for finite-difference derivative estimates
        acc : float
            Requested accuracy
        equivalencies : list or None, optional, keyword-only
            List of *additional* equivalencies that are should be applied in
            case x, y and/or z have units. Default is None.

        Returns
        -------
        fitted_model : `~astropy.modeling.FittableModel`
            If ``inplace`` is `False` (the default), this is a copy of the
            input model with parameters set by the fitter. If ``inplace`` is
            `True`, this is the same model as the input model, with parameters
            updated to be those set by the fitter.

        """

class SimplexLSQFitter(Fitter):
    """
    Simplex algorithm and least squares statistic.

    Raises
    ------
    `ModelLinearityError`
        A linear model is passed to a nonlinear fitter

    """
    supported_constraints: Incomplete
    fit_info: Incomplete
    def __init__(self) -> None: ...
    def __call__(self, model, x, y, z: Incomplete | None = None, weights: Incomplete | None = None, *, inplace: bool = False, **kwargs):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
            input coordinates
        y : array
            input coordinates
        z : array, optional
            input coordinates
        weights : array, optional
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        kwargs : dict
            optional keyword arguments to be passed to the optimizer or the statistic
        maxiter : int
            maximum number of iterations
        acc : float
            Relative error in approximate solution
        equivalencies : list or None, optional, keyword-only
            List of *additional* equivalencies that are should be applied in
            case x, y and/or z have units. Default is None.
        inplace : bool, optional
            If `False` (the default), a copy of the model with the fitted
            parameters set will be returned. If `True`, the returned model will
            be the same instance as the model passed in, and the parameter
            values will be changed inplace.

        Returns
        -------
        fitted_model : `~astropy.modeling.FittableModel`
            If ``inplace`` is `False` (the default), this is a copy of the
            input model with parameters set by the fitter. If ``inplace`` is
            `True`, this is the same model as the input model, with parameters
            updated to be those set by the fitter.

        """

class JointFitter(metaclass=_FitterMeta):
    """
    Fit models which share a parameter.
    For example, fit two gaussians to two data sets but keep
    the FWHM the same.

    Parameters
    ----------
    models : list
        a list of model instances
    jointparameters : list
        a list of joint parameters
    initvals : list
        a list of initial values

    """
    models: Incomplete
    initvals: Incomplete
    jointparams: Incomplete
    fitparams: Incomplete
    modeldims: Incomplete
    ndim: Incomplete
    def __init__(self, models, jointparameters, initvals) -> None: ...
    def model_to_fit_params(self): ...
    def objective_function(self, fps, *args):
        """
        Function to minimize.

        Parameters
        ----------
        fps : list
            the fitted parameters - result of an one iteration of the
            fitting algorithm
        args : dict
            tuple of measured and input coordinates
            args is always passed as a tuple from optimize.leastsq

        """
    def _verify_input(self) -> None: ...
    def __call__(self, *args):
        """
        Fit data to these models keeping some of the parameters common to the
        two models.
        """
