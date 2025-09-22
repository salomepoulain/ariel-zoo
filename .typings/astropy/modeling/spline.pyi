import abc
from .core import FittableModel
from _typeshed import Incomplete

__all__ = ['Spline1D', 'SplineInterpolateFitter', 'SplineSmoothingFitter', 'SplineExactKnotsFitter', 'SplineSplrepFitter']

class _Spline(FittableModel, metaclass=abc.ABCMeta):
    """Base class for spline models."""
    _knot_names: Incomplete
    _coeff_names: Incomplete
    optional_inputs: Incomplete
    _user_knots: bool
    def __init__(self, knots: Incomplete | None = None, coeffs: Incomplete | None = None, degree: Incomplete | None = None, bounds: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None) -> None: ...
    @property
    def param_names(self):
        """
        Coefficient names generated based on the spline's degree and
        number of knots.
        """
    @staticmethod
    def _optional_arg(arg): ...
    def _create_optional_inputs(self) -> None: ...
    def _intercept_optional_inputs(self, **kwargs): ...
    def evaluate(self, *args, **kwargs):
        """Extract the optional kwargs passed to call."""
    def __call__(self, *args, **kwargs):
        """
        Make model callable to model evaluation.
        """
    def _create_parameter(self, name: str, index: int, attr: str, fixed: bool = False):
        """
        Create a spline parameter linked to an attribute array.

        Parameters
        ----------
        name : str
            Name for the parameter
        index : int
            The index of the parameter in the array
        attr : str
            The name for the attribute array
        fixed : optional, bool
            If the parameter should be fixed or not
        """
    def _create_parameters(self, base_name: str, attr: str, fixed: bool = False):
        """
        Create a spline parameters linked to an attribute array for all
        elements in that array.

        Parameters
        ----------
        base_name : str
            Base name for the parameters
        attr : str
            The name for the attribute array
        fixed : optional, bool
            If the parameters should be fixed or not
        """
    @abc.abstractmethod
    def _init_parameters(self): ...
    @abc.abstractmethod
    def _init_data(self, knots, coeffs, bounds: Incomplete | None = None): ...
    def _init_spline(self, knots, coeffs, bounds: Incomplete | None = None) -> None: ...
    _c: Incomplete
    _t: Incomplete
    _degree: Incomplete
    def _init_tck(self, degree) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...

class Spline1D(_Spline):
    '''
    One dimensional Spline Model.

    Parameters
    ----------
    knots :  optional
        Define the knots for the spline. Can be 1) the number of interior
        knots for the spline, 2) the array of all knots for the spline, or
        3) If both bounds are defined, the interior knots for the spline
    coeffs : optional
        The array of knot coefficients for the spline
    degree : optional
        The degree of the spline. It must be 1 <= degree <= 5, default is 3.
    bounds : optional
        The upper and lower bounds of the spline.

    Notes
    -----
    Much of the functionality of this model is provided by
    `scipy.interpolate.BSpline` which can be directly accessed via the
    bspline property.

    Fitting for this model is provided by wrappers for:
    `scipy.interpolate.UnivariateSpline`,
    `scipy.interpolate.InterpolatedUnivariateSpline`,
    and `scipy.interpolate.LSQUnivariateSpline`.

    If one fails to define any knots/coefficients, no parameters will
    be added to this model until a fitter is called. This is because
    some of the fitters for splines vary the number of parameters and so
    we cannot define the parameter set until after fitting in these cases.

    Since parameters are not necessarily known at model initialization,
    setting model parameters directly via the model interface has been
    disabled.

    Direct constructors are provided for this model which incorporate the
    fitting to data directly into model construction.

    Knot parameters are declared as "fixed" parameters by default to
    enable the use of other `astropy.modeling` fitters to be used to
    fit this model.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Spline1D
    >>> from astropy.modeling import fitting
    >>> np.random.seed(42)
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)
    >>> xs = np.linspace(-3, 3, 1000)

    A 1D interpolating spline can be fit to data:

    >>> fitter = fitting.SplineInterpolateFitter()
    >>> spl = fitter(Spline1D(), x, y)

    Similarly, a smoothing spline can be fit to data:

    >>> fitter = fitting.SplineSmoothingFitter()
    >>> spl = fitter(Spline1D(), x, y, s=0.5)

    Similarly, a spline can be fit to data using an exact set of interior knots:

    >>> t = [-1, 0, 1]
    >>> fitter = fitting.SplineExactKnotsFitter()
    >>> spl = fitter(Spline1D(), x, y, t=t)
    '''
    n_inputs: int
    n_outputs: int
    _separable: bool
    optional_inputs: Incomplete
    def __init__(self, knots: Incomplete | None = None, coeffs: Incomplete | None = None, degree: int = 3, bounds: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None) -> None: ...
    @property
    def t(self):
        """
        The knots vector.
        """
    _t: Incomplete
    @t.setter
    def t(self, value) -> None: ...
    @property
    def t_interior(self):
        """
        The interior knots.
        """
    @property
    def c(self):
        """
        The coefficients vector.
        """
    _c: Incomplete
    @c.setter
    def c(self, value) -> None: ...
    @property
    def degree(self):
        """
        The degree of the spline polynomials.
        """
    @property
    def _initialized(self): ...
    @property
    def tck(self):
        """
        Scipy 'tck' tuple representation.
        """
    @tck.setter
    def tck(self, value) -> None: ...
    @property
    def bspline(self):
        """
        Scipy bspline object representation.
        """
    @bspline.setter
    def bspline(self, value) -> None: ...
    @property
    def knots(self):
        """
        Dictionary of knot parameters.
        """
    @property
    def user_knots(self):
        """If the knots have been supplied by the user."""
    _user_knots: Incomplete
    @user_knots.setter
    def user_knots(self, value) -> None: ...
    @property
    def coeffs(self):
        """
        Dictionary of coefficient parameters.
        """
    _knot_names: Incomplete
    _coeff_names: Incomplete
    def _init_parameters(self) -> None: ...
    bounding_box: Incomplete
    def _init_bounds(self, bounds: Incomplete | None = None): ...
    def _init_knots(self, knots, has_bounds, lower, upper) -> None: ...
    def _init_coeffs(self, coeffs: Incomplete | None = None) -> None: ...
    def _init_data(self, knots, coeffs, bounds: Incomplete | None = None) -> None: ...
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the spline.

        Parameters
        ----------
        x :
            (positional) The points where the model is evaluating the spline at
        nu : optional
            (kwarg) The derivative of the spline for evaluation, 0 <= nu <= degree + 1.
            Default: 0.
        """
    def derivative(self, nu: int = 1):
        """
        Create a spline that is the derivative of this one.

        Parameters
        ----------
        nu : int, optional
            Derivative order, default is 1.
        """
    def antiderivative(self, nu: int = 1):
        """
        Create a spline that is an antiderivative of this one.

        Parameters
        ----------
        nu : int, optional
            Antiderivative order, default is 1.

        Notes
        -----
        Assumes constant of integration is 0
        """

class _SplineFitter(abc.ABC, metaclass=abc.ABCMeta):
    """
    Base Spline Fitter.
    """
    fit_info: Incomplete
    def __init__(self) -> None: ...
    def _set_fit_info(self, spline) -> None: ...
    @abc.abstractmethod
    def _fit_method(self, model, x, y, **kwargs): ...
    def __call__(self, model, x, y, z: Incomplete | None = None, **kwargs): ...

class SplineInterpolateFitter(_SplineFitter):
    """
    Fit an interpolating spline.
    """
    def _fit_method(self, model, x, y, **kwargs): ...

class SplineSmoothingFitter(_SplineFitter):
    """
    Fit a smoothing spline.
    """
    def _fit_method(self, model, x, y, **kwargs): ...

class SplineExactKnotsFitter(_SplineFitter):
    """
    Fit a spline using least-squares regression.
    """
    def _fit_method(self, model, x, y, **kwargs): ...

class SplineSplrepFitter(_SplineFitter):
    """
    Fit a spline using the `scipy.interpolate.splrep` function interface.
    """
    fit_info: Incomplete
    def __init__(self) -> None: ...
    def _fit_method(self, model, x, y, **kwargs): ...
    def _set_fit_info(self, spline) -> None: ...
