import abc
from .core import Fittable1DModel, Fittable2DModel
from _typeshed import Incomplete

__all__ = ['AiryDisk2D', 'Moffat1D', 'Moffat2D', 'Box1D', 'Box2D', 'Const1D', 'Const2D', 'Ellipse2D', 'Disk2D', 'Gaussian1D', 'Gaussian2D', 'GeneralSersic2D', 'Linear1D', 'Lorentz1D', 'Lorentz2D', 'RickerWavelet1D', 'RickerWavelet2D', 'RedshiftScaleFactor', 'Multiply', 'Planar2D', 'Scale', 'Sersic1D', 'Sersic2D', 'Shift', 'Sine1D', 'Cosine1D', 'Tangent1D', 'ArcSine1D', 'ArcCosine1D', 'ArcTangent1D', 'Trapezoid1D', 'TrapezoidDisk2D', 'Ring2D', 'Voigt1D', 'KingProjectedAnalytic1D', 'Exponential1D', 'Logarithmic1D']

class Gaussian1D(Fittable1DModel):
    """
    One dimensional Gaussian model.

    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`.
        Amplitude (peak value) of the Gaussian - for a normalized profile
        (integrating to 1), set amplitude = 1 / (stddev * np.sqrt(2 * np.pi))
    mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian.
    stddev : float or `~astropy.units.Quantity`.
        Standard deviation of the Gaussian with FWHM = 2 * stddev * np.sqrt(2 * np.log(2)).

    Notes
    -----
    The ``x``, ``mean``, and ``stddev`` inputs must have compatible
    units or be unitless numbers.

    Model formula:

        .. math:: f(x) = A e^{- \\frac{\\left(x - x_{0}\\right)^{2}}{2 \\sigma^{2}}}

    Examples
    --------
    >>> from astropy.modeling import models
    >>> def tie_center(model):
    ...         mean = 50 * model.stddev
    ...         return mean
    >>> tied_parameters = {'mean': tie_center}

    Specify that 'mean' is a tied parameter in one of two ways:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                             tied=tied_parameters)

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.mean.tied
    False
    >>> g1.mean.tied = tie_center
    >>> g1.mean.tied
    <function tie_center at 0x...>

    Fixed parameters:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        fixed={'stddev': True})
    >>> g1.stddev.fixed
    True

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.stddev.fixed
    False
    >>> g1.stddev.fixed = True
    >>> g1.stddev.fixed
    True

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Gaussian1D

        plt.figure()
        s1 = Gaussian1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()

    See Also
    --------
    Gaussian2D, Box1D, Moffat1D, Lorentz1D
    """
    amplitude: Incomplete
    mean: Incomplete
    stddev: Incomplete
    def bounding_box(self, factor: float = 5.5):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of `stddev` used to define the limits.
            The default is 5.5, corresponding to a relative error < 1e-7.

        Examples
        --------
        >>> from astropy.modeling.models import Gaussian1D
        >>> model = Gaussian1D(mean=0, stddev=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-11.0, upper=11.0)
            }
            model=Gaussian1D(inputs=('x',))
            order='C'
        )

        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different factor,
        like:

        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.0, upper=4.0)
            }
            model=Gaussian1D(inputs=('x',))
            order='C'
        )
        """
    @property
    def fwhm(self):
        """Gaussian full width at half maximum."""
    @staticmethod
    def evaluate(x, amplitude, mean, stddev):
        """
        Gaussian1D model function.
        """
    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev):
        """
        Gaussian1D model function derivatives.
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Gaussian2D(Fittable2DModel):
    """
    Two dimensional Gaussian model.

    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`.
        Amplitude (peak value) of the Gaussian.
    x_mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian in x.
    y_mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian in y.
    x_stddev : float or `~astropy.units.Quantity` or None.
        Standard deviation of the Gaussian in x before rotating by theta. Must
        be None if a covariance matrix (``cov_matrix``) is provided. If no
        ``cov_matrix`` is given, ``None`` means the default value (1).
    y_stddev : float or `~astropy.units.Quantity` or None.
        Standard deviation of the Gaussian in y before rotating by theta. Must
        be None if a covariance matrix (``cov_matrix``) is provided. If no
        ``cov_matrix`` is given, ``None`` means the default value (1).
    theta : float or `~astropy.units.Quantity`, optional.
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise. Must be `None` if a covariance matrix
        (``cov_matrix``) is provided. If no ``cov_matrix`` is given,
        `None` means the default value (0).
    cov_matrix : ndarray, optional
        A 2x2 covariance matrix. If specified, overrides the ``x_stddev``,
        ``y_stddev``, and ``theta`` defaults.

    Notes
    -----
    The ``x, y``, ``[x,y]_mean``, and ``[x,y]_stddev`` inputs must have
    compatible units or be unitless numbers.

    Model formula:

        .. math::

            f(x, y) = A e^{-a\\left(x - x_{0}\\right)^{2}  -b\\left(x - x_{0}\\right)
            \\left(y - y_{0}\\right)  -c\\left(y - y_{0}\\right)^{2}}

    Using the following definitions:

        .. math::
            a = \\left(\\frac{\\cos^{2}{\\left (\\theta \\right )}}{2 \\sigma_{x}^{2}} +
            \\frac{\\sin^{2}{\\left (\\theta \\right )}}{2 \\sigma_{y}^{2}}\\right)

            b = \\left(\\frac{\\sin{\\left (2 \\theta \\right )}}{2 \\sigma_{x}^{2}} -
            \\frac{\\sin{\\left (2 \\theta \\right )}}{2 \\sigma_{y}^{2}}\\right)

            c = \\left(\\frac{\\sin^{2}{\\left (\\theta \\right )}}{2 \\sigma_{x}^{2}} +
            \\frac{\\cos^{2}{\\left (\\theta \\right )}}{2 \\sigma_{y}^{2}}\\right)

    If using a ``cov_matrix``, the model is of the form:
        .. math::
            f(x, y) = A e^{-0.5 \\left(
                    \\vec{x} - \\vec{x}_{0}\\right)^{T} \\Sigma^{-1} \\left(\\vec{x} - \\vec{x}_{0}
                \\right)}

    where :math:`\\vec{x} = [x, y]`, :math:`\\vec{x}_{0} = [x_{0}, y_{0}]`,
    and :math:`\\Sigma` is the covariance matrix:

        .. math::
            \\Sigma = \\left(\\begin{array}{ccc}
            \\sigma_x^2               & \\rho \\sigma_x \\sigma_y \\\\\n            \\rho \\sigma_x \\sigma_y   & \\sigma_y^2
            \\end{array}\\right)

    :math:`\\rho` is the correlation between ``x`` and ``y``, which should
    be between -1 and +1.  Positive correlation corresponds to a
    ``theta`` in the range 0 to 90 degrees.  Negative correlation
    corresponds to a ``theta`` in the range of 0 to -90 degrees.

    See [1]_ for more details about the 2D Gaussian function.

    See Also
    --------
    Gaussian1D, Box2D, Moffat2D

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function
    """
    amplitude: Incomplete
    x_mean: Incomplete
    y_mean: Incomplete
    x_stddev: Incomplete
    y_stddev: Incomplete
    theta: Incomplete
    def __init__(self, amplitude=..., x_mean=..., y_mean=..., x_stddev: Incomplete | None = None, y_stddev: Incomplete | None = None, theta: Incomplete | None = None, cov_matrix: Incomplete | None = None, **kwargs) -> None: ...
    @property
    def x_fwhm(self):
        """Gaussian full width at half maximum in X."""
    @property
    def y_fwhm(self):
        """Gaussian full width at half maximum in Y."""
    def bounding_box(self, factor: float = 5.5):
        """
        Tuple defining the default ``bounding_box`` limits in each dimension,
        ``((y_low, y_high), (x_low, x_high))``.

        The default offset from the mean is 5.5-sigma, corresponding
        to a relative error < 1e-7. The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of `x_stddev` and `y_stddev` used to define the limits.
            The default is 5.5.

        Examples
        --------
        >>> from astropy.modeling.models import Gaussian2D
        >>> model = Gaussian2D(x_mean=0, y_mean=0, x_stddev=1, y_stddev=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.5, upper=5.5)
                y: Interval(lower=-11.0, upper=11.0)
            }
            model=Gaussian2D(inputs=('x', 'y'))
            order='C'
        )

        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different factor
        like:

        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-2.0, upper=2.0)
                y: Interval(lower=-4.0, upper=4.0)
            }
            model=Gaussian2D(inputs=('x', 'y'))
            order='C'
        )
        """
    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
        """Two dimensional Gaussian function."""
    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
        """Two dimensional Gaussian function derivative with respect to parameters."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Shift(Fittable1DModel):
    """
    Shift a coordinate.

    Parameters
    ----------
    offset : float
        Offset to add to a coordinate.
    """
    offset: Incomplete
    linear: bool
    _has_inverse_bounding_box: bool
    @property
    def input_units(self): ...
    @property
    def inverse(self):
        """One dimensional inverse Shift model function."""
    @staticmethod
    def evaluate(x, offset):
        """One dimensional Shift model function."""
    @staticmethod
    def sum_of_implicit_terms(x):
        """Evaluate the implicit term (x) of one dimensional Shift model."""
    @staticmethod
    def fit_deriv(x, *params):
        """One dimensional Shift model derivative with respect to parameter."""
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Scale(Fittable1DModel):
    """
    Multiply a model by a dimensionless factor.

    Parameters
    ----------
    factor : float
        Factor by which to scale a coordinate.

    Notes
    -----
    If ``factor`` is a `~astropy.units.Quantity` then the units will be
    stripped before the scaling operation.

    """
    factor: Incomplete
    linear: bool
    fittable: bool
    _input_units_strict: bool
    _input_units_allow_dimensionless: bool
    _has_inverse_bounding_box: bool
    @property
    def input_units(self): ...
    @property
    def inverse(self):
        """One dimensional inverse Scale model function."""
    @staticmethod
    def evaluate(x, factor):
        """One dimensional Scale model function."""
    @staticmethod
    def fit_deriv(x, *params):
        """One dimensional Scale model derivative with respect to parameter."""
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Multiply(Fittable1DModel):
    """
    Multiply a model by a quantity or number.

    Parameters
    ----------
    factor : float
        Factor by which to multiply a coordinate.
    """
    factor: Incomplete
    linear: bool
    fittable: bool
    _has_inverse_bounding_box: bool
    @property
    def inverse(self):
        """One dimensional inverse multiply model function."""
    @staticmethod
    def evaluate(x, factor):
        """One dimensional multiply model function."""
    @staticmethod
    def fit_deriv(x, *params):
        """One dimensional multiply model derivative with respect to parameter."""
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class RedshiftScaleFactor(Fittable1DModel):
    """
    One dimensional redshift scale factor model.

    Parameters
    ----------
    z : float
        Redshift value.

    Notes
    -----
    Model formula:

        .. math:: f(x) = x (1 + z)
    """
    z: Incomplete
    _has_inverse_bounding_box: bool
    @staticmethod
    def evaluate(x, z):
        """One dimensional RedshiftScaleFactor model function."""
    @staticmethod
    def fit_deriv(x, z):
        """One dimensional RedshiftScaleFactor model derivative."""
    @property
    def inverse(self):
        """Inverse RedshiftScaleFactor model."""

class Sersic1D(Fittable1DModel):
    """
    One dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Surface brightness at ``r_eff``.
    r_eff : float
        Effective (half-light) radius.
    n : float
        Sersic index controlling the shape of the profile. Particular
        values of ``n`` are equivalent to the following profiles:

            * n=4 : `de Vaucouleurs <https://en.wikipedia.org/wiki/De_Vaucouleurs%27s_law>`_ :math:`r^{1/4}` profile
            * n=1 : Exponential profile
            * n=0.5 : Gaussian profile

    See Also
    --------
    Gaussian1D, Moffat1D, Lorentz1D

    Notes
    -----
    The ``r`` and ``r_eff`` inputs must have compatible units or be
    unitless numbers.

    Model formula:

    .. math::

        I(r) = I_{e} \\exp\\left\\{
               -b_{n} \\left[\\left(\\frac{r}{r_{e}}\\right)^{(1/n)}
               -1\\right]\\right\\}

    where :math:`I_{e}` is the ``amplitude`` and :math:`r_{e}` is ``reff``.

    The constant :math:`b_{n}` is defined such that :math:`r_{e}`
    contains half the total luminosity. It can be solved for numerically
    from the following equation:

    .. math::

        \\Gamma(2n) = 2\\gamma (2n, b_{n})

    where :math:`\\Gamma(a)` is the `gamma function
    <https://en.wikipedia.org/wiki/Gamma_function>`_ and
    :math:`\\gamma(a, x)` is the `lower incomplete gamma function
    <https://en.wikipedia.org/wiki/Incomplete_gamma_function>`_.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Sersic1D
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(111, xscale='log', yscale='log')
        s1 = Sersic1D(amplitude=1, r_eff=5)
        r = np.arange(0, 100, 0.01)

        for n in range(1, 10):
             s1.n = n
             plt.plot(r, s1(r))

        plt.axis([1e-1, 30, 1e-2, 1e3])
        plt.xlabel('log Radius')
        plt.ylabel('log Surface Brightness')
        plt.text(0.25, 1.5, 'n=1')
        plt.text(0.25, 300, 'n=10')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    """
    amplitude: Incomplete
    r_eff: Incomplete
    n: Incomplete
    _gammaincinv: Incomplete
    @classmethod
    def evaluate(cls, r, amplitude, r_eff, n):
        """One dimensional Sersic profile function."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class _Trigonometric1D(Fittable1DModel, metaclass=abc.ABCMeta):
    """
    Base class for one dimensional trigonometric and inverse trigonometric models.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude
    frequency : float
        Oscillation frequency
    phase : float
        Oscillation phase
    """
    amplitude: Incomplete
    frequency: Incomplete
    phase: Incomplete
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Sine1D(_Trigonometric1D):
    """
    One dimensional Sine model.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude
    frequency : float
        Oscillation frequency
    phase : float
        Oscillation phase

    See Also
    --------
    ArcSine1D, Cosine1D, Tangent1D, Const1D, Linear1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = A \\sin(2 \\pi f x + 2 \\pi p)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Sine1D

        plt.figure()
        s1 = Sine1D(amplitude=1, frequency=.25)
        r=np.arange(0, 10, .01)

        for amplitude in range(1,4):
             s1.amplitude = amplitude
             plt.plot(r, s1(r), color=str(0.25 * amplitude), lw=2)

        plt.axis([0, 10, -5, 5])
        plt.show()
    """
    @staticmethod
    def evaluate(x, amplitude, frequency, phase):
        """One dimensional Sine model function."""
    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional Sine model derivative."""
    @property
    def inverse(self):
        """One dimensional inverse of Sine."""

class Cosine1D(_Trigonometric1D):
    """
    One dimensional Cosine model.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude
    frequency : float
        Oscillation frequency
    phase : float
        Oscillation phase

    See Also
    --------
    ArcCosine1D, Sine1D, Tangent1D, Const1D, Linear1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = A \\cos(2 \\pi f x + 2 \\pi p)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Cosine1D

        plt.figure()
        s1 = Cosine1D(amplitude=1, frequency=.25)
        r=np.arange(0, 10, .01)

        for amplitude in range(1,4):
             s1.amplitude = amplitude
             plt.plot(r, s1(r), color=str(0.25 * amplitude), lw=2)

        plt.axis([0, 10, -5, 5])
        plt.show()
    """
    @staticmethod
    def evaluate(x, amplitude, frequency, phase):
        """One dimensional Cosine model function."""
    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional Cosine model derivative."""
    @property
    def inverse(self):
        """One dimensional inverse of Cosine."""

class Tangent1D(_Trigonometric1D):
    """
    One dimensional Tangent model.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude
    frequency : float
        Oscillation frequency
    phase : float
        Oscillation phase

    See Also
    --------
    Sine1D, Cosine1D, Const1D, Linear1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = A \\tan(2 \\pi f x + 2 \\pi p)

    Note that the tangent function is undefined for inputs of the form
    pi/2 + n*pi for all integers n. Thus thus the default bounding box
    has been restricted to:

        .. math:: [(-1/4 - p)/f, (1/4 - p)/f]

    which is the smallest interval for the tangent function to be continuous
    on.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Tangent1D

        plt.figure()
        s1 = Tangent1D(amplitude=1, frequency=.25)
        r=np.arange(0, 10, .01)

        for amplitude in range(1,4):
             s1.amplitude = amplitude
             plt.plot(r, s1(r), color=str(0.25 * amplitude), lw=2)

        plt.axis([0, 10, -5, 5])
        plt.show()
    """
    @staticmethod
    def evaluate(x, amplitude, frequency, phase):
        """One dimensional Tangent model function."""
    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional Tangent model derivative."""
    @property
    def inverse(self):
        """One dimensional inverse of Tangent."""
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.
        """

class _InverseTrigonometric1D(_Trigonometric1D, metaclass=abc.ABCMeta):
    """
    Base class for one dimensional inverse trigonometric models.
    """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class ArcSine1D(_InverseTrigonometric1D):
    """
    One dimensional ArcSine model returning values between -pi/2 and pi/2
    only.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude for corresponding Sine
    frequency : float
        Oscillation frequency for corresponding Sine
    phase : float
        Oscillation phase for corresponding Sine

    See Also
    --------
    Sine1D, ArcCosine1D, ArcTangent1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = ((arcsin(x / A) / 2pi) - p) / f

    The arcsin function being used for this model will only accept inputs
    in [-A, A]; otherwise, a runtime warning will be thrown and the result
    will be NaN. To avoid this, the bounding_box has been properly set to
    accommodate this; therefore, it is recommended that this model always
    be evaluated with the ``with_bounding_box=True`` option.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import ArcSine1D

        plt.figure()
        s1 = ArcSine1D(amplitude=1, frequency=.25)
        r=np.arange(-1, 1, .01)

        for amplitude in range(1,4):
             s1.amplitude = amplitude
             plt.plot(r, s1(r), color=str(0.25 * amplitude), lw=2)

        plt.axis([-1, 1, -np.pi/2, np.pi/2])
        plt.show()
    """
    @staticmethod
    def evaluate(x, amplitude, frequency, phase):
        """One dimensional ArcSine model function."""
    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional ArcSine model derivative."""
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.
        """
    @property
    def inverse(self):
        """One dimensional inverse of ArcSine."""

class ArcCosine1D(_InverseTrigonometric1D):
    """
    One dimensional ArcCosine returning values between 0 and pi only.


    Parameters
    ----------
    amplitude : float
        Oscillation amplitude for corresponding Cosine
    frequency : float
        Oscillation frequency for corresponding Cosine
    phase : float
        Oscillation phase for corresponding Cosine

    See Also
    --------
    Cosine1D, ArcSine1D, ArcTangent1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = ((arccos(x / A) / 2pi) - p) / f

    The arccos function being used for this model will only accept inputs
    in [-A, A]; otherwise, a runtime warning will be thrown and the result
    will be NaN. To avoid this, the bounding_box has been properly set to
    accommodate this; therefore, it is recommended that this model always
    be evaluated with the ``with_bounding_box=True`` option.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import ArcCosine1D

        plt.figure()
        s1 = ArcCosine1D(amplitude=1, frequency=.25)
        r=np.arange(-1, 1, .01)

        for amplitude in range(1,4):
             s1.amplitude = amplitude
             plt.plot(r, s1(r), color=str(0.25 * amplitude), lw=2)

        plt.axis([-1, 1, 0, np.pi])
        plt.show()
    """
    @staticmethod
    def evaluate(x, amplitude, frequency, phase):
        """One dimensional ArcCosine model function."""
    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional ArcCosine model derivative."""
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.
        """
    @property
    def inverse(self):
        """One dimensional inverse of ArcCosine."""

class ArcTangent1D(_InverseTrigonometric1D):
    """
    One dimensional ArcTangent model returning values between -pi/2 and
    pi/2 only.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude for corresponding Tangent
    frequency : float
        Oscillation frequency for corresponding Tangent
    phase : float
        Oscillation phase for corresponding Tangent

    See Also
    --------
    Tangent1D, ArcSine1D, ArcCosine1D


    Notes
    -----
    Model formula:

        .. math:: f(x) = ((arctan(x / A) / 2pi) - p) / f

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import ArcTangent1D

        plt.figure()
        s1 = ArcTangent1D(amplitude=1, frequency=.25)
        r=np.arange(-10, 10, .01)

        for amplitude in range(1,4):
             s1.amplitude = amplitude
             plt.plot(r, s1(r), color=str(0.25 * amplitude), lw=2)

        plt.axis([-10, 10, -np.pi/2, np.pi/2])
        plt.show()
    """
    @staticmethod
    def evaluate(x, amplitude, frequency, phase):
        """One dimensional ArcTangent model function."""
    @staticmethod
    def fit_deriv(x, amplitude, frequency, phase):
        """One dimensional ArcTangent model derivative."""
    @property
    def inverse(self):
        """One dimensional inverse of ArcTangent."""

class Linear1D(Fittable1DModel):
    """
    One dimensional Line model.

    Parameters
    ----------
    slope : float
        Slope of the straight line

    intercept : float
        Intercept of the straight line

    See Also
    --------
    Const1D

    Notes
    -----
    Model formula:

        .. math:: f(x) = a x + b
    """
    slope: Incomplete
    intercept: Incomplete
    linear: bool
    @staticmethod
    def evaluate(x, slope, intercept):
        """One dimensional Line model function."""
    @staticmethod
    def fit_deriv(x, *params):
        """One dimensional Line model derivative with respect to parameters."""
    @property
    def inverse(self): ...
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Planar2D(Fittable2DModel):
    """
    Two dimensional Plane model.

    Parameters
    ----------
    slope_x : float
        Slope of the plane in X

    slope_y : float
        Slope of the plane in Y

    intercept : float
        Z-intercept of the plane

    Notes
    -----
    Model formula:

        .. math:: f(x, y) = a x + b y + c
    """
    slope_x: Incomplete
    slope_y: Incomplete
    intercept: Incomplete
    linear: bool
    @staticmethod
    def evaluate(x, y, slope_x, slope_y, intercept):
        """Two dimensional Plane model function."""
    @staticmethod
    def fit_deriv(x, y, *params):
        """Two dimensional Plane model derivative with respect to parameters."""
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Lorentz1D(Fittable1DModel):
    """
    One dimensional Lorentzian model.

    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`.
        Peak value. For a normalized profile (integrating to 1),
        set amplitude = 2 / (np.pi * fwhm).
    x_0 : float or `~astropy.units.Quantity`.
        Position of the peak.
    fwhm : float or `~astropy.units.Quantity`.
        Full width at half maximum (FWHM).

    See Also
    --------
    Lorentz2D, Gaussian1D, Box1D, RickerWavelet1D

    Notes
    -----
    The ``x``, ``x_0``, and ``fwhm`` inputs must have compatible units
    or be unitless numbers.

    Model formula:

    .. math::

        f(x) = \\frac{A \\gamma^{2}}{\\gamma^{2} + \\left(x - x_{0}\\right)^{2}}

    where :math:`\\gamma` is the half width at half maximum (HWHM),
    which is half the FWHM.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Lorentz1D

        plt.figure()
        s1 = Lorentz1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            plt.plot(r, s1(r), lw=2, label=f'Amplitude={factor}')

        plt.axis([-5, 5, -1, 4])
        plt.legend()
        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    fwhm: Incomplete
    @staticmethod
    def evaluate(x, amplitude, x_0, fwhm):
        """One dimensional Lorentzian model function."""
    @staticmethod
    def fit_deriv(x, amplitude, x_0, fwhm):
        """One dimensional Lorentzian model derivative with respect to parameters."""
    def bounding_box(self, factor: int = 25):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of FWHM used to define the limits.
            Default is chosen to include most (99%) of the
            area under the curve, while still showing the
            central feature of interest.

        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Lorentz2D(Fittable2DModel):
    """
    Two-dimensional Lorentzian model.

    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`.
        Peak value.
    x_0 : float or `~astropy.units.Quantity`.
        Position of the peak in x.
    y_0 : float or `~astropy.units.Quantity`.
        Position of the peak in y.
    fwhm : float or `~astropy.units.Quantity`.
        Full width at half maximum (FWHM).

    See Also
    --------
    Lorentz1D, Gaussian2D, Moffat2D

    Notes
    -----
    The ``x``, ``y``, ``x_0``, ``y_0``, and ``fwhm`` inputs must have
    compatible units or as unitless numbers.

    Model formula:

    .. math::

        f(x, y) = \\frac{A \\gamma^{2}}{\\gamma^{2}
                  + \\left(x - x_{0}\\right)^2 + \\left(y - y_{0}\\right)^{2}}

    where :math:`\\gamma` is the half width at half maximum (HWHM), which
    is half the FWHM.

    The area under the `Lorentz2D` profile is infinite, therefore this
    model profile cannot be normalized to sum to 1.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Lorentz2D

        plt.figure()
        model = Lorentz2D(x_0=12, y_0=12, fwhm=3)
        yy, xx = np.mgrid[:25, :25]
        data = model(xx, yy)

        plt.imshow(data)
        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    fwhm: Incomplete
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, fwhm):
        """Two dimensional Lorentzian model function."""
    @staticmethod
    def fit_deriv(x, y, amplitude, x_0, y_0, fwhm):
        """Two dimensional Lorentzian model derivative with respect to parameters."""
    def bounding_box(self, factor: int = 25):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high), (y_low, y_high)``.

        Parameters
        ----------
        factor : float
            The multiple of FWHM used to define the limits.
            Default is chosen to include most (99%) of the
            area under the curve, while still showing the
            central feature of interest.
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Voigt1D(Fittable1DModel):
    """
    One dimensional model for the Voigt profile.

    Parameters
    ----------
    x_0 : float or `~astropy.units.Quantity`
        Position of the peak
    amplitude_L : float or `~astropy.units.Quantity`.
        The Lorentzian amplitude (peak of the associated Lorentz function)
        - for a normalized profile (integrating to 1), set
        amplitude_L = 2 / (np.pi * fwhm_L)
    fwhm_L : float or `~astropy.units.Quantity`
        The Lorentzian full width at half maximum
    fwhm_G : float or `~astropy.units.Quantity`.
        The Gaussian full width at half maximum
    method : str, optional
        Algorithm for computing the complex error function; one of
        'Humlicek2' (default, fast and generally more accurate than ``rtol=3.e-5``) or
        'Scipy', alternatively 'wofz' (requires ``scipy``, almost as fast and
        reference in accuracy).

    See Also
    --------
    Gaussian1D, Lorentz1D

    Notes
    -----
    The ``x``, ``x_0``, and ``fwhm_*`` inputs must have compatible units
    or be unitless numbers.

    Voigt function is calculated as real part of the complex error function computed from either
    Humlicek's rational approximations (JQSRT 21:309, 1979; 27:437, 1982) following
    Schreier 2018 (MNRAS 479, 3068; and ``hum2zpf16m`` from his cpfX.py module); or
    `~scipy.special.wofz` (implementing 'Faddeeva.cc').

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Voigt1D
        import matplotlib.pyplot as plt

        plt.figure()
        x = np.arange(0, 10, 0.01)
        v1 = Voigt1D(x_0=5, amplitude_L=10, fwhm_L=0.5, fwhm_G=0.9)
        plt.plot(x, v1(x))
        plt.show()
    """
    x_0: Incomplete
    amplitude_L: Incomplete
    fwhm_L: Incomplete
    fwhm_G: Incomplete
    sqrt_pi: Incomplete
    sqrt_ln2: Incomplete
    sqrt_ln2pi: Incomplete
    _last_z: Incomplete
    _last_w: Incomplete
    _faddeeva: Incomplete
    method: Incomplete
    def __init__(self, x_0=..., amplitude_L=..., fwhm_L=..., fwhm_G=..., method: Incomplete | None = None, **kwargs) -> None: ...
    def _wrap_wofz(self, z):
        """Call complex error (Faddeeva) function w(z) implemented by algorithm `method`;
        cache results for consecutive calls from `evaluate`, `fit_deriv`.
        """
    def evaluate(self, x, x_0, amplitude_L, fwhm_L, fwhm_G):
        """One dimensional Voigt function scaled to Lorentz peak amplitude."""
    def fit_deriv(self, x, x_0, amplitude_L, fwhm_L, fwhm_G):
        """
        Derivative of the one dimensional Voigt function with respect to parameters.
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...
    @staticmethod
    def _hum2zpf16c(z, s: float = 10.0):
        """Complex error function w(z = x + iy) combining Humlicek's rational approximations.

        |x| + y > 10:  Humlicek (JQSRT, 1982) rational approximation for region II;
        else:          Humlicek (JQSRT, 1979) rational approximation with n=16 and delta=y0=1.35

        Version using a mask and np.place;
        single complex argument version of Franz Schreier's cpfX.hum2zpf16m.
        Originally licensed under a 3-clause BSD style license - see
        https://atmos.eoc.dlr.de/tools/lbl4IR/cpfX.py
        """

class Const1D(Fittable1DModel):
    """
    One dimensional Constant model.

    Parameters
    ----------
    amplitude : float
        Value of the constant function

    See Also
    --------
    Const2D

    Notes
    -----
    Model formula:

        .. math:: f(x) = A

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Const1D

        plt.figure()
        s1 = Const1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()
    """
    amplitude: Incomplete
    linear: bool
    @staticmethod
    def evaluate(x, amplitude):
        """One dimensional Constant model function."""
    @staticmethod
    def fit_deriv(x, amplitude):
        """One dimensional Constant model derivative with respect to parameters."""
    @property
    def input_units(self) -> None: ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Const2D(Fittable2DModel):
    """
    Two dimensional Constant model.

    Parameters
    ----------
    amplitude : float
        Value of the constant function

    See Also
    --------
    Const1D

    Notes
    -----
    Model formula:

        .. math:: f(x, y) = A
    """
    amplitude: Incomplete
    linear: bool
    @staticmethod
    def evaluate(x, y, amplitude):
        """Two dimensional Constant model function."""
    @property
    def input_units(self) -> None: ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Ellipse2D(Fittable2DModel):
    """
    A 2D Ellipse model.

    Parameters
    ----------
    amplitude : float
        Value of the ellipse.

    x_0 : float
        x position of the center of the disk.

    y_0 : float
        y position of the center of the disk.

    a : float
        The length of the semimajor axis.

    b : float
        The length of the semiminor axis.

    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise from the positive x axis.

    See Also
    --------
    Disk2D, Box2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = \\left \\{
                    \\begin{array}{ll}
                      \\mathrm{amplitude} & : \\left[\\frac{(x - x_0) \\cos
                        \\theta + (y - y_0) \\sin \\theta}{a}\\right]^2 +
                        \\left[\\frac{-(x - x_0) \\sin \\theta + (y - y_0)
                        \\cos \\theta}{b}\\right]^2  \\leq 1 \\\\\n                      0 & : \\mathrm{otherwise}
                    \\end{array}
                  \\right.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Ellipse2D
        from astropy.coordinates import Angle
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        x0, y0 = 25, 25
        a, b = 20, 10
        theta = Angle(30, 'deg')
        e = Ellipse2D(amplitude=100., x_0=x0, y_0=y0, a=a, b=b,
                      theta=theta.radian)
        y, x = np.mgrid[0:50, 0:50]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(e(x, y), origin='lower', interpolation='none', cmap='Greys_r')
        e2 = mpatches.Ellipse((x0, y0), 2*a, 2*b, angle=theta.degree, edgecolor='red',
                              facecolor='none')
        ax.add_patch(e2)
        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    a: Incomplete
    b: Incomplete
    theta: Incomplete
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, a, b, theta):
        """Two dimensional Ellipse model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        ``((y_low, y_high), (x_low, x_high))``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Disk2D(Fittable2DModel):
    """
    Two dimensional radial symmetric Disk model.

    Parameters
    ----------
    amplitude : float
        Value of the disk function
    x_0 : float
        x position center of the disk
    y_0 : float
        y position center of the disk
    R_0 : float
        Radius of the disk

    See Also
    --------
    Box2D, TrapezoidDisk2D

    Notes
    -----
    Model formula:

        .. math::

            f(r) = \\left \\{
                     \\begin{array}{ll}
                       A & : r \\leq R_0 \\\\\n                       0 & : r > R_0
                     \\end{array}
                   \\right.
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    R_0: Incomplete
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, R_0):
        """Two dimensional Disk model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        ``((y_low, y_high), (x_low, x_high))``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Ring2D(Fittable2DModel):
    """
    Two dimensional radial symmetric Ring model.

    Parameters
    ----------
    amplitude : float
        Value of the disk function
    x_0 : float
        x position center of the disk
    y_0 : float
        y position center of the disk
    r_in : float
        Inner radius of the ring
    width : float
        Width of the ring.
    r_out : float
        Outer Radius of the ring. Can be specified instead of width.

    See Also
    --------
    Disk2D, TrapezoidDisk2D

    Notes
    -----
    Model formula:

        .. math::

            f(r) = \\left \\{
                     \\begin{array}{ll}
                       A & : r_{in} \\leq r \\leq r_{out} \\\\\n                       0 & : \\text{else}
                     \\end{array}
                   \\right.

    Where :math:`r_{out} = r_{in} + r_{width}`.
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    r_in: Incomplete
    width: Incomplete
    def __init__(self, amplitude=..., x_0=..., y_0=..., r_in: Incomplete | None = None, width: Incomplete | None = None, r_out: Incomplete | None = None, **kwargs) -> None: ...
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, r_in, width):
        """Two dimensional Ring model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box``.

        ``((y_low, y_high), (x_low, x_high))``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Box1D(Fittable1DModel):
    """
    One dimensional Box model.

    Parameters
    ----------
    amplitude : float
        Amplitude A
    x_0 : float
        Position of the center of the box function
    width : float
        Width of the box

    See Also
    --------
    Box2D, TrapezoidDisk2D

    Notes
    -----
    Model formula:

      .. math::

            f(x) = \\left \\{
                     \\begin{array}{ll}
                       A & : x_0 - w/2 \\leq x \\leq x_0 + w/2 \\\\\n                       0 & : \\text{else}
                     \\end{array}
                   \\right.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Box1D

        plt.figure()
        s1 = Box1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            s1.width = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    width: Incomplete
    @staticmethod
    def evaluate(x, amplitude, x_0, width):
        """One dimensional Box model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        ``(x_low, x_high))``
        """
    @property
    def input_units(self): ...
    @property
    def return_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Box2D(Fittable2DModel):
    """
    Two dimensional Box model.

    Parameters
    ----------
    amplitude : float
        Amplitude
    x_0 : float
        x position of the center of the box function
    x_width : float
        Width in x direction of the box
    y_0 : float
        y position of the center of the box function
    y_width : float
        Width in y direction of the box

    See Also
    --------
    Box1D, Gaussian2D, Moffat2D

    Notes
    -----
    Model formula:

      .. math::

            f(x, y) = \\left \\{
                     \\begin{array}{ll}
            A : & x_0 - w_x/2 \\leq x \\leq x_0 + w_x/2 \\text{ and} \\\\\n                & y_0 - w_y/2 \\leq y \\leq y_0 + w_y/2 \\\\\n            0 : & \\text{else}
                     \\end{array}
                   \\right.

    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    x_width: Incomplete
    y_width: Incomplete
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, x_width, y_width):
        """Two dimensional Box model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box``.

        ``((y_low, y_high), (x_low, x_high))``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Trapezoid1D(Fittable1DModel):
    """
    One dimensional Trapezoid model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the trapezoid
    x_0 : float
        Center position of the trapezoid
    width : float
        Width of the constant part of the trapezoid.
    slope : float
        Slope of the tails of the trapezoid

    See Also
    --------
    Box1D, Gaussian1D, Moffat1D

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Trapezoid1D

        plt.figure()
        s1 = Trapezoid1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            s1.width = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    width: Incomplete
    slope: Incomplete
    @staticmethod
    def evaluate(x, amplitude, x_0, width, slope):
        """One dimensional Trapezoid model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        ``(x_low, x_high))``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class TrapezoidDisk2D(Fittable2DModel):
    """
    Two dimensional circular Trapezoid model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the trapezoid
    x_0 : float
        x position of the center of the trapezoid
    y_0 : float
        y position of the center of the trapezoid
    R_0 : float
        Radius of the constant part of the trapezoid.
    slope : float
        Slope of the tails of the trapezoid in x direction.

    See Also
    --------
    Disk2D, Box2D
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    R_0: Incomplete
    slope: Incomplete
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, R_0, slope):
        """Two dimensional Trapezoid Disk model function."""
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box``.

        ``((y_low, y_high), (x_low, x_high))``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class RickerWavelet1D(Fittable1DModel):
    '''
    One dimensional Ricker Wavelet model (sometimes known as a "Mexican Hat"
    model).

    .. note::

        See https://github.com/astropy/astropy/pull/9445 for discussions
        related to renaming of this model.

    Parameters
    ----------
    amplitude : float
        Amplitude
    x_0 : float
        Position of the peak
    sigma : float
        Width of the Ricker wavelet

    See Also
    --------
    RickerWavelet2D, Box1D, Gaussian1D, Trapezoid1D

    Notes
    -----
    Model formula:

    .. math::

        f(x) = {A \\left(1 - \\frac{\\left(x - x_{0}\\right)^{2}}{\\sigma^{2}}\\right)
        e^{- \\frac{\\left(x - x_{0}\\right)^{2}}{2 \\sigma^{2}}}}

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import RickerWavelet1D

        plt.figure()
        s1 = RickerWavelet1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            s1.width = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -2, 4])
        plt.show()
    '''
    amplitude: Incomplete
    x_0: Incomplete
    sigma: Incomplete
    @staticmethod
    def evaluate(x, amplitude, x_0, sigma):
        """One dimensional Ricker Wavelet model function."""
    def bounding_box(self, factor: float = 10.0):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of sigma used to define the limits.

        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class RickerWavelet2D(Fittable2DModel):
    '''
    Two dimensional Ricker Wavelet model (sometimes known as a "Mexican Hat"
    model).

    .. note::

        See https://github.com/astropy/astropy/pull/9445 for discussions
        related to renaming of this model.

    Parameters
    ----------
    amplitude : float
        Amplitude
    x_0 : float
        x position of the peak
    y_0 : float
        y position of the peak
    sigma : float
        Width of the Ricker wavelet

    See Also
    --------
    RickerWavelet1D, Gaussian2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = A \\left(1 - \\frac{\\left(x - x_{0}\\right)^{2}
        + \\left(y - y_{0}\\right)^{2}}{\\sigma^{2}}\\right)
        e^{\\frac{- \\left(x - x_{0}\\right)^{2}
        - \\left(y - y_{0}\\right)^{2}}{2 \\sigma^{2}}}
    '''
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    sigma: Incomplete
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, sigma):
        """Two dimensional Ricker Wavelet model function."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class AiryDisk2D(Fittable2DModel):
    """
    Two dimensional Airy disk model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the Airy function.
    x_0 : float
        x position of the maximum of the Airy function.
    y_0 : float
        y position of the maximum of the Airy function.
    radius : float
        The radius of the Airy disk (radius of the first zero).

    See Also
    --------
    Box2D, TrapezoidDisk2D, Gaussian2D

    Notes
    -----
    Model formula:

        .. math:: f(r) = A \\left[
                \\frac{2 J_1(\\frac{\\pi r}{R/R_z})}{\\frac{\\pi r}{R/R_z}}
            \\right]^2

    Where :math:`J_1` is the first order Bessel function of the first
    kind, :math:`r` is radial distance from the maximum of the Airy
    function (:math:`r = \\sqrt{(x - x_0)^2 + (y - y_0)^2}`), :math:`R`
    is the input ``radius`` parameter, and :math:`R_z =
    1.2196698912665045`).

    For an optical system, the radius of the first zero represents the
    limiting angular resolution and is approximately 1.22 * lambda / D,
    where lambda is the wavelength of the light and D is the diameter of
    the aperture.

    See [1]_ for more details about the Airy disk.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Airy_disk
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    radius: Incomplete
    _rz: Incomplete
    _j1: Incomplete
    @classmethod
    def evaluate(cls, x, y, amplitude, x_0, y_0, radius):
        """Two dimensional Airy model function."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Moffat1D(Fittable1DModel):
    """
    One dimensional Moffat model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the model.
    x_0 : float
        x position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.

    See Also
    --------
    Gaussian1D, Box1D

    Notes
    -----
    Model formula:

    .. math::

        f(x) = A \\left(1 + \\frac{\\left(x - x_{0}\\right)^{2}}{\\gamma^{2}}\\right)^{- \\alpha}

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Moffat1D

        plt.figure()
        s1 = Moffat1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            s1.width = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    gamma: Incomplete
    alpha: Incomplete
    @property
    def fwhm(self):
        """
        Moffat full width at half maximum.
        Derivation of the formula is available in
        `this notebook by Yoonsoo Bach
        <https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat>`_.
        """
    @staticmethod
    def evaluate(x, amplitude, x_0, gamma, alpha):
        """One dimensional Moffat model function."""
    @staticmethod
    def fit_deriv(x, amplitude, x_0, gamma, alpha):
        """One dimensional Moffat model derivative with respect to parameters."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Moffat2D(Fittable2DModel):
    """
    Two dimensional Moffat model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the model.
    x_0 : float
        x position of the maximum of the Moffat model.
    y_0 : float
        y position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.

    See Also
    --------
    Gaussian2D, Box2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = A \\left(1 + \\frac{\\left(x - x_{0}\\right)^{2} +
        \\left(y - y_{0}\\right)^{2}}{\\gamma^{2}}\\right)^{- \\alpha}

    Note that if :math:`\\alpha` is 1, the `Moffat2D` profile is a
    `Lorentz2D` profile. In that case, the integral of the profile is
    infinite.
    """
    amplitude: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    gamma: Incomplete
    alpha: Incomplete
    @property
    def fwhm(self):
        """
        Moffat full width at half maximum.
        Derivation of the formula is available in
        `this notebook by Yoonsoo Bach
        <https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat>`_.
        """
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, gamma, alpha):
        """Two dimensional Moffat model function."""
    @staticmethod
    def fit_deriv(x, y, amplitude, x_0, y_0, gamma, alpha):
        """Two dimensional Moffat model derivative with respect to parameters."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Sersic2D(Fittable2DModel):
    """
    Two dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Surface brightness at ``r_eff``.
    r_eff : float
        Effective (half-light) radius.
    n : float
        Sersic index controlling the shape of the profile. Particular
        values of ``n`` are equivalent to the following profiles:

            * n=4 : `de Vaucouleurs <https://en.wikipedia.org/wiki/De_Vaucouleurs%27s_law>`_ :math:`r^{1/4}` profile
            * n=1 : Exponential profile
            * n=0.5 : Gaussian profile
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity of the isophote, defined as 1.0 minus the ratio of
        the lengths of the semimajor and semiminor axes:

        .. math:: ellip = 1 - \\frac{b}{a}
    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise from the positive x axis.

    See Also
    --------
    GeneralSersic2D, Gaussian2D, Moffat2D

    Notes
    -----
    The ``x``, ``y``, ``x_0``, ``y_0``, and ``r_eff`` inputs must have
    compatible units or be unitless numbers.

    Model formula:

    .. math::

        I(x, y) = I_{e} \\exp\\left\\{
                  -b_{n} \\left[\\left(\\frac{r(x, y)}{r_{e}}\\right)^{(1/n)}
                  -1\\right]\\right\\}

    where :math:`I_{e}` is the ``amplitude``, :math:`r_{e}` is ``reff``,
    and :math:`r(x, y)` is a rotated ellipse defined as:

    .. math::

        r(x, y)^2 = A^2 + \\left(\\frac{B}{1 - ellip}\\right)^2

    .. math::

        A = (x - x_0) \\cos(\\theta) + (y - y_0) \\sin(\\theta)

    .. math::

        B = -(x - x_0) \\sin(\\theta) + (y - y_0) \\cos(\\theta)

    The constant :math:`b_{n}` is defined such that :math:`r_{e}`
    contains half the total luminosity. It can be solved for numerically
    from the following equation:

    .. math::

        \\Gamma(2n) = 2\\gamma (2n, b_{n})

    where :math:`\\Gamma(a)` is the `gamma function
    <https://en.wikipedia.org/wiki/Gamma_function>`_ and
    :math:`\\gamma(a, x)` is the `lower incomplete gamma function
    <https://en.wikipedia.org/wiki/Incomplete_gamma_function>`_.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Sersic2D
        import matplotlib.pyplot as plt

        x, y = np.meshgrid(np.arange(100), np.arange(100))

        mod = Sersic2D(amplitude=1, r_eff=25, n=4, x_0=50, y_0=50,
                       ellip=0.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)

        fig, ax = plt.subplots()
        im = ax.imshow(log_img, origin='lower', interpolation='nearest',
                       vmin=-1, vmax=2)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Log Brightness', rotation=270, labelpad=25)
        cbar.set_ticks([-1, 0, 1, 2])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    """
    amplitude: Incomplete
    r_eff: Incomplete
    n: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    ellip: Incomplete
    theta: Incomplete
    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta, c: int = 0):
        """Two dimensional Sersic profile function."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class GeneralSersic2D(Sersic2D):
    '''
    Generalized two dimensional Sersic surface brightness profile that
    allows for "boxy" or "disky" (kite-like) isophote shapes.

    Parameters
    ----------
    amplitude : float
        Surface brightness at ``r_eff``.
    r_eff : float
        Effective (half-light) radius.
    n : float
        Sersic index controlling the shape of the profile. Particular
        values of ``n`` are equivalent to the following profiles:

            * n=4 : `de Vaucouleurs <https://en.wikipedia.org/wiki/De_Vaucouleurs%27s_law>`_ :math:`r^{1/4}` profile
            * n=1 : Exponential profile
            * n=0.5 : Gaussian profile
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity of the isophote, defined as 1.0 minus the ratio of
        the lengths of the semimajor and semiminor axes:

        .. math:: ellip = 1 - \\frac{b}{a}
    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise from the positive x axis.
    c : float, optional
        Parameter controlling the shape of the generalized ellipses.
        Negative values correspond to disky (kite-like) isophotes and
        positive values correspond to boxy isophotes. Setting ``c=0``
        provides perfectly elliptical isophotes (the same model as
        `Sersic2D`).

    See Also
    --------
    Sersic2D, Gaussian2D, Moffat2D

    Notes
    -----
    Model formula:

    .. math::

        I(x, y) = I_{e} \\exp\\left\\{
                  -b_{n} \\left[\\left(\\frac{r(x, y)}{r_{e}}\\right)^{(1/n)}
                  -1\\right]\\right\\}

    where :math:`I_{e}` is the ``amplitude``, :math:`r_{e}`
    is ``reff``, and :math:`r(x, y)` is a rotated
    "generalized" ellipse (see `Athanassoula et al. 1990
    <https://ui.adsabs.harvard.edu/abs/1990MNRAS.245..130A/abstract>`_)
    defined as:

    .. math::

        r(x, y)^2 = |A|^{c + 2}
                    + \\left(\\frac{|B|}{1 - ellip}\\right)^{c + 2}

    .. math::

        A = (x - x_0) \\cos(\\theta) + (y - y_0) \\sin(\\theta)

    .. math::

        B = -(x - x_0) \\sin(\\theta) + (y - y_0) \\cos(\\theta)

    The constant :math:`b_{n}` is defined such that :math:`r_{e}`
    contains half the total luminosity. It can be solved for numerically
    from the following equation:

    .. math::

        \\Gamma(2n) = 2\\gamma (2n, b_{n})

    where :math:`\\Gamma(a)` is the `gamma function
    <https://en.wikipedia.org/wiki/Gamma_function>`_ and
    :math:`\\gamma(a, x)` is the `lower incomplete gamma function
    <https://en.wikipedia.org/wiki/Incomplete_gamma_function>`_.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import GeneralSersic2D
        import matplotlib.pyplot as plt

        x, y = np.meshgrid(np.arange(100), np.arange(100))

        mod = GeneralSersic2D(amplitude=1, r_eff=25, n=4, x_0=50, y_0=50,
                              c=-1.0, ellip=0.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)

        fig, ax = plt.subplots()
        im = ax.imshow(log_img, origin=\'lower\', interpolation=\'nearest\',
                       vmin=-1, vmax=2)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(\'Log Brightness\', rotation=270, labelpad=25)
        cbar.set_ticks([-1, 0, 1, 2])
        plt.title(\'Disky isophote with c=-1.0\')
        plt.xlabel(\'x\')
        plt.ylabel(\'y\')
        plt.show()

    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import GeneralSersic2D
        import matplotlib.pyplot as plt

        x, y = np.meshgrid(np.arange(100), np.arange(100))

        mod = GeneralSersic2D(amplitude=1, r_eff=25, n=4, x_0=50, y_0=50,
                              c=1.0, ellip=0.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)

        fig, ax = plt.subplots()
        im = ax.imshow(log_img, origin=\'lower\', interpolation=\'nearest\',
                       vmin=-1, vmax=2)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(\'Log Brightness\', rotation=270, labelpad=25)
        cbar.set_ticks([-1, 0, 1, 2])
        plt.title(\'Boxy isophote with c=1.0\')
        plt.xlabel(\'x\')
        plt.ylabel(\'y\')
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    .. [2] https://ui.adsabs.harvard.edu/abs/1990MNRAS.245..130A/abstract
    '''
    amplitude: Incomplete
    r_eff: Incomplete
    n: Incomplete
    x_0: Incomplete
    y_0: Incomplete
    ellip: Incomplete
    theta: Incomplete
    c: Incomplete

class KingProjectedAnalytic1D(Fittable1DModel):
    '''
    Projected (surface density) analytic King Model.


    Parameters
    ----------
    amplitude : float
        Amplitude or scaling factor.
    r_core : float
        Core radius (f(r_c) ~ 0.5 f_0)
    r_tide : float
        Tidal radius.


    Notes
    -----
    This model approximates a King model with an analytic function. The derivation of this
    equation can be found in King \'62 (equation 14). This is just an approximation of the
    full model and the parameters derived from this model should be taken with caution.
    It usually works for models with a concentration (c = log10(r_t/r_c) parameter < 2.

    Model formula:

    .. math::

        f(x) = A r_c^2  \\left(\\frac{1}{\\sqrt{(x^2 + r_c^2)}} -
        \\frac{1}{\\sqrt{(r_t^2 + r_c^2)}}\\right)^2

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import KingProjectedAnalytic1D
        import matplotlib.pyplot as plt

        plt.figure()
        rt_list = [1, 2, 5, 10, 20]
        for rt in rt_list:
            r = np.linspace(0.1, rt, 100)

            mod = KingProjectedAnalytic1D(amplitude = 1, r_core = 1., r_tide = rt)
            sig = mod(r)


            plt.loglog(r, sig/sig[0], label=f"c ~ {mod.concentration:0.2f}")

        plt.xlabel("r")
        plt.ylabel(r"$\\sigma/\\sigma_0$")
        plt.legend()
        plt.show()

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1962AJ.....67..471K
    '''
    amplitude: Incomplete
    r_core: Incomplete
    r_tide: Incomplete
    @property
    def concentration(self):
        """Concentration parameter of the king model."""
    @staticmethod
    def _core_func(x, r_core, r_tide, power: int = 1): ...
    @staticmethod
    def _filter(x, r_tide, result) -> None:
        """Set invalid r values to 0"""
    def evaluate(self, x, amplitude, r_core, r_tide):
        """
        Analytic King model function.
        """
    def fit_deriv(self, x, amplitude, r_core, r_tide):
        """
        Analytic King model function derivatives.
        """
    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        The model is not defined for r > r_tide.

        ``(r_low, r_high)``
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Logarithmic1D(Fittable1DModel):
    """
    One dimensional logarithmic model.

    Parameters
    ----------
    amplitude : float, optional
    tau : float, optional

    See Also
    --------
    Exponential1D, Gaussian1D
    """
    amplitude: Incomplete
    tau: Incomplete
    @staticmethod
    def evaluate(x, amplitude, tau): ...
    @staticmethod
    def fit_deriv(x, amplitude, tau): ...
    @property
    def inverse(self): ...
    def _tau_validator(self, val) -> None: ...
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Exponential1D(Fittable1DModel):
    """
    One dimensional exponential model.

    Parameters
    ----------
    amplitude : float, optional
    tau : float, optional

    See Also
    --------
    Logarithmic1D, Gaussian1D
    """
    amplitude: Incomplete
    tau: Incomplete
    @staticmethod
    def evaluate(x, amplitude, tau): ...
    @staticmethod
    def fit_deriv(x, amplitude, tau):
        """Derivative with respect to parameters."""
    @property
    def inverse(self): ...
    def _tau_validator(self, val) -> None:
        """tau cannot be 0."""
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...
