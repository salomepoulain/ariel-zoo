import abc
from .core import FittableModel, Model
from _typeshed import Incomplete

__all__ = ['Chebyshev1D', 'Chebyshev2D', 'Hermite1D', 'Hermite2D', 'InverseSIP', 'Legendre1D', 'Legendre2D', 'Polynomial1D', 'Polynomial2D', 'SIP', 'OrthoPolynomialBase', 'PolynomialModel']

class PolynomialBase(FittableModel, metaclass=abc.ABCMeta):
    """
    Base class for all polynomial-like models with an arbitrary number of
    parameters in the form of coefficients.

    In this case Parameter instances are returned through the class's
    ``__getattr__`` rather than through class descriptors.
    """
    _param_names: Incomplete
    linear: bool
    col_fit_deriv: bool
    @property
    def param_names(self):
        """Coefficient names generated based on the model's polynomial degree
        and number of dimensions.

        Subclasses should implement this to return parameter names in the
        desired format.

        On most `Model` classes this is a class attribute, but for polynomial
        models it is an instance attribute since each polynomial model instance
        can have different parameters depending on the degree of the polynomial
        and the number of dimensions, for example.
        """

class PolynomialModel(PolynomialBase, metaclass=abc.ABCMeta):
    """
    Base class for polynomial models.

    Its main purpose is to determine how many coefficients are needed
    based on the polynomial order and dimension and to provide their
    default values, names and ordering.
    """
    _degree: Incomplete
    _order: Incomplete
    _param_names: Incomplete
    def __init__(self, degree, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    @property
    def degree(self):
        """Degree of polynomial."""
    def get_num_coeff(self, ndim):
        """
        Return the number of coefficients in one parameter set.
        """
    def _invlex(self): ...
    def _generate_coeff_names(self, ndim): ...

class _PolyDomainWindow1D(PolynomialModel, metaclass=abc.ABCMeta):
    """
    This class sets ``domain`` and ``window`` of 1D polynomials.
    """
    def __init__(self, degree, domain: Incomplete | None = None, window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    @property
    def window(self): ...
    _window: Incomplete
    @window.setter
    def window(self, val) -> None: ...
    @property
    def domain(self): ...
    _domain: Incomplete
    @domain.setter
    def domain(self, val) -> None: ...
    _default_domain_window: Incomplete
    def _set_default_domain_window(self, domain, window) -> None:
        """
        This method sets the ``domain`` and ``window`` attributes on 1D subclasses.

        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class OrthoPolynomialBase(PolynomialBase):
    """
    This is a base class for the 2D Chebyshev and Legendre models.

    The polynomials implemented here require a maximum degree in x and y.

    For explanation of ``x_domain``, ``y_domain``, ```x_window`` and ```y_window``
    see :ref:`Notes regarding usage of domain and window <astropy:domain-window-note>`.


    Parameters
    ----------
    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : tuple or None, optional
        domain of the x independent variable
    x_window : tuple or None, optional
        range of the x independent variable
    y_domain : tuple or None, optional
        domain of the y independent variable
    y_window : tuple or None, optional
        range of the y independent variable
    **params : dict
        {keyword: value} pairs, representing {parameter_name: value}
    """
    n_inputs: int
    n_outputs: int
    x_degree: Incomplete
    y_degree: Incomplete
    _order: Incomplete
    _default_domain_window: Incomplete
    _param_names: Incomplete
    def __init__(self, x_degree, y_degree, x_domain: Incomplete | None = None, x_window: Incomplete | None = None, y_domain: Incomplete | None = None, y_window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    @property
    def x_domain(self): ...
    _x_domain: Incomplete
    @x_domain.setter
    def x_domain(self, val) -> None: ...
    @property
    def y_domain(self): ...
    _y_domain: Incomplete
    @y_domain.setter
    def y_domain(self, val) -> None: ...
    @property
    def x_window(self): ...
    _x_window: Incomplete
    @x_window.setter
    def x_window(self, val) -> None: ...
    @property
    def y_window(self): ...
    _y_window: Incomplete
    @y_window.setter
    def y_window(self, val) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def get_num_coeff(self):
        """
        Determine how many coefficients are needed.

        Returns
        -------
        numc : int
            number of coefficients
        """
    def _invlex(self): ...
    def invlex_coeff(self, coeffs): ...
    def _alpha(self): ...
    def imhorner(self, x, y, coeff): ...
    def _generate_coeff_names(self): ...
    def _fcache(self, x, y) -> None:
        '''
        Computation and store the individual functions.

        To be implemented by subclasses"
        '''
    def evaluate(self, x, y, *coeffs): ...
    def prepare_inputs(self, x, y, **kwargs): ...

class Chebyshev1D(_PolyDomainWindow1D):
    """
    Univariate Chebyshev series.

    It is defined as:

    .. math::

        P(x) = \\sum_{i=0}^{i=n}C_{i} * T_{i}(x)

    where ``T_i(x)`` is the corresponding Chebyshev polynomial of the 1st kind.

    For explanation of ```domain``, and ``window`` see
    :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : tuple or None, optional
    window : tuple or None, optional
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window.
    **params : dict
        keyword : value pairs, representing parameter_name: value

    Notes
    -----
    This model does not support the use of units/quantities, because each term
    in the sum of Chebyshev polynomials is a polynomial in x - since the
    coefficients within each Chebyshev polynomial are fixed, we can't use
    quantities for x since the units would not be compatible. For example, the
    third Chebyshev polynomial (T2) is 2x^2-1, but if x was specified with
    units, 2x^2 and -1 would have incompatible units.
    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    def __init__(self, degree, domain: Incomplete | None = None, window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def fit_deriv(self, x, *params):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        x : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    def prepare_inputs(self, x, **kwargs): ...
    def evaluate(self, x, *coeffs): ...
    @staticmethod
    def clenshaw(x, coeffs):
        """Evaluates the polynomial using Clenshaw's algorithm."""

class Hermite1D(_PolyDomainWindow1D):
    '''
    Univariate Hermite series.

    It is defined as:

    .. math::

        P(x) = \\sum_{i=0}^{i=n}C_{i} * H_{i}(x)

    where ``H_i(x)`` is the corresponding Hermite polynomial ("Physicist\'s kind").

    For explanation of ``domain``, and ``window`` see
    :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : tuple or None, optional
    window : tuple or None, optional
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    **params : dict
        keyword : value pairs, representing parameter_name: value

    Notes
    -----
    This model does not support the use of units/quantities, because each term
    in the sum of Hermite polynomials is a polynomial in x - since the
    coefficients within each Hermite polynomial are fixed, we can\'t use
    quantities for x since the units would not be compatible. For example, the
    third Hermite polynomial (H2) is 4x^2-2, but if x was specified with units,
    4x^2 and -2 would have incompatible units.
    '''
    n_inputs: int
    n_outputs: int
    _separable: bool
    def __init__(self, degree, domain: Incomplete | None = None, window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def fit_deriv(self, x, *params):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        x : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    def prepare_inputs(self, x, **kwargs): ...
    def evaluate(self, x, *coeffs): ...
    @staticmethod
    def clenshaw(x, coeffs): ...

class Hermite2D(OrthoPolynomialBase):
    """
    Bivariate Hermite series.

    It is defined as

    .. math:: P_{nm}(x,y) = \\sum_{n,m=0}^{n=d,m=d}C_{nm} H_n(x) H_m(y)

    where ``H_n(x)`` and ``H_m(y)`` are Hermite polynomials.

    For explanation of ``x_domain``, ``y_domain``, ``x_window`` and ``y_window``
    see :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : tuple or None, optional
        domain of the x independent variable
    y_domain : tuple or None, optional
        domain of the y independent variable
    x_window : tuple or None, optional
        range of the x independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    y_window : tuple or None, optional
        range of the y independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    **params : dict
        keyword: value pairs, representing parameter_name: value

    Notes
    -----
    This model does not support the use of units/quantities, because each term
    in the sum of Hermite polynomials is a polynomial in x and/or y - since the
    coefficients within each Hermite polynomial are fixed, we can't use
    quantities for x and/or y since the units would not be compatible. For
    example, the third Hermite polynomial (H2) is 4x^2-2, but if x was
    specified with units, 4x^2 and -2 would have incompatible units.
    """
    _separable: bool
    def __init__(self, x_degree, y_degree, x_domain: Incomplete | None = None, x_window: Incomplete | None = None, y_domain: Incomplete | None = None, y_window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def _fcache(self, x, y):
        """
        Calculate the individual Hermite functions once and store them in a
        dictionary to be reused.
        """
    def fit_deriv(self, x, y, *params):
        """
        Derivatives with respect to the coefficients.

        This is an array with Hermite polynomials:

        .. math::

            H_{x_0}H_{y_0}, H_{x_1}H_{y_0}...H_{x_n}H_{y_0}...H_{x_n}H_{y_m}

        Parameters
        ----------
        x : ndarray
            input
        y : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    def _hermderiv1d(self, x, deg):
        """
        Derivative of 1D Hermite series.
        """

class Legendre1D(_PolyDomainWindow1D):
    """
    Univariate Legendre series.

    It is defined as:

    .. math::

        P(x) = \\sum_{i=0}^{i=n}C_{i} * L_{i}(x)

    where ``L_i(x)`` is the corresponding Legendre polynomial.

    For explanation of ``domain``, and ``window`` see
    :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : tuple or None, optional
    window : tuple or None, optional
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    **params : dict
        keyword: value pairs, representing parameter_name: value


    Notes
    -----
    This model does not support the use of units/quantities, because each term
    in the sum of Legendre polynomials is a polynomial in x - since the
    coefficients within each Legendre polynomial are fixed, we can't use
    quantities for x since the units would not be compatible. For example, the
    third Legendre polynomial (P2) is 1.5x^2-0.5, but if x was specified with
    units, 1.5x^2 and -0.5 would have incompatible units.
    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    def __init__(self, degree, domain: Incomplete | None = None, window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def prepare_inputs(self, x, **kwargs): ...
    def evaluate(self, x, *coeffs): ...
    def fit_deriv(self, x, *params):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        x : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    @staticmethod
    def clenshaw(x, coeffs): ...

class Polynomial1D(_PolyDomainWindow1D):
    """
    1D Polynomial model.

    It is defined as:

    .. math::

        P = \\sum_{i=0}^{i=n}C_{i} * x^{i}

    For explanation of ``domain``, and ``window`` see
    :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    degree : int
        degree of the series
    domain : tuple or None, optional
        If None, it is set to (-1, 1)
    window : tuple or None, optional
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    **params : dict
        keyword: value pairs, representing parameter_name: value

    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    _default_domain_window: Incomplete
    domain: Incomplete
    window: Incomplete
    def __init__(self, degree, domain: Incomplete | None = None, window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def prepare_inputs(self, x, **kwargs): ...
    def evaluate(self, x, *coeffs): ...
    def fit_deriv(self, x, *params):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        x : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    @staticmethod
    def horner(x, coeffs): ...
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class Polynomial2D(PolynomialModel):
    """
    2D Polynomial  model.

    Represents a general polynomial of degree n:

    .. math::

        P(x,y) = c_{00} + c_{10}x + ...+ c_{n0}x^n + c_{01}y + ...+ c_{0n}y^n
        + c_{11}xy + c_{12}xy^2 + ... + c_{1(n-1)}xy^{n-1}+ ... + c_{(n-1)1}x^{n-1}y

    For explanation of ``x_domain``, ``y_domain``, ``x_window`` and ``y_window``
    see :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    degree : int
        Polynomial degree: largest sum of exponents (:math:`i + j`) of
        variables in each monomial term of the form :math:`x^i y^j`. The
        number of terms in a 2D polynomial of degree ``n`` is given by binomial
        coefficient :math:`C(n + 2, 2) = (n + 2)! / (2!\\,n!) = (n + 1)(n + 2) / 2`.
    x_domain : tuple or None, optional
        domain of the x independent variable
        If None, it is set to (-1, 1)
    y_domain : tuple or None, optional
        domain of the y independent variable
        If None, it is set to (-1, 1)
    x_window : tuple or None, optional
        range of the x independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the x_domain to x_window
    y_window : tuple or None, optional
        range of the y independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the y_domain to y_window
    **params : dict
        keyword: value pairs, representing parameter_name: value
    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    _default_domain_window: Incomplete
    def __init__(self, degree, x_domain: Incomplete | None = None, y_domain: Incomplete | None = None, x_window: Incomplete | None = None, y_window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def prepare_inputs(self, x, y, **kwargs): ...
    def evaluate(self, x, y, *coeffs): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def fit_deriv(self, x, y, *params):
        """
        Computes the Vandermonde matrix.

        Parameters
        ----------
        x : ndarray
            input
        y : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    def invlex_coeff(self, coeffs): ...
    def multivariate_horner(self, x, y, coeffs):
        """
        Multivariate Horner's scheme.

        Parameters
        ----------
        x, y : array
        coeffs : array
            Coefficients in inverse lexical order.
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...
    @property
    def x_domain(self): ...
    _x_domain: Incomplete
    @x_domain.setter
    def x_domain(self, val) -> None: ...
    @property
    def y_domain(self): ...
    _y_domain: Incomplete
    @y_domain.setter
    def y_domain(self, val) -> None: ...
    @property
    def x_window(self): ...
    _x_window: Incomplete
    @x_window.setter
    def x_window(self, val) -> None: ...
    @property
    def y_window(self): ...
    _y_window: Incomplete
    @y_window.setter
    def y_window(self, val) -> None: ...

class Chebyshev2D(OrthoPolynomialBase):
    """
    Bivariate Chebyshev series..

    It is defined as

    .. math:: P_{nm}(x,y) = \\sum_{n,m=0}^{n=d,m=d}C_{nm}  T_n(x ) T_m(y)

    where ``T_n(x)`` and ``T_m(y)`` are Chebyshev polynomials of the first kind.

    For explanation of ``x_domain``, ``y_domain``, ``x_window`` and ``y_window``
    see :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : tuple or None, optional
        domain of the x independent variable
    y_domain : tuple or None, optional
        domain of the y independent variable
    x_window : tuple or None, optional
        range of the x independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    y_window : tuple or None, optional
        range of the y independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window

    **params : dict
        keyword: value pairs, representing parameter_name: value

    Notes
    -----
    This model does not support the use of units/quantities, because each term
    in the sum of Chebyshev polynomials is a polynomial in x and/or y - since
    the coefficients within each Chebyshev polynomial are fixed, we can't use
    quantities for x and/or y since the units would not be compatible. For
    example, the third Chebyshev polynomial (T2) is 2x^2-1, but if x was
    specified with units, 2x^2 and -1 would have incompatible units.
    """
    _separable: bool
    def __init__(self, x_degree, y_degree, x_domain: Incomplete | None = None, x_window: Incomplete | None = None, y_domain: Incomplete | None = None, y_window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def _fcache(self, x, y):
        """
        Calculate the individual Chebyshev functions once and store them in a
        dictionary to be reused.
        """
    def fit_deriv(self, x, y, *params):
        """
        Derivatives with respect to the coefficients.

        This is an array with Chebyshev polynomials:

        .. math::

            T_{x_0}T_{y_0}, T_{x_1}T_{y_0}...T_{x_n}T_{y_0}...T_{x_n}T_{y_m}

        Parameters
        ----------
        x : ndarray
            input
        y : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    def _chebderiv1d(self, x, deg):
        """
        Derivative of 1D Chebyshev series.
        """

class Legendre2D(OrthoPolynomialBase):
    """
    Bivariate Legendre series.

    Defined as:

    .. math:: P_{n_m}(x,y) = \\sum_{n,m=0}^{n=d,m=d}C_{nm}  L_n(x ) L_m(y)

    where ``L_n(x)`` and ``L_m(y)`` are Legendre polynomials.

    For explanation of ``x_domain``, ``y_domain``, ``x_window`` and ``y_window``
    see :ref:`Notes regarding usage of domain and window <domain-window-note>`.

    Parameters
    ----------
    x_degree : int
        degree in x
    y_degree : int
        degree in y
    x_domain : tuple or None, optional
        domain of the x independent variable
    y_domain : tuple or None, optional
        domain of the y independent variable
    x_window : tuple or None, optional
        range of the x independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    y_window : tuple or None, optional
        range of the y independent variable
        If None, it is set to (-1, 1)
        Fitters will remap the domain to this window
    **params : dict
        keyword: value pairs, representing parameter_name: value

    Notes
    -----
    Model formula:

    .. math::

        P(x) = \\sum_{i=0}^{i=n}C_{i} * L_{i}(x)

    where ``L_{i}`` is the corresponding Legendre polynomial.

    This model does not support the use of units/quantities, because each term
    in the sum of Legendre polynomials is a polynomial in x - since the
    coefficients within each Legendre polynomial are fixed, we can't use
    quantities for x since the units would not be compatible. For example, the
    third Legendre polynomial (P2) is 1.5x^2-0.5, but if x was specified with
    units, 1.5x^2 and -0.5 would have incompatible units.
    """
    _separable: bool
    def __init__(self, x_degree, y_degree, x_domain: Incomplete | None = None, x_window: Incomplete | None = None, y_domain: Incomplete | None = None, y_window: Incomplete | None = None, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def _fcache(self, x, y):
        """
        Calculate the individual Legendre functions once and store them in a
        dictionary to be reused.
        """
    def fit_deriv(self, x, y, *params):
        """Derivatives with respect to the coefficients.

        This is an array with Legendre polynomials:

        Lx0Ly0  Lx1Ly0...LxnLy0...LxnLym

        Parameters
        ----------
        x : ndarray
            input
        y : ndarray
            input
        *params
            throw-away parameter list returned by non-linear fitters

        Returns
        -------
        result : ndarray
            The Vandermonde matrix
        """
    def _legendderiv1d(self, x, deg):
        """Derivative of 1D Legendre polynomial."""

class _SIP1D(PolynomialBase):
    """
    This implements the Simple Imaging Polynomial Model (SIP) in 1D.

    It's unlikely it will be used in 1D so this class is private
    and SIP should be used instead.
    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    order: Incomplete
    coeff_prefix: Incomplete
    _param_names: Incomplete
    def __init__(self, order, coeff_prefix, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None, **params) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def evaluate(self, x, y, *coeffs): ...
    def get_num_coeff(self, ndim):
        """
        Return the number of coefficients in one param set.
        """
    def _generate_coeff_names(self, coeff_prefix): ...
    def _coeff_matrix(self, coeff_prefix, coeffs): ...
    def _eval_sip(self, x, y, coef): ...

class SIP(Model):
    """
    Simple Imaging Polynomial (SIP) model.

    The SIP convention is used to represent distortions in FITS image headers.
    See [1]_ for a description of the SIP convention.

    Parameters
    ----------
    crpix : list or (2,) ndarray
        CRPIX values
    a_order : int
        SIP polynomial order for first axis
    b_order : int
        SIP order for second axis
    a_coeff : dict
        SIP coefficients for first axis
    b_coeff : dict
        SIP coefficients for the second axis
    ap_order : int
        order for the inverse transformation (AP coefficients)
    bp_order : int
        order for the inverse transformation (BP coefficients)
    ap_coeff : dict
        coefficients for the inverse transform
    bp_coeff : dict
        coefficients for the inverse transform

    References
    ----------
    .. [1] `David Shupe, et al, ADASS, ASP Conference Series, Vol. 347, 2005
        <https://ui.adsabs.harvard.edu/abs/2005ASPC..347..491S>`_
    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    _crpix: Incomplete
    _a_order: Incomplete
    _b_order: Incomplete
    _a_coeff: Incomplete
    _b_coeff: Incomplete
    _ap_order: Incomplete
    _bp_order: Incomplete
    _ap_coeff: Incomplete
    _bp_coeff: Incomplete
    shift_a: Incomplete
    shift_b: Incomplete
    sip1d_a: Incomplete
    sip1d_b: Incomplete
    _inputs: Incomplete
    _outputs: Incomplete
    def __init__(self, crpix, a_order, b_order, a_coeff={}, b_coeff={}, ap_order: Incomplete | None = None, bp_order: Incomplete | None = None, ap_coeff={}, bp_coeff={}, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def inverse(self): ...
    def evaluate(self, x, y): ...

class InverseSIP(Model):
    """
    Inverse Simple Imaging Polynomial.

    Parameters
    ----------
    ap_order : int
        order for the inverse transformation (AP coefficients)
    bp_order : int
        order for the inverse transformation (BP coefficients)
    ap_coeff : dict
        coefficients for the inverse transform
    bp_coeff : dict
        coefficients for the inverse transform

    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    _ap_order: Incomplete
    _bp_order: Incomplete
    _ap_coeff: Incomplete
    _bp_coeff: Incomplete
    sip1d_ap: Incomplete
    sip1d_bp: Incomplete
    def __init__(self, ap_order, bp_order, ap_coeff={}, bp_coeff={}, n_models: Incomplete | None = None, model_set_axis: Incomplete | None = None, name: Incomplete | None = None, meta: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def evaluate(self, x, y): ...
