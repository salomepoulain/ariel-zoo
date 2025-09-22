from .core import Fittable1DModel
from _typeshed import Incomplete

__all__ = ['BlackBody', 'Drude1D', 'Plummer1D', 'NFW']

class BlackBody(Fittable1DModel):
    """
    Blackbody model using the Planck function.

    Parameters
    ----------
    temperature : `~astropy.units.Quantity` ['temperature']
        Blackbody temperature.

    scale : float or `~astropy.units.Quantity` ['dimensionless']
        Scale factor.  If dimensionless, input units will assumed
        to be in Hz and output units in (erg / (cm ** 2 * s * Hz * sr).
        If not dimensionless, must be equivalent to either
        (erg / (cm ** 2 * s * Hz * sr) or erg / (cm ** 2 * s * AA * sr),
        in which case the result will be returned in the requested units and
        the scale will be stripped of units (with the float value applied).

    Notes
    -----
    Model formula:

        .. math:: B_{\\nu}(T) = A \\frac{2 h \\nu^{3} / c^{2}}{exp(h \\nu / k T) - 1}

    Examples
    --------
    >>> from astropy.modeling import models
    >>> from astropy import units as u
    >>> bb = models.BlackBody(temperature=5000*u.K)
    >>> bb(6000 * u.AA)  # doctest: +FLOAT_CMP
    <Quantity 1.53254685e-05 erg / (Hz s sr cm2)>

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import BlackBody
        from astropy import units as u
        from astropy.visualization import quantity_support

        bb = BlackBody(temperature=5778*u.K)
        wav = np.arange(1000, 110000) * u.AA
        flux = bb(wav)

        with quantity_support():
            plt.figure()
            plt.semilogx(wav, flux)
            plt.axvline(bb.nu_max.to(u.AA, equivalencies=u.spectral()).value, ls='--')
            plt.show()
    """
    temperature: Incomplete
    scale: Incomplete
    _input_units_allow_dimensionless: bool
    input_units_equivalencies: Incomplete
    _native_units: Incomplete
    _native_output_units: Incomplete
    _output_units: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def evaluate(self, x, temperature, scale):
        """Evaluate the model.

        Parameters
        ----------
        x : float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['frequency']
            Frequency at which to compute the blackbody. If no units are given,
            this defaults to Hz (or AA if `scale` was initialized with units
            equivalent to erg / (cm ** 2 * s * AA * sr)).

        temperature : float, `~numpy.ndarray`, or `~astropy.units.Quantity`
            Temperature of the blackbody. If no units are given, this defaults
            to Kelvin.

        scale : float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['dimensionless']
            Desired scale for the blackbody.

        Returns
        -------
        y : number or ndarray
            Blackbody spectrum. The units are determined from the units of
            ``scale``.

        .. note::

            Use `numpy.errstate` to suppress Numpy warnings, if desired.

        .. warning::

            Output values might contain ``nan`` and ``inf``.

        Raises
        ------
        ValueError
            Invalid temperature.

        ZeroDivisionError
            Wavelength is zero (when converting to frequency).
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...
    @property
    def bolometric_flux(self):
        """Bolometric flux."""
    @property
    def lambda_max(self):
        """Peak wavelength when the curve is expressed as power density."""
    @property
    def nu_max(self):
        """Peak frequency when the curve is expressed as power density."""

class Drude1D(Fittable1DModel):
    """
    Drude model based one the behavior of electons in materials (esp. metals).

    Parameters
    ----------
    amplitude : float
        Peak value
    x_0 : float
        Position of the peak
    fwhm : float
        Full width at half maximum

    Model formula:

        .. math:: f(x) = A \\frac{(fwhm/x_0)^2}{((x/x_0 - x_0/x)^2 + (fwhm/x_0)^2}

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Drude1D

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(7.5 , 12.5 , 0.1)

        dmodel = Drude1D(amplitude=1.0, fwhm=1.0, x_0=10.0)
        ax.plot(x, dmodel(x))

        ax.set_xlabel('x')
        ax.set_ylabel('F(x)')

        plt.show()
    """
    amplitude: Incomplete
    x_0: Incomplete
    fwhm: Incomplete
    @staticmethod
    def evaluate(x, amplitude, x_0, fwhm):
        """
        One dimensional Drude model function.
        """
    @staticmethod
    def fit_deriv(x, amplitude, x_0, fwhm):
        """
        Drude1D model function derivatives.
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...
    @property
    def return_units(self): ...
    def _x_0_validator(self, val) -> None:
        """Ensure `x_0` is not 0."""
    def bounding_box(self, factor: int = 50):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of FWHM used to define the limits.
        """

class Plummer1D(Fittable1DModel):
    """One dimensional Plummer density profile model.

    Parameters
    ----------
    mass : float
        Total mass of cluster.
    r_plum : float
        Scale parameter which sets the size of the cluster core.

    Notes
    -----
    Model formula:

    .. math::

        \\rho(r)=\\frac{3M}{4\\pi a^3}(1+\\frac{r^2}{a^2})^{-5/2}

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1911MNRAS..71..460P
    """
    mass: Incomplete
    r_plum: Incomplete
    @staticmethod
    def evaluate(x, mass, r_plum):
        """
        Evaluate plummer density profile model.
        """
    @staticmethod
    def fit_deriv(x, mass, r_plum):
        """
        Plummer1D model derivatives.
        """
    @property
    def input_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...

class NFW(Fittable1DModel):
    '''
    Navarro–Frenk–White (NFW) profile - model for radial distribution of dark matter.

    Parameters
    ----------
    mass : float or `~astropy.units.Quantity` [\'mass\']
        Mass of NFW peak within specified overdensity radius.
    concentration : float
        Concentration of the NFW profile.
    redshift : float
        Redshift of the NFW profile.
    massfactor : tuple or str
        Mass overdensity factor and type for provided profiles:
            Tuple version:
                ("virial",) : virial radius

                ("critical", N)  : radius where density is N times that of the critical density

                ("mean", N)  : radius where density is N times that of the mean density

            String version:
                "virial" : virial radius

                "Nc"  : radius where density is N times that of the critical density (e.g. "200c")

                "Nm"  : radius where density is N times that of the mean density (e.g. "500m")
    cosmo : :class:`~astropy.cosmology.Cosmology`
        Background cosmology for density calculation. If None, the default cosmology will be used.

    Notes
    -----
    Model formula:

    .. math:: \\rho(r)=\\frac{\\delta_c\\rho_{c}}{r/r_s(1+r/r_s)^2}

    References
    ----------
    .. [1] https://arxiv.org/pdf/astro-ph/9508025
    .. [2] https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile
    .. [3] https://en.wikipedia.org/wiki/Virial_mass
    '''
    mass: Incomplete
    concentration: Incomplete
    redshift: Incomplete
    _input_units_allow_dimensionless: bool
    def __init__(self, mass=..., concentration=..., redshift=..., massfactor=('critical', 200), cosmo: Incomplete | None = None, **kwargs) -> None: ...
    def evaluate(self, r, mass, concentration, redshift):
        """
        One dimensional NFW profile function.

        Parameters
        ----------
        r : float or `~astropy.units.Quantity` ['length']
            Radial position of density to be calculated for the NFW profile.
        mass : float or `~astropy.units.Quantity` ['mass']
            Mass of NFW peak within specified overdensity radius.
        concentration : float
            Concentration of the NFW profile.
        redshift : float
            Redshift of the NFW profile.

        Returns
        -------
        density : float or `~astropy.units.Quantity` ['density']
            NFW profile mass density at location ``r``. The density units are:
            [``mass`` / ``r`` ^3]

        Notes
        -----
        .. warning::

            Output values might contain ``nan`` and ``inf``.
        """
    density_delta: Incomplete
    def _density_delta(self, massfactor, cosmo, redshift):
        """
        Calculate density delta.
        """
    @staticmethod
    def A_NFW(y):
        """
        Dimensionless volume integral of the NFW profile, used as an intermediate step in some
        calculations for this model.

        Notes
        -----
        Model formula:

        .. math:: A_{NFW} = [\\ln(1+y) - \\frac{y}{1+y}]
        """
    density_s: Incomplete
    def _density_s(self, mass, concentration):
        """
        Calculate scale density of the NFW profile.
        """
    @property
    def rho_scale(self):
        """
        Scale density of the NFW profile. Often written in the literature as :math:`\\rho_s`.
        """
    radius_s: Incomplete
    def _radius_s(self, mass, concentration):
        """
        Calculate scale radius of the NFW profile.
        """
    @property
    def r_s(self):
        """
        Scale radius of the NFW profile.
        """
    @property
    def r_virial(self):
        """
        Mass factor defined virial radius of the NFW profile (R200c for M200c, Rvir for Mvir, etc.).
        """
    @property
    def r_max(self):
        """
        Radius of maximum circular velocity.
        """
    @property
    def v_max(self):
        """
        Maximum circular velocity.
        """
    def circular_velocity(self, r):
        """
        Circular velocities of the NFW profile.

        Parameters
        ----------
        r : float or `~astropy.units.Quantity` ['length']
            Radial position of velocity to be calculated for the NFW profile.

        Returns
        -------
        velocity : float or `~astropy.units.Quantity` ['speed']
            NFW profile circular velocity at location ``r``. The velocity units are:
            [km / s]

        Notes
        -----
        Model formula:

        .. math:: v_{circ}(r)^2 = \\frac{1}{x}\\frac{\\ln(1+cx)-(cx)/(1+cx)}{\\ln(1+c)-c/(1+c)}

        .. math:: x = r/r_s

        .. warning::

            Output values might contain ``nan`` and ``inf``.
        """
    @property
    def input_units(self): ...
    @property
    def return_units(self): ...
    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit): ...
