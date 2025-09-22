import abc
import astropy.units as u
from abc import abstractmethod
from astropy.cosmology.core import Cosmology, FlatCosmologyMixin
from astropy.cosmology.parameter import Parameter
from astropy.units import Quantity
from collections.abc import Mapping
from functools import cached_property
from numpy.typing import ArrayLike, NDArray
from typing import Any, Self, TypeVar

__all__ = ['FLRW', 'FlatFLRWMixin']

_FLRWT = TypeVar('_FLRWT', bound='FLRW')
_FlatFLRWMixinT = TypeVar('_FlatFLRWMixinT', bound='FlatFLRWMixin')

class _ScaleFactor:
    """The object has attributes and methods for computing the cosmological scale factor.

    The scale factor is defined as :math:`a = 1 / (1 + z)`.

    Attributes
    ----------
    scale_factor0 : `~astropy.units.Quantity`
        Scale factor at redshift 0.

    Methods
    -------
    scale_factor
        Compute the scale factor at a given redshift.
    """
    @property
    def scale_factor0(self) -> Quantity:
        """Scale factor at redshift 0.

        The scale factor is defined as :math:`a = \\frac{a_0}{1 + z}`. The common
        convention is to set :math:`a_0 = 1`. However, in some cases, e.g. in
        some old CMB papers, :math:`a_0` is used to normalize `a` to be a
        convenient number at the redshift of interest for that paper. Explicitly
        using :math:`a_0` in both calculation and code avoids ambiguity.
        """
    def scale_factor(self, z: Quantity | ArrayLike) -> Quantity | NDArray[Any] | float:
        """Scale factor at redshift ``z``.

        The scale factor is defined as :math:`a = 1 / (1 + z)`.

        Parameters
        ----------
        z : Quantity-like ['redshift'] | array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        |Quantity| | ndarray | float
            Scale factor at each input redshift.
            Returns `float` if the input is scalar.
        """

class FLRW(Cosmology, _ScaleFactor, metaclass=abc.ABCMeta):
    """An isotropic and homogeneous (Friedmann-Lemaitre-Robertson-Walker) cosmology.

    This is an abstract base class -- you cannot instantiate examples of this
    class, but must work with one of its subclasses, such as
    :class:`~astropy.cosmology.LambdaCDM` or :class:`~astropy.cosmology.wCDM`.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0.  If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0. Note that this does not include massive
        neutrinos.

    Ode0 : float
        Omega dark energy: density of dark energy in units of the critical
        density at z=0.

    Tcmb0 : float or scalar quantity-like ['temperature'], optional
        Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K].
        Setting this to zero will turn off both photons and neutrinos
        (even massive ones).

    Neff : float, optional
        Effective number of Neutrino species. Default 3.04.

    m_nu : quantity-like ['energy', 'mass'] or array-like, optional
        Mass of each neutrino species in [eV] (mass-energy equivalency enabled).
        If this is a scalar Quantity, then all neutrino species are assumed to
        have that mass. Otherwise, the mass of each species. The actual number
        of neutrino species (and hence the number of elements of m_nu if it is
        not scalar) must be the floor of Neff. Typically this means you should
        provide three neutrino masses unless you are considering something like
        a sterile neutrino.

    Ob0 : float, optional
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.

    name : str or None (optional, keyword-only)
        Name for this cosmological object.

    meta : mapping or None (optional, keyword-only)
        Metadata for the cosmology, e.g., a reference.

    Notes
    -----
    Class instances are immutable -- you cannot change the parameters' values.
    That is, all of the above attributes (except meta) are read only.

    For details on how to create performant custom subclasses, see the
    documentation on :ref:`astropy-cosmology-fast-integrals`.
    """
    H0: Parameter
    Om0: Parameter
    Ode0: Parameter
    Tcmb0: Parameter
    Neff: Parameter
    m_nu: Parameter
    Ob0: Parameter
    def __post_init__(self) -> None: ...
    def Ob0(self, param, value):
        """Validate baryon density to a non-negative float > matter density."""
    def m_nu(self, param, value):
        """Validate neutrino masses to right value, units, and shape.

        There are no neutrinos if floor(Neff) or Tcmb0 are 0. The number of
        neutrinos must match floor(Neff). Neutrino masses cannot be
        negative.
        """
    @property
    def is_flat(self) -> bool:
        """Return bool; `True` if the cosmology is flat."""
    @property
    def Otot0(self) -> float:
        """Omega total; the total density/critical density at z=0."""
    @cached_property
    def Odm0(self) -> float:
        """Omega dark matter; dark matter density/critical density at z=0."""
    @cached_property
    def Ok0(self) -> float:
        """Omega curvature; the effective curvature density/critical density at z=0."""
    @cached_property
    def Tnu0(self) -> u.Quantity:
        """Temperature of the neutrino background as |Quantity| at z=0."""
    @property
    def has_massive_nu(self) -> bool:
        """Does this cosmology have at least one massive neutrino species?"""
    @cached_property
    def h(self) -> float:
        """Dimensionless Hubble constant: h = H_0 / 100 [km/sec/Mpc]."""
    @cached_property
    def hubble_time(self) -> u.Quantity:
        """Hubble time."""
    @cached_property
    def hubble_distance(self) -> u.Quantity:
        """Hubble distance."""
    @cached_property
    def critical_density0(self) -> u.Quantity:
        """Critical density at z=0."""
    @cached_property
    def Ogamma0(self) -> float:
        """Omega gamma; the density/critical density of photons at z=0."""
    @cached_property
    def Onu0(self) -> float:
        """Omega nu; the density/critical density of neutrinos at z=0."""
    @abstractmethod
    def w(self, z):
        """The dark energy equation of state.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        w : ndarray or float
            The dark energy equation of state.
            `float` if scalar input.

        Notes
        -----
        The dark energy equation of state is defined as
        :math:`w(z) = P(z)/\\rho(z)`, where :math:`P(z)` is the pressure at
        redshift z and :math:`\\rho(z)` is the density at redshift z, both in
        units where c=1.

        This must be overridden by subclasses.
        """
    def Otot(self, z):
        """The total density parameter at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshifts.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Otot : ndarray or float
            The total density relative to the critical density at each redshift.
            Returns float if input scalar.
        """
    def Om(self, z):
        """Return the density parameter for non-relativistic matter at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Om : ndarray or float
            The density of non-relativistic matter relative to the critical
            density at each redshift.
            Returns `float` if the input is scalar.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest; see `Onu`.
        """
    def Ob(self, z):
        """Return the density parameter for baryonic matter at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Ob : ndarray or float
            The density of baryonic matter relative to the critical density at
            each redshift.
            Returns `float` if the input is scalar.
        """
    def Odm(self, z):
        """Return the density parameter for dark matter at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Odm : ndarray or float
            The density of non-relativistic dark matter relative to the
            critical density at each redshift.
            Returns `float` if the input is scalar.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest.
        """
    def Ok(self, z):
        """Return the equivalent density parameter for curvature at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Ok : ndarray or float
            The equivalent density parameter for curvature at each redshift.
            Returns `float` if the input is scalar.
        """
    def Ode(self, z):
        """Return the density parameter for dark energy at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Ode : ndarray or float
            The density of non-relativistic matter relative to the critical
            density at each redshift.
            Returns `float` if the input is scalar.
        """
    def Ogamma(self, z):
        """Return the density parameter for photons at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Ogamma : ndarray or float
            The energy density of photons relative to the critical density at
            each redshift.
            Returns `float` if the input is scalar.
        """
    def Onu(self, z):
        """Return the density parameter for neutrinos at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Onu : ndarray or float
            The energy density of neutrinos relative to the critical density at
            each redshift. Note that this includes their kinetic energy (if
            they have mass), so it is not equal to the commonly used
            :math:`\\sum \\frac{m_{\\nu}}{94 eV}`, which does not include
            kinetic energy.
            Returns `float` if the input is scalar.
        """
    def Tcmb(self, z):
        """Return the CMB temperature at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Tcmb : Quantity ['temperature']
            The temperature of the CMB in K.
        """
    def Tnu(self, z):
        """Return the neutrino temperature at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Tnu : Quantity ['temperature']
            The temperature of the cosmic neutrino background in K.
        """
    def nu_relative_density(self, z):
        """Neutrino density function relative to the energy density in photons.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        f : ndarray or float
            The neutrino density scaling factor relative to the density in
            photons at each redshift.
            Only returns `float` if z is scalar.

        Notes
        -----
        The density in neutrinos is given by

        .. math::

           \\rho_{\\nu} \\left(a\\right) = 0.2271 \\, N_{eff} \\,
           f\\left(m_{\\nu} a / T_{\\nu 0} \\right) \\,
           \\rho_{\\gamma} \\left( a \\right)

        where

        .. math::

           f \\left(y\\right) = \\frac{120}{7 \\pi^4}
           \\int_0^{\\infty} \\, dx \\frac{x^2 \\sqrt{x^2 + y^2}}
           {e^x + 1}

        assuming that all neutrino species have the same mass.
        If they have different masses, a similar term is calculated for each
        one. Note that ``f`` has the asymptotic behavior :math:`f(0) = 1`. This
        method returns :math:`0.2271 f` using an analytical fitting formula
        given in Komatsu et al. 2011, ApJS 192, 18.
        """
    def _w_integrand(self, ln1pz, /):
        """Internal convenience function for w(z) integral (eq. 5 of [1]_).

        Parameters
        ----------
        ln1pz : `~numbers.Number` or scalar ndarray, positional-only
            Assumes scalar input, since this should only be called inside an
            integral.

            .. versionchanged:: 7.0
                The argument is positional-only.

        References
        ----------
        .. [1] Linder, E. (2003). Exploring the Expansion History of the
               Universe. Phys. Rev. Lett., 90, 091301.
        """
    def de_density_scale(self, z):
        """Evaluates the redshift dependence of the dark energy density.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        I : ndarray or float
            The scaling of the energy density of dark energy with redshift.
            Returns `float` if the input is scalar.

        Notes
        -----
        The scaling factor, I, is defined by :math:`\\rho(z) = \\rho_0 I`,
        and is given by

        .. math::

           I = \\exp \\left( 3 \\int_{a}^1 \\frac{ da^{\\prime} }{ a^{\\prime} }
                          \\left[ 1 + w\\left( a^{\\prime} \\right) \\right] \\right)

        The actual integral used is rewritten from [1]_ to be in terms of z.

        It will generally helpful for subclasses to overload this method if
        the integral can be done analytically for the particular dark
        energy equation of state that they implement.

        References
        ----------
        .. [1] Linder, E. (2003). Exploring the Expansion History of the
               Universe. Phys. Rev. Lett., 90, 091301.
        """
    def efunc(self, z):
        """Function used to calculate H(z), the Hubble parameter.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        E : ndarray or float
            The redshift scaling of the Hubble constant.
            Returns `float` if the input is scalar.
            Defined such that :math:`H(z) = H_0 E(z)`.

        Notes
        -----
        It is not necessary to override this method, but if de_density_scale
        takes a particularly simple form, it may be advantageous to.
        """
    def inv_efunc(self, z):
        """Inverse of ``efunc``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        E : ndarray or float
            The redshift scaling of the inverse Hubble constant.
            Returns `float` if the input is scalar.
        """
    def _lookback_time_integrand_scalar(self, z, /):
        """Integrand of the lookback time (equation 30 of [1]_).

        Parameters
        ----------
        z : float, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                The argument is positional-only.

        Returns
        -------
        I : float
            The integrand for the lookback time.

        References
        ----------
        .. [1] Hogg, D. (1999). Distance measures in cosmology, section 11.
               arXiv e-prints, astro-ph/9905116.
        """
    def lookback_time_integrand(self, z):
        """Integrand of the lookback time (equation 30 of [1]_).

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        I : float or array
            The integrand for the lookback time.

        References
        ----------
        .. [1] Hogg, D. (1999). Distance measures in cosmology, section 11.
               arXiv e-prints, astro-ph/9905116.
        """
    def _abs_distance_integrand_scalar(self, z, /):
        """Integrand of the absorption distance (eq. 4, [1]_).

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                The argument is positional-only.

        Returns
        -------
        dX : float
            The integrand for the absorption distance (dimensionless).

        References
        ----------
        .. [1] Bahcall, John N. and Peebles, P.J.E. 1969, ApJ, 156L, 7B
        """
    def abs_distance_integrand(self, z):
        """Integrand of the absorption distance (eq. 4, [1]_).

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        dX : float or array
            The integrand for the absorption distance (dimensionless).

        References
        ----------
        .. [1] Bahcall, John N. and Peebles, P.J.E. 1969, ApJ, 156L, 7B
        """
    def H(self, z):
        """Hubble parameter (km/s/Mpc) at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        H : Quantity ['frequency']
            Hubble parameter at each input redshift.
        """
    def lookback_time(self, z):
        """Lookback time in Gyr to redshift ``z``.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        t : Quantity ['time']
            Lookback time in Gyr to each input redshift.

        See Also
        --------
        z_at_value : Find the redshift corresponding to a lookback time.
        """
    def _lookback_time(self, z, /):
        """Lookback time in Gyr to redshift ``z``.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                The argument is positional-only.

        Returns
        -------
        t : Quantity ['time']
            Lookback time in Gyr to each input redshift.
        """
    def _integral_lookback_time(self, z, /):
        """Lookback time to redshift ``z``. Value in units of Hubble time.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                The argument is positional-only.

        Returns
        -------
        t : float or ndarray
            Lookback time to each input redshift in Hubble time units.
            Returns `float` if input scalar, `~numpy.ndarray` otherwise.
        """
    def lookback_distance(self, z):
        """The lookback distance is the light travel time distance to a given redshift.

        It is simply c * lookback_time. It may be used to calculate
        the proper distance between two redshifts, e.g. for the mean free path
        to ionizing radiation.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Lookback distance in Mpc
        """
    def age(self, z):
        """Age of the universe in Gyr at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        t : Quantity ['time']
            The age of the universe in Gyr at each input redshift.

        See Also
        --------
        z_at_value : Find the redshift corresponding to an age.
        """
    def _age(self, z, /):
        """Age of the universe in Gyr at redshift ``z``.

        This internal function exists to be re-defined for optimizations.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                The argument is positional-only.

        Returns
        -------
        t : Quantity ['time']
            The age of the universe in Gyr at each input redshift.
        """
    def _integral_age(self, z, /):
        """Age of the universe at redshift ``z``. Value in units of Hubble time.

        Calculated using explicit integration.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

        Returns
        -------
        t : float or ndarray
            The age of the universe at each input redshift in Hubble time units.
            Returns `float` if input scalar, `~numpy.ndarray` otherwise.

        See Also
        --------
        z_at_value : Find the redshift corresponding to an age.
        """
    def critical_density(self, z):
        """Critical density in grams per cubic cm at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        rho : Quantity ['mass density']
            Critical density at each input redshift.
        """
    def comoving_distance(self, z):
        """Comoving line-of-sight distance in Mpc at a given redshift.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving distance in Mpc to each input redshift.
        """
    def _comoving_distance_z1z2(self, z1, z2, /):
        """Comoving line-of-sight distance in Mpc between redshifts ``z1`` and ``z2``.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving distance in Mpc between each input redshift.
        """
    def _integral_comoving_distance_z1z2_scalar(self, z1, z2, /):
        """Comoving line-of-sight distance in Mpc between objects at redshifts ``z1`` and ``z2``.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : float or ndarray
            Comoving distance in Mpc between each input redshift.
            Returns `float` if input scalar, `~numpy.ndarray` otherwise.
        """
    def _integral_comoving_distance_z1z2(self, z1, z2, /):
        """Comoving line-of-sight distance in Mpc between objects at redshifts ``z1`` and ``z2``.

        The comoving distance along the line-of-sight between two objects remains
        constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'] or array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving distance in Mpc between each input redshift.
        """
    def comoving_transverse_distance(self, z):
        """Comoving transverse distance in Mpc at a given redshift.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving transverse distance in Mpc at each input redshift.

        Notes
        -----
        This quantity is also called the 'proper motion distance' in some texts.
        """
    def _comoving_transverse_distance_z1z2(self, z1, z2, /):
        """Comoving transverse distance in Mpc between two redshifts.

        This value is the transverse comoving distance at redshift ``z2`` as
        seen from redshift ``z1`` corresponding to an angular separation of
        1 radian. This is the same as the comoving distance if :math:`\\Omega_k`
        is zero (as in the current concordance Lambda-CDM model).

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'], array-like, positional-only
            Input redshifts.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving transverse distance in Mpc between input redshift.

        Notes
        -----
        This quantity is also called the 'proper motion distance' in some texts.
        """
    def angular_diameter_distance(self, z):
        """Angular diameter distance in Mpc at a given redshift.

        This gives the proper (sometimes called 'physical') transverse
        distance corresponding to an angle of 1 radian for an object
        at redshift ``z`` ([1]_, [2]_, [3]_).

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Angular diameter distance in Mpc at each input redshift.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.
        """
    def luminosity_distance(self, z):
        """Luminosity distance in Mpc at redshift ``z``.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Luminosity distance in Mpc at each input redshift.

        See Also
        --------
        z_at_value : Find the redshift corresponding to a luminosity distance.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
    def angular_diameter_distance_z1z2(self, z1, z2):
        """Angular diameter distance between objects at 2 redshifts.

        Useful for gravitational lensing, for example computing the angular
        diameter distance between a lensed galaxy and the foreground lens.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'], array-like
            Input redshifts. For most practical applications such as
            gravitational lensing, ``z2`` should be larger than ``z1``. The
            method will work for ``z2 < z1``; however, this will return
            negative distances.

        Returns
        -------
        d : Quantity ['length']
            The angular diameter distance between each input redshift pair.
            Returns scalar if input is scalar, array else-wise.
        """
    def absorption_distance(self, z, /):
        """Absorption distance at redshift ``z`` (eq. 4, [1]_).

        This is used to calculate the number of objects with some cross section
        of absorption and number density intersecting a sightline per unit
        redshift path [1]_.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, positional-only
            Input redshift.

        Returns
        -------
        X : float or ndarray
            Absorption distance (dimensionless) at each input redshift.
            Returns `float` if input scalar, `~numpy.ndarray` otherwise.

        References
        ----------
        .. [1] Bahcall, John N. and Peebles, P.J.E. 1969, ApJ, 156L, 7B
        """
    def distmod(self, z):
        """Distance modulus at redshift ``z``.

        The distance modulus is defined as the (apparent magnitude - absolute
        magnitude) for an object at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        distmod : Quantity ['length']
            Distance modulus at each input redshift, in magnitudes.

        See Also
        --------
        z_at_value : Find the redshift corresponding to a distance modulus.
        """
    def comoving_volume(self, z):
        """Comoving volume in cubic Mpc at redshift ``z``.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\\Omega_k` is not.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        V : Quantity ['volume']
            Comoving volume in :math:`Mpc^3` at each input redshift.
        """
    def differential_comoving_volume(self, z):
        """Differential comoving volume at redshift z.

        Useful for calculating the effective comoving volume.
        For example, allows for integration over a comoving volume that has a
        sensitivity function that changes with redshift. The total comoving
        volume is given by integrating ``differential_comoving_volume`` to
        redshift ``z`` and multiplying by a solid angle.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        dV : Quantity
            Differential comoving volume per redshift per steradian at each
            input redshift.
        """
    def kpc_comoving_per_arcmin(self, z):
        """Separation in transverse comoving kpc equal to an arcmin at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            The distance in comoving kpc corresponding to an arcmin at each
            input redshift.
        """
    def kpc_proper_per_arcmin(self, z):
        """Separation in transverse proper kpc equal to an arcminute at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            The distance in proper kpc corresponding to an arcmin at each input
            redshift.
        """
    def arcsec_per_kpc_comoving(self, z):
        """Angular separation in arcsec equal to a comoving kpc at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        theta : Quantity ['angle']
            The angular separation in arcsec corresponding to a comoving kpc at
            each input redshift.
        """
    def arcsec_per_kpc_proper(self, z):
        """Angular separation in arcsec corresponding to a proper kpc at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        theta : Quantity ['angle']
            The angular separation in arcsec corresponding to a proper kpc at
            each input redshift.
        """

class FlatFLRWMixin(FlatCosmologyMixin):
    """Mixin class for flat FLRW cosmologies.

    Do NOT instantiate directly. Must precede the base class in the
    multiple-inheritance so that this mixin's ``__init__`` proceeds the
    base class'. Note that all instances of ``FlatFLRWMixin`` are flat, but
    not all flat cosmologies are instances of ``FlatFLRWMixin``. As
    example, ``LambdaCDM`` **may** be flat (for the a specific set of
    parameter values), but ``FlatLambdaCDM`` **will** be flat.
    """
    Ode0: Parameter
    def __init_subclass__(cls) -> None: ...
    def __post_init__(self) -> None: ...
    def nonflat(self) -> _FLRWT: ...
    @property
    def Otot0(self):
        """Omega total; the total density/critical density at z=0."""
    def Otot(self, z):
        """The total density parameter at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        Otot : ndarray or float
            Returns float if input scalar. Value of 1.
        """
    def clone(self, *, meta: Mapping | None = None, to_nonflat: bool = False, **kwargs) -> Self: ...
