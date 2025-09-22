from .base import FLRW, FlatFLRWMixin
from astropy.cosmology.parameter import Parameter

__all__ = ['w0wzCDM', 'Flatw0wzCDM']

class w0wzCDM(FLRW):
    """FLRW cosmology with a variable dark energy EoS and curvature.

    The equation for the dark energy equation of state (EoS) uses the simple form:
    :math:`w(z) = w_0 + w_z z`.

    This form is not recommended for z > 1.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0.

    Ode0 : float
        Omega dark energy: density of dark energy in units of the critical
        density at z=0.

    w0 : float, optional
        Dark energy equation of state at z=0. This is pressure/density for
        dark energy in units where c=1.

    wz : float, optional
        Derivative of the dark energy equation of state with respect to z.
        A cosmological constant has w0=-1.0 and wz=0.0.

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

    Ob0 : float or None, optional
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.  If this is set to None (the default), any computation
        that requires its value will raise an exception.

    name : str or None (optional, keyword-only)
        Name for this cosmological object.

    meta : mapping or None (optional, keyword-only)
        Metadata for the cosmology, e.g., a reference.

    Examples
    --------
    >>> from astropy.cosmology import w0wzCDM
    >>> cosmo = w0wzCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-0.9, wz=0.2)

    The comoving distance in Mpc at redshift z:

    >>> z = 0.5
    >>> dc = cosmo.comoving_distance(z)
    """
    w0: Parameter
    wz: Parameter
    def __post_init__(self) -> None: ...
    def w(self, z):
        """Returns dark energy equation of state at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'] or array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        w : ndarray or float
            The dark energy equation of state.
            Returns `float` if the input is scalar.

        Notes
        -----
        The dark energy equation of state is defined as
        :math:`w(z) = P(z)/\\rho(z)`, where :math:`P(z)` is the pressure at
        redshift z and :math:`\\rho(z)` is the density at redshift z, both in
        units where c=1. Here this is given by :math:`w(z) = w_0 + w_z z`.
        """
    def de_density_scale(self, z):
        """Evaluates the redshift dependence of the dark energy density.

        Parameters
        ----------
        z : Quantity-like ['redshift'] or array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        I : ndarray or float
            The scaling of the energy density of dark energy with redshift.
            Returns `float` if the input is scalar.

        References
        ----------
        .. [1] Linder, E. (2003). Exploring the Expansion History of the Universe.
               Physics Review Letters, 90(9), 091301.

        Notes
        -----
        The scaling factor, I, is defined by :math:`\\rho(z) = \\rho_0 I`,
        and in this case is given by ([1]_)

        .. math::

           I = \\left(1 + z\\right)^{3 \\left(1 + w_0 - w_z\\right)}
                     \\exp \\left(3 w_z z\\right)
        """

class Flatw0wzCDM(FlatFLRWMixin, w0wzCDM):
    """FLRW cosmology with a variable dark energy EoS and no curvature.

    The equation for the dark energy equation of state (EoS) uses the simple form:
    :math:`w(z) = w_0 + w_z z`.

    This form is not recommended for z > 1.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0.

    w0 : float, optional
        Dark energy equation of state at z=0. This is pressure/density for
        dark energy in units where c=1.

    wz : float, optional
        Derivative of the dark energy equation of state with respect to z.
        A cosmological constant has w0=-1.0 and wz=0.0.

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

    Ob0 : float or None, optional
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.  If this is set to None (the default), any computation
        that requires its value will raise an exception.

    name : str or None (optional, keyword-only)
        Name for this cosmological object.

    meta : mapping or None (optional, keyword-only)
        Metadata for the cosmology, e.g., a reference.

    Examples
    --------
    >>> from astropy.cosmology import Flatw0wzCDM
    >>> cosmo = Flatw0wzCDM(H0=70, Om0=0.3, w0=-0.9, wz=0.2)

    The comoving distance in Mpc at redshift z:

    >>> cosmo.comoving_distance(0.5)
    <Quantity 1849.74726272 Mpc>
    """
    def __post_init__(self) -> None: ...
