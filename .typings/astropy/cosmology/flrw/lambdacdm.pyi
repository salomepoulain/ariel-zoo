from .base import FLRW, FlatFLRWMixin

__all__ = ['LambdaCDM', 'FlatLambdaCDM']

class LambdaCDM(FLRW):
    """FLRW cosmology with a cosmological constant and curvature.

    This has no additional attributes beyond those of FLRW.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0.  If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0.

    Ode0 : float
        Omega dark energy: density of the cosmological constant in units of
        the critical density at z=0.

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
    >>> from astropy.cosmology import LambdaCDM
    >>> cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    The comoving distance in Mpc at redshift z:

    >>> z = 0.5
    >>> dc = cosmo.comoving_distance(z)
    """
    def __post_init__(self) -> None: ...
    def _optimize_flat_norad(self) -> None:
        """Set optimizations for flat LCDM cosmologies with no radiation."""
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
        units where c=1. Here this is :math:`w(z) = -1`.
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

        Notes
        -----
        The scaling factor, I, is defined by :math:`\\rho(z) = \\rho_0 I`,
        and in this case is given by :math:`I = 1`.
        """
    def _elliptic_comoving_distance_z1z2(self, z1, z2, /):
        """Comoving transverse distance in Mpc between two redshifts.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\\Omega_k` is zero.

        For :math:`\\Omega_{rad} = 0` the comoving distance can be directly
        calculated as an elliptic integral [1]_.

        Not valid or appropriate for flat cosmologies (Ok0=0).

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

        References
        ----------
        .. [1] Kantowski, R., Kao, J., & Thomas, R. (2000). Distance-Redshift
               in Inhomogeneous FLRW. arXiv e-prints, astro-ph/0002334.
        """
    def _dS_comoving_distance_z1z2(self, z1, z2, /):
        """De Sitter comoving LoS distance in Mpc between two redshifts.

        The Comoving line-of-sight distance in Mpc between objects at
        redshifts ``z1`` and ``z2`` in a flat, :math:`\\Omega_{\\Lambda}=1`
        cosmology (de Sitter).

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        The de Sitter case has an analytic solution.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'] or array-like, positional-only
            Input redshifts. Must be 1D or scalar.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving distance in Mpc between each input redshift.
        """
    def _EdS_comoving_distance_z1z2(self, z1, z2, /):
        """Einstein-de Sitter comoving LoS distance in Mpc between two redshifts.

        The Comoving line-of-sight distance in Mpc between objects at
        redshifts ``z1`` and ``z2`` in a flat, :math:`\\Omega_M=1`
        cosmology (Einstein - de Sitter).

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        For :math:`\\Omega_M=1`, :math:`\\Omega_{rad}=0` the comoving
        distance has an analytic solution.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'] or array-like, positional-only
            Input redshifts. Must be 1D or scalar.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving distance in Mpc between each input redshift.
        """
    def _hypergeometric_comoving_distance_z1z2(self, z1, z2, /):
        """Hypergeoemtric comoving LoS distance in Mpc between two redshifts.

        The Comoving line-of-sight distance in Mpc at redshifts ``z1`` and
        ``z2``.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        For :math:`\\Omega_{rad} = 0` the comoving distance can be directly
        calculated as a hypergeometric function [1]_.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'] or array-like, positional-only
            Input redshifts.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        d : Quantity ['length']
            Comoving distance in Mpc between each input redshift.

        References
        ----------
        .. [1] Baes, M., Camps, P., & Van De Putte, D. (2017). Analytical
               expressions and numerical evaluation of the luminosity
               distance in a flat cosmology. MNRAS, 468(1), 927-930.
        """
    def _T_hypergeometric(self, x, /):
        """Compute value using Gauss Hypergeometric function 2F1.

        .. math::

           T(x) = 2 \\sqrt(x) _{2}F_{1}\\left(\\frac{1}{6}, \\frac{1}{2};
                                            \\frac{7}{6}; -x^3 \\right)

        Notes
        -----
        The :func:`scipy.special.hyp2f1` code already implements the
        hypergeometric transformation suggested by Baes et al. [1]_ for use in
        actual numerical evaluations.

        References
        ----------
        .. [1] Baes, M., Camps, P., & Van De Putte, D. (2017). Analytical
           expressions and numerical evaluation of the luminosity distance
           in a flat cosmology. MNRAS, 468(1), 927-930.
        """
    def _dS_age(self, z, /):
        """Age of the universe in Gyr at redshift ``z``.

        The age of a de Sitter Universe is infinite.

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
    def _EdS_age(self, z, /):
        """Age of the universe in Gyr at redshift ``z``.

        For :math:`\\Omega_{rad} = 0` (:math:`T_{CMB} = 0`; massless neutrinos)
        the age can be directly calculated as an elliptic integral [1]_.

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

        References
        ----------
        .. [1] Thomas, R., & Kantowski, R. (2000). Age-redshift relation for
               standard cosmology. PRD, 62(10), 103507.
        """
    def _flat_age(self, z, /):
        """Age of the universe in Gyr at redshift ``z``.

        For :math:`\\Omega_{rad} = 0` (:math:`T_{CMB} = 0`; massless neutrinos)
        the age can be directly calculated as an elliptic integral [1]_.

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

        References
        ----------
        .. [1] Thomas, R., & Kantowski, R. (2000). Age-redshift relation for
               standard cosmology. PRD, 62(10), 103507.
        """
    def _EdS_lookback_time(self, z, /):
        """Lookback time in Gyr to redshift ``z``.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.

        For :math:`\\Omega_{rad} = 0` (:math:`T_{CMB} = 0`; massless neutrinos)
        the age can be directly calculated as an elliptic integral.
        The lookback time is here calculated based on the ``age(0) - age(z)``.

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
    def _dS_lookback_time(self, z, /):
        """Lookback time in Gyr to redshift ``z``.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.

        For :math:`\\Omega_{rad} = 0` (:math:`T_{CMB} = 0`; massless neutrinos)
        the age can be directly calculated.

        .. math::

           a = exp(H * t) \\  \\text{where t=0 at z=0}

           t = (1/H) (ln 1 - ln a) = (1/H) (0 - ln (1/(1+z))) = (1/H) ln(1+z)

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
    def _flat_lookback_time(self, z, /):
        """Lookback time in Gyr to redshift ``z``.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.

        For :math:`\\Omega_{rad} = 0` (:math:`T_{CMB} = 0`; massless neutrinos)
        the age can be directly calculated.
        The lookback time is here calculated based on the ``age(0) - age(z)``.

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
        """
    def inv_efunc(self, z):
        """Function used to calculate :math:`\\frac{1}{H_z}`.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        E : ndarray or float
            The inverse redshift scaling of the Hubble constant.
            Returns `float` if the input is scalar.
            Defined such that :math:`H_z = H_0 / E`.
        """

class FlatLambdaCDM(FlatFLRWMixin, LambdaCDM):
    """FLRW cosmology with a cosmological constant and no curvature.

    This has no additional attributes beyond those of FLRW.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0.

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
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    The comoving distance in Mpc at redshift z:

    >>> z = 0.5
    >>> dc = cosmo.comoving_distance(z)

    To get an equivalent cosmology, but of type `astropy.cosmology.LambdaCDM`,
    use :attr:`astropy.cosmology.FlatFLRWMixin.nonflat`.

    >>> print(cosmo.nonflat)
    LambdaCDM(H0=70.0 km / (Mpc s), Om0=0.3, Ode0=0.7, ...
    """
    def __post_init__(self) -> None: ...
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
        """
    def inv_efunc(self, z):
        """Function used to calculate :math:`\\frac{1}{H_z}`.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        E : ndarray or float
            The inverse redshift scaling of the Hubble constant.
            Returns `float` if the input is scalar.
            Defined such that :math:`H_z = H_0 / E`.
        """
