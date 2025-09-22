from .angles import Latitude, Longitude
from _typeshed import Incomplete
from astropy import units as u
from astropy.units.quantity import QuantityInfoBase
from typing import NamedTuple

__all__ = ['EarthLocation']

class GeodeticLocation(NamedTuple):
    """A namedtuple for geodetic coordinates.

    The longitude is increasing to the east, so west longitudes are negative.
    """
    lon: Longitude
    lat: Latitude
    height: u.Quantity

class EarthLocationInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_attrs: Incomplete
    def _construct_from_dict(self, map): ...
    def new_like(self, cols, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new EarthLocation instance which is consistent with the
        input ``cols`` and has ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        cols : list
            List of input columns
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : EarthLocation (or subclass)
            Empty instance of this class consistent with ``cols``
        """

class EarthLocation(u.Quantity):
    """
    Location on the Earth.

    Initialization is first attempted assuming geocentric (x, y, z) coordinates
    are given; if that fails, another attempt is made assuming geodetic
    coordinates (longitude, latitude, height above a reference ellipsoid).
    When using the geodetic forms, Longitudes are measured increasing to the
    east, so west longitudes are negative. Internally, the coordinates are
    stored as geocentric.

    To ensure a specific type of coordinates is used, use the corresponding
    class methods (`from_geocentric` and `from_geodetic`) or initialize the
    arguments with names (``x``, ``y``, ``z`` for geocentric; ``lon``, ``lat``,
    ``height`` for geodetic).  See the class methods for details.


    Notes
    -----
    This class fits into the coordinates transformation framework in that it
    encodes a position on the `~astropy.coordinates.ITRS` frame.  To get a
    proper `~astropy.coordinates.ITRS` object from this object, use the ``itrs``
    property.
    """
    _ellipsoid: str
    _location_dtype: Incomplete
    _array_dtype: Incomplete
    _site_registry: Incomplete
    info: Incomplete
    def __new__(cls, *args, **kwargs): ...
    @classmethod
    def from_geocentric(cls, x, y, z, unit: Incomplete | None = None):
        """
        Location on Earth, initialized from geocentric coordinates.

        Parameters
        ----------
        x, y, z : `~astropy.units.Quantity` or array-like
            Cartesian coordinates.  If not quantities, ``unit`` should be given.
        unit : unit-like or None
            Physical unit of the coordinate values.  If ``x``, ``y``, and/or
            ``z`` are quantities, they will be converted to this unit.

        Raises
        ------
        astropy.units.UnitsError
            If the units on ``x``, ``y``, and ``z`` do not match or an invalid
            unit is given.
        ValueError
            If the shapes of ``x``, ``y``, and ``z`` do not match.
        TypeError
            If ``x`` is not a `~astropy.units.Quantity` and no unit is given.
        """
    @classmethod
    def from_geodetic(cls, lon, lat, height: float = 0.0, ellipsoid: Incomplete | None = None):
        """
        Location on Earth, initialized from geodetic coordinates.

        Parameters
        ----------
        lon : `~astropy.coordinates.Longitude` or float
            Earth East longitude.  Can be anything that initialises an
            `~astropy.coordinates.Angle` object (if float, in degrees).
        lat : `~astropy.coordinates.Latitude` or float
            Earth latitude.  Can be anything that initialises an
            `~astropy.coordinates.Latitude` object (if float, in degrees).
        height : `~astropy.units.Quantity` ['length'] or float, optional
            Height above reference ellipsoid (if float, in meters; default: 0).
        ellipsoid : str, optional
            Name of the reference ellipsoid to use (default: 'WGS84').
            Available ellipsoids are:  'WGS84', 'GRS80', 'WGS72'.

        Raises
        ------
        astropy.units.UnitsError
            If the units on ``lon`` and ``lat`` are inconsistent with angular
            ones, or that on ``height`` with a length.
        ValueError
            If ``lon``, ``lat``, and ``height`` do not have the same shape, or
            if ``ellipsoid`` is not recognized as among the ones implemented.

        Notes
        -----
        For the conversion to geocentric coordinates, the ERFA routine
        ``gd2gc`` is used.  See https://github.com/liberfa/erfa
        """
    @classmethod
    def of_site(cls, site_name, *, refresh_cache: bool = False):
        """
        Return an object of this class for a known observatory/site by name.

        This is intended as a quick convenience function to get basic site
        information, not a fully-featured exhaustive registry of observatories
        and all their properties.

        Additional information about the site is stored in the ``.info.meta``
        dictionary of sites obtained using this method (see the examples below).

        .. note::
            This function is meant to access the site registry from the astropy
            data server, which is saved in the user's local cache.  If you would
            like a site to be added there, issue a pull request to the
            `astropy-data repository <https://github.com/astropy/astropy-data>`_ .
            If the cache already exists the function will use it even if the
            version in the astropy-data repository has been updated unless the
            ``refresh_cache=True`` option is used.  If there is no cache and the
            online version cannot be reached, this function falls back on a
            built-in list, which currently only contains the Greenwich Royal
            Observatory as an example case.

        Parameters
        ----------
        site_name : str
            Name of the observatory (case-insensitive).
        refresh_cache : bool, optional
            If `True`, force replacement of the cached registry with a
            newly downloaded version.  (Default: `False`)

            .. versionadded:: 5.3

        Returns
        -------
        site : `~astropy.coordinates.EarthLocation` (or subclass) instance
            The location of the observatory. The returned class will be the same
            as this class.

        Examples
        --------
        >>> from astropy.coordinates import EarthLocation
        >>> keck = EarthLocation.of_site('Keck Observatory')  # doctest: +REMOTE_DATA
        >>> keck.geodetic  # doctest: +REMOTE_DATA +FLOAT_CMP
        GeodeticLocation(lon=<Longitude -155.47833333 deg>, lat=<Latitude 19.82833333 deg>, height=<Quantity 4160. m>)
        >>> keck.info  # doctest: +REMOTE_DATA
        name = W. M. Keck Observatory
        dtype = (float64, float64, float64)
        unit = m
        class = EarthLocation
        n_bad = 0
        >>> keck.info.meta  # doctest: +REMOTE_DATA
        {'source': 'IRAF Observatory Database', 'timezone': 'US/Hawaii'}

        See Also
        --------
        get_site_names : the list of sites that this function can access
        """
    @classmethod
    def of_address(cls, address, get_height: bool = False, google_api_key: Incomplete | None = None):
        """
        Return an object of this class for a given address by querying either
        the OpenStreetMap Nominatim tool [1]_ (default) or the Google geocoding
        API [2]_, which requires a specified API key.

        This is intended as a quick convenience function to get easy access to
        locations. If you need to specify a precise location, you should use the
        initializer directly and pass in a longitude, latitude, and elevation.

        In the background, this just issues a web query to either of
        the APIs noted above. This is not meant to be abused! Both
        OpenStreetMap and Google use IP-based query limiting and will ban your
        IP if you send more than a few thousand queries per hour [2]_.

        .. warning::
            If the query returns more than one location (e.g., searching on
            ``address='springfield'``), this function will use the **first**
            returned location.

        Parameters
        ----------
        address : str
            The address to get the location for. As per the Google maps API,
            this can be a fully specified street address (e.g., 123 Main St.,
            New York, NY) or a city name (e.g., Danbury, CT), or etc.
        get_height : bool, optional
            This only works when using the Google API! See the ``google_api_key``
            block below. Use the retrieved location to perform a second query to
            the Google maps elevation API to retrieve the height of the input
            address [3]_.
        google_api_key : str, optional
            A Google API key with the Geocoding API and (optionally) the
            elevation API enabled. See [4]_ for more information.

        Returns
        -------
        location : `~astropy.coordinates.EarthLocation` (or subclass) instance
            The location of the input address.
            Will be type(this class)

        References
        ----------
        .. [1] https://nominatim.openstreetmap.org/
        .. [2] https://developers.google.com/maps/documentation/geocoding/start
        .. [3] https://developers.google.com/maps/documentation/elevation/start
        .. [4] https://developers.google.com/maps/documentation/geocoding/get-api-key

        """
    @classmethod
    def get_site_names(cls, *, refresh_cache: bool = False):
        """
        Get list of names of observatories for use with
        `~astropy.coordinates.EarthLocation.of_site`.

        .. note::
            This function is meant to access the site registry from the astropy
            data server, which is saved in the user's local cache.  If you would
            like a site to be added there, issue a pull request to the
            `astropy-data repository <https://github.com/astropy/astropy-data>`_ .
            If the cache already exists the function will use it even if the
            version in the astropy-data repository has been updated unless the
            ``refresh_cache=True`` option is used.  If there is no cache and the
            online version cannot be reached, this function falls back on a
            built-in list, which currently only contains the Greenwich Royal
            Observatory as an example case.

        Parameters
        ----------
        refresh_cache : bool, optional
            If `True`, force replacement of the cached registry with a
            newly downloaded version.  (Default: `False`)

            .. versionadded:: 5.3

        Returns
        -------
        names : list of str
            List of valid observatory names

        See Also
        --------
        of_site : Gets the actual location object for one of the sites names
            this returns.
        """
    @classmethod
    def _get_site_registry(cls, force_download: bool = False, force_builtin: bool = False):
        """
        Gets the site registry.  The first time this either downloads or loads
        from the data file packaged with astropy.  Subsequent calls will use the
        cached version unless explicitly overridden.

        Parameters
        ----------
        force_download : bool or str
            If not False, force replacement of the cached registry with a
            downloaded version. If a str, that will be used as the URL to
            download from (if just True, the default URL will be used).
        force_builtin : bool
            If True, load from the data file bundled with astropy and set the
            cache to that.

        Returns
        -------
        reg : astropy.coordinates.sites.SiteRegistry
        """
    @property
    def ellipsoid(self):
        """The default ellipsoid used to convert to geodetic coordinates."""
    @ellipsoid.setter
    def ellipsoid(self, ellipsoid) -> None: ...
    @property
    def geodetic(self):
        """Convert to geodetic coordinates for the default ellipsoid."""
    def to_geodetic(self, ellipsoid: Incomplete | None = None):
        """Convert to geodetic coordinates.

        Parameters
        ----------
        ellipsoid : str, optional
            Reference ellipsoid to use.  Default is the one the coordinates
            were initialized with.  Available are: 'WGS84', 'GRS80', 'WGS72'

        Returns
        -------
        lon, lat, height : `~astropy.units.Quantity`
            The tuple is a ``GeodeticLocation`` namedtuple and is comprised of
            instances of `~astropy.coordinates.Longitude`,
            `~astropy.coordinates.Latitude`, and `~astropy.units.Quantity`.

        Raises
        ------
        ValueError
            if ``ellipsoid`` is not recognized as among the ones implemented.

        Notes
        -----
        For the conversion to geodetic coordinates, the ERFA routine
        ``gc2gd`` is used.  See https://github.com/liberfa/erfa
        """
    @property
    def lon(self):
        """Longitude of the location, for the default ellipsoid."""
    @property
    def lat(self):
        """Latitude of the location, for the default ellipsoid."""
    @property
    def height(self):
        """Height of the location, for the default ellipsoid."""
    @property
    def geocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities."""
    def to_geocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities."""
    def get_itrs(self, obstime: Incomplete | None = None, location: Incomplete | None = None):
        """
        Generates an `~astropy.coordinates.ITRS` object with the location of
        this object at the requested ``obstime``, either geocentric, or
        topocentric relative to a given ``location``.

        Parameters
        ----------
        obstime : `~astropy.time.Time` or None
            The ``obstime`` to apply to the new `~astropy.coordinates.ITRS`, or
            if None, the default ``obstime`` will be used.
        location : `~astropy.coordinates.EarthLocation` or None
            A possible observer's location, for a topocentric ITRS position.
            If not given (default), a geocentric ITRS object will be created.

        Returns
        -------
        itrs : `~astropy.coordinates.ITRS`
            The new object in the ITRS frame, either geocentric or topocentric
            relative to the given ``location``.
        """
    itrs: Incomplete
    def get_gcrs(self, obstime):
        """GCRS position with velocity at ``obstime`` as a GCRS coordinate.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the GCRS position/velocity at.

        Returns
        -------
        gcrs : `~astropy.coordinates.GCRS` instance
            With velocity included.
        """
    def _get_gcrs_posvel(self, obstime, ref_to_itrs, gcrs_to_ref):
        """Calculate GCRS position and velocity given transformation matrices.

        The reference frame z axis must point to the Celestial Intermediate Pole
        (as is the case for CIRS and TETE).

        This private method is used in intermediate_rotation_transforms,
        where some of the matrices are already available for the coordinate
        transformation.

        The method is faster by an order of magnitude than just adding a zero
        velocity to ITRS and transforming to GCRS, because it avoids calculating
        the velocity via finite differencing of the results of the transformation
        at three separate times.
        """
    def get_gcrs_posvel(self, obstime):
        """
        Calculate the GCRS position and velocity of this object at the
        requested ``obstime``.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the GCRS position/velocity at.

        Returns
        -------
        obsgeoloc : `~astropy.coordinates.CartesianRepresentation`
            The GCRS position of the object
        obsgeovel : `~astropy.coordinates.CartesianRepresentation`
            The GCRS velocity of the object
        """
    def gravitational_redshift(self, obstime, bodies=['sun', 'jupiter', 'moon'], masses={}):
        """Return the gravitational redshift at this EarthLocation.

        Calculates the gravitational redshift, of order 3 m/s, due to the
        requested solar system bodies.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the redshift at.

        bodies : iterable, optional
            The bodies (other than the Earth) to include in the redshift
            calculation.  List elements should be any body name
            `get_body_barycentric` accepts.  Defaults to Jupiter, the Sun, and
            the Moon.  Earth is always included (because the class represents
            an *Earth* location).

        masses : dict[str, `~astropy.units.Quantity`], optional
            The mass or gravitational parameters (G * mass) to assume for the
            bodies requested in ``bodies``. Can be used to override the
            defaults for the Sun, Jupiter, the Moon, and the Earth, or to
            pass in masses for other bodies.

        Returns
        -------
        redshift : `~astropy.units.Quantity`
            Gravitational redshift in velocity units at given obstime.
        """
    @property
    def x(self):
        """The X component of the geocentric coordinates."""
    @property
    def y(self):
        """The Y component of the geocentric coordinates."""
    @property
    def z(self):
        """The Z component of the geocentric coordinates."""
    def __getitem__(self, item): ...
    def __array_finalize__(self, obj) -> None: ...
    def __len__(self) -> int: ...
    def _to_value(self, unit, equivalencies=[]):
        """Helper method for to and to_value."""
