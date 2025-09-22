from .base import BaseDifferential as BaseDifferential, BaseRepresentation as BaseRepresentation
from .cartesian import CartesianRepresentation as CartesianRepresentation
from _typeshed import Incomplete
from astropy.coordinates.angles import Angle as Angle, Latitude as Latitude, Longitude as Longitude
from astropy.coordinates.distances import Distance as Distance
from astropy.coordinates.matrix_utilities import is_O3 as is_O3
from astropy.utils import classproperty as classproperty
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED

class UnitSphericalRepresentation(BaseRepresentation):
    """
    Representation of points on a unit sphere.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity` ['angle'] or str
        The longitude and latitude of the point(s), in angular units. The
        latitude should be between -90 and 90 degrees, and the longitude will
        be wrapped to an angle between 0 and 360 degrees. These can also be
        instances of `~astropy.coordinates.Angle`,
        `~astropy.coordinates.Longitude`, or `~astropy.coordinates.Latitude`.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        instance (see `._compatible_differentials` for valid types), or a
        dictionary of of differential instances with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes: Incomplete
    def _dimensional_representation(cls): ...
    def __init__(self, lon, lat: Incomplete | None = None, differentials: Incomplete | None = None, copy: bool = True) -> None: ...
    def _compatible_differentials(cls): ...
    @property
    def lon(self):
        """
        The longitude of the point(s).
        """
    @property
    def lat(self):
        """
        The latitude of the point(s).
        """
    def unit_vectors(self): ...
    def scale_factors(self, omit_coslat: bool = False): ...
    def to_cartesian(self):
        """
        Converts spherical polar coordinates to 3D rectangular cartesian
        coordinates.
        """
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates to spherical polar
        coordinates.
        """
    def represent_as(self, other_class, differential_class: Incomplete | None = None): ...
    def transform(self, matrix):
        """Transform the unit-spherical coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).

        Returns
        -------
        `~astropy.coordinates.UnitSphericalRepresentation` or `~astropy.coordinates.SphericalRepresentation`
            If ``matrix`` is O(3) -- :math:`M \\dot M^T = I` -- like a rotation,
            then the result is a `~astropy.coordinates.UnitSphericalRepresentation`.
            All other matrices will change the distance, so the dimensional
            representation is used instead.

        """
    def _scale_operation(self, op, *args): ...
    def __neg__(self): ...
    def norm(self):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units, which is
        always unity for vectors on the unit sphere.

        Returns
        -------
        norm : `~astropy.units.Quantity` ['dimensionless']
            Dimensionless ones, with the same shape as the representation.
        """
    def _combine_operation(self, op, other, reverse: bool = False): ...
    def mean(self, *args, **kwargs):
        """Vector mean.

        The representation is converted to cartesian, the means of the x, y,
        and z components are calculated, and the result is converted to a
        `~astropy.coordinates.SphericalRepresentation`.

        Refer to `~numpy.mean` for full documentation of the arguments, noting
        that ``axis`` is the entry in the ``shape`` of the representation, and
        that the ``out`` argument cannot be used.
        """
    def sum(self, *args, **kwargs):
        """Vector sum.

        The representation is converted to cartesian, the sums of the x, y,
        and z components are calculated, and the result is converted to a
        `~astropy.coordinates.SphericalRepresentation`.

        Refer to `~numpy.sum` for full documentation of the arguments, noting
        that ``axis`` is the entry in the ``shape`` of the representation, and
        that the ``out`` argument cannot be used.
        """
    def cross(self, other):
        """Cross product of two representations.

        The calculation is done by converting both ``self`` and ``other``
        to `~astropy.coordinates.CartesianRepresentation`, and converting the
        result back to `~astropy.coordinates.SphericalRepresentation`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            The representation to take the cross product with.

        Returns
        -------
        cross_product : `~astropy.coordinates.SphericalRepresentation`
            With vectors perpendicular to both ``self`` and ``other``.
        """

class RadialRepresentation(BaseRepresentation):
    """
    Representation of the distance of points from the origin.

    Note that this is mostly intended as an internal helper representation.
    It can do little else but being used as a scale in multiplication.

    Parameters
    ----------
    distance : `~astropy.units.Quantity` ['length']
        The distance of the point(s) from the origin.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        instance (see `._compatible_differentials` for valid types), or a
        dictionary of of differential instances with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes: Incomplete
    def __init__(self, distance, differentials: Incomplete | None = None, copy: bool = True) -> None: ...
    @property
    def distance(self):
        """
        The distance from the origin to the point(s).
        """
    def unit_vectors(self) -> None:
        """Cartesian unit vectors are undefined for radial representation."""
    def scale_factors(self): ...
    def to_cartesian(self) -> None:
        """Cannot convert radial representation to cartesian."""
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates to radial coordinate.
        """
    def __mul__(self, other): ...
    def norm(self):
        """Vector norm.

        Just the distance itself.

        Returns
        -------
        norm : `~astropy.units.Quantity` ['dimensionless']
            Dimensionless ones, with the same shape as the representation.
        """
    def _combine_operation(self, op, other, reverse: bool = False): ...
    def transform(self, matrix):
        """Radial representations cannot be transformed by a Cartesian matrix.

        Parameters
        ----------
        matrix : array-like
            The transformation matrix in a Cartesian basis.
            Must be a multiplication: a diagonal matrix with identical elements.
            Must have shape (..., 3, 3), where the last 2 indices are for the
            matrix on each other axis. Make sure that the matrix shape is
            compatible with the shape of this representation.

        Raises
        ------
        ValueError
            If the matrix is not a multiplication.

        """

def _spherical_op_funcs(op, *args):
    """For given operator, return functions that adjust lon, lat, distance."""

class SphericalRepresentation(BaseRepresentation):
    """
    Representation of points in 3D spherical coordinates.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity` ['angle']
        The longitude and latitude of the point(s), in angular units. The
        latitude should be between -90 and 90 degrees, and the longitude will
        be wrapped to an angle between 0 and 360 degrees. These can also be
        instances of `~astropy.coordinates.Angle`,
        `~astropy.coordinates.Longitude`, or `~astropy.coordinates.Latitude`.

    distance : `~astropy.units.Quantity` ['length']
        The distance to the point(s). If the distance is a length, it is
        passed to the :class:`~astropy.coordinates.Distance` class, otherwise
        it is passed to the :class:`~astropy.units.Quantity` class.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        instance (see `._compatible_differentials` for valid types), or a
        dictionary of of differential instances with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes: Incomplete
    _unit_representation = UnitSphericalRepresentation
    _distance: Incomplete
    def __init__(self, lon, lat: Incomplete | None = None, distance: Incomplete | None = None, differentials: Incomplete | None = None, copy: bool = True) -> None: ...
    def _compatible_differentials(cls): ...
    @property
    def lon(self):
        """
        The longitude of the point(s).
        """
    @property
    def lat(self):
        """
        The latitude of the point(s).
        """
    @property
    def distance(self):
        """
        The distance from the origin to the point(s).
        """
    def unit_vectors(self): ...
    def scale_factors(self, omit_coslat: bool = False): ...
    def represent_as(self, other_class, differential_class: Incomplete | None = None): ...
    def to_cartesian(self):
        """
        Converts spherical polar coordinates to 3D rectangular cartesian
        coordinates.
        """
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates to spherical polar
        coordinates.
        """
    def transform(self, matrix):
        """Transform the spherical coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).

        """
    def norm(self):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.  For
        spherical coordinates, this is just the absolute value of the distance.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.
        """
    def _scale_operation(self, op, *args): ...

class PhysicsSphericalRepresentation(BaseRepresentation):
    """
    Representation of points in 3D spherical coordinates (using the physics
    convention of using ``phi`` and ``theta`` for azimuth and inclination
    from the pole).

    Parameters
    ----------
    phi, theta : `~astropy.units.Quantity` or str
        The azimuth and inclination of the point(s), in angular units. The
        inclination should be between 0 and 180 degrees, and the azimuth will
        be wrapped to an angle between 0 and 360 degrees. These can also be
        instances of `~astropy.coordinates.Angle`.  If ``copy`` is False, `phi`
        will be changed inplace if it is not between 0 and 360 degrees.

    r : `~astropy.units.Quantity`
        The distance to the point(s). If the distance is a length, it is
        passed to the :class:`~astropy.coordinates.Distance` class, otherwise
        it is passed to the :class:`~astropy.units.Quantity` class.

    differentials : dict, `~astropy.coordinates.PhysicsSphericalDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single
        `~astropy.coordinates.PhysicsSphericalDifferential` instance, or a dictionary of of
        differential instances with keys set to a string representation of the
        SI unit with which the differential (derivative) is taken. For example,
        for a velocity differential on a positional representation, the key
        would be ``'s'`` for seconds, indicating that the derivative is a time
        derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes: Incomplete
    _r: Incomplete
    def __init__(self, phi, theta: Incomplete | None = None, r: Incomplete | None = None, differentials: Incomplete | None = None, copy: bool = True) -> None: ...
    @property
    def phi(self):
        """
        The azimuth of the point(s).
        """
    @property
    def theta(self):
        """
        The elevation of the point(s).
        """
    @property
    def r(self):
        """
        The distance from the origin to the point(s).
        """
    def unit_vectors(self): ...
    def scale_factors(self): ...
    def represent_as(self, other_class, differential_class: Incomplete | None = None): ...
    def to_cartesian(self):
        """
        Converts spherical polar coordinates to 3D rectangular cartesian
        coordinates.
        """
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates to spherical polar
        coordinates.
        """
    def transform(self, matrix):
        """Transform the spherical coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).

        """
    def norm(self):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.  For
        spherical coordinates, this is just the absolute value of the radius.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.
        """
    def _scale_operation(self, op, *args): ...

class BaseSphericalDifferential(BaseDifferential):
    def _d_lon_coslat(self, base):
        """Convert longitude differential d_lon to d_lon_coslat.

        Parameters
        ----------
        base : instance of ``cls.base_representation``
            The base from which the latitude will be taken.
        """
    @classmethod
    def _get_d_lon(cls, d_lon_coslat, base):
        """Convert longitude differential d_lon_coslat to d_lon.

        Parameters
        ----------
        d_lon_coslat : `~astropy.units.Quantity`
            Longitude differential that includes ``cos(lat)``.
        base : instance of ``cls.base_representation``
            The base from which the latitude will be taken.
        """
    def _combine_operation(self, op, other, reverse: bool = False):
        """Combine two differentials, or a differential with a representation.

        If ``other`` is of the same differential type as ``self``, the
        components will simply be combined.  If both are different parts of
        a `~astropy.coordinates.SphericalDifferential` (e.g., a
        `~astropy.coordinates.UnitSphericalDifferential` and a
        `~astropy.coordinates.RadialDifferential`), they will combined
        appropriately.

        If ``other`` is a representation, it will be used as a base for which
        to evaluate the differential, and the result is a new representation.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            The other differential or representation.
        reverse : bool
            Whether the operands should be reversed (e.g., as we got here via
            ``self.__rsub__`` because ``self`` is a subclass of ``other``).
        """

class UnitSphericalDifferential(BaseSphericalDifferential):
    """Differential(s) of points on a unit sphere.

    Parameters
    ----------
    d_lon, d_lat : `~astropy.units.Quantity`
        The longitude and latitude of the differentials.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = UnitSphericalRepresentation
    def _dimensional_differential(cls): ...
    def __init__(self, d_lon, d_lat: Incomplete | None = None, copy: bool = True) -> None: ...
    @classmethod
    def from_cartesian(cls, other, base): ...
    def to_cartesian(self, base): ...
    def represent_as(self, other_class, base: Incomplete | None = None): ...
    @classmethod
    def from_representation(cls, representation, base: Incomplete | None = None): ...
    def transform(self, matrix, base, transformed_base):
        """Transform differential using a 3x3 matrix in a Cartesian basis.

        This returns a new differential and does not modify the original one.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 (or stack thereof) matrix, such as a rotation matrix.
        base : instance of ``cls.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        transformed_base : instance of ``cls.base_representation``
            Base relative to which the transformed differentials are defined.
            If the other class is a differential representation, the base will
            be converted to its ``base_representation``.
        """
    def _scale_operation(self, op, *args, scaled_base: bool = False): ...

class SphericalDifferential(BaseSphericalDifferential):
    """Differential(s) of points in 3D spherical coordinates.

    Parameters
    ----------
    d_lon, d_lat : `~astropy.units.Quantity`
        The differential longitude and latitude.
    d_distance : `~astropy.units.Quantity`
        The differential distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = SphericalRepresentation
    _unit_differential = UnitSphericalDifferential
    def __init__(self, d_lon, d_lat: Incomplete | None = None, d_distance: Incomplete | None = None, copy: bool = True) -> None: ...
    def represent_as(self, other_class, base: Incomplete | None = None): ...
    @classmethod
    def from_representation(cls, representation, base: Incomplete | None = None): ...
    def _scale_operation(self, op, *args, scaled_base: bool = False): ...

class BaseSphericalCosLatDifferential(BaseDifferential):
    """Differentials from points on a spherical base representation.

    With cos(lat) assumed to be included in the longitude differential.
    """
    @classmethod
    def _get_base_vectors(cls, base):
        """Get unit vectors and scale factors from (unit)spherical base.

        Parameters
        ----------
        base : instance of ``self.base_representation``
            The points for which the unit vectors and scale factors should be
            retrieved.

        Returns
        -------
        unit_vectors : dict of `~astropy.coordinates.CartesianRepresentation`
            In the directions of the coordinates of base.
        scale_factors : dict of `~astropy.units.Quantity`
            Scale factors for each of the coordinates.  The scale factor for
            longitude does not include the cos(lat) factor.

        Raises
        ------
        TypeError : if the base is not of the correct type
        """
    def _d_lon(self, base):
        """Convert longitude differential with cos(lat) to one without.

        Parameters
        ----------
        base : instance of ``cls.base_representation``
            The base from which the latitude will be taken.
        """
    @classmethod
    def _get_d_lon_coslat(cls, d_lon, base):
        """Convert longitude differential d_lon to d_lon_coslat.

        Parameters
        ----------
        d_lon : `~astropy.units.Quantity`
            Value of the longitude differential without ``cos(lat)``.
        base : instance of ``cls.base_representation``
            The base from which the latitude will be taken.
        """
    def _combine_operation(self, op, other, reverse: bool = False):
        """Combine two differentials, or a differential with a representation.

        If ``other`` is of the same differential type as ``self``, the
        components will simply be combined.  If both are different parts of
        a `~astropy.coordinates.SphericalDifferential` (e.g., a
        `~astropy.coordinates.UnitSphericalDifferential` and a
        `~astropy.coordinates.RadialDifferential`), they will combined
        appropriately.

        If ``other`` is a representation, it will be used as a base for which
        to evaluate the differential, and the result is a new representation.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            The other differential or representation.
        reverse : bool
            Whether the operands should be reversed (e.g., as we got here via
            ``self.__rsub__`` because ``self`` is a subclass of ``other``).
        """

class UnitSphericalCosLatDifferential(BaseSphericalCosLatDifferential):
    """Differential(s) of points on a unit sphere.

    Parameters
    ----------
    d_lon_coslat, d_lat : `~astropy.units.Quantity`
        The longitude and latitude of the differentials.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = UnitSphericalRepresentation
    attr_classes: Incomplete
    def _dimensional_differential(cls): ...
    def __init__(self, d_lon_coslat, d_lat: Incomplete | None = None, copy: bool = True) -> None: ...
    @classmethod
    def from_cartesian(cls, other, base): ...
    def to_cartesian(self, base): ...
    def represent_as(self, other_class, base: Incomplete | None = None): ...
    @classmethod
    def from_representation(cls, representation, base: Incomplete | None = None): ...
    def transform(self, matrix, base, transformed_base):
        """Transform differential using a 3x3 matrix in a Cartesian basis.

        This returns a new differential and does not modify the original one.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 (or stack thereof) matrix, such as a rotation matrix.
        base : instance of ``cls.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        transformed_base : instance of ``cls.base_representation``
            Base relative to which the transformed differentials are defined.
            If the other class is a differential representation, the base will
            be converted to its ``base_representation``.
        """
    def _scale_operation(self, op, *args, scaled_base: bool = False): ...

class SphericalCosLatDifferential(BaseSphericalCosLatDifferential):
    """Differential(s) of points in 3D spherical coordinates.

    Parameters
    ----------
    d_lon_coslat, d_lat : `~astropy.units.Quantity`
        The differential longitude (with cos(lat) included) and latitude.
    d_distance : `~astropy.units.Quantity`
        The differential distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = SphericalRepresentation
    _unit_differential = UnitSphericalCosLatDifferential
    attr_classes: Incomplete
    def __init__(self, d_lon_coslat, d_lat: Incomplete | None = None, d_distance: Incomplete | None = None, copy: bool = True) -> None: ...
    def represent_as(self, other_class, base: Incomplete | None = None): ...
    @classmethod
    def from_representation(cls, representation, base: Incomplete | None = None): ...
    def _scale_operation(self, op, *args, scaled_base: bool = False): ...

class RadialDifferential(BaseDifferential):
    """Differential(s) of radial distances.

    Parameters
    ----------
    d_distance : `~astropy.units.Quantity`
        The differential distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = RadialRepresentation
    def to_cartesian(self, base): ...
    def norm(self, base: Incomplete | None = None): ...
    @classmethod
    def from_cartesian(cls, other, base): ...
    @classmethod
    def from_representation(cls, representation, base: Incomplete | None = None): ...
    def _combine_operation(self, op, other, reverse: bool = False): ...

class PhysicsSphericalDifferential(BaseDifferential):
    """Differential(s) of 3D spherical coordinates using physics convention.

    Parameters
    ----------
    d_phi, d_theta : `~astropy.units.Quantity`
        The differential azimuth and inclination.
    d_r : `~astropy.units.Quantity`
        The differential radial distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = PhysicsSphericalRepresentation
    def __init__(self, d_phi, d_theta: Incomplete | None = None, d_r: Incomplete | None = None, copy: bool = True) -> None: ...
    def represent_as(self, other_class, base: Incomplete | None = None): ...
    @classmethod
    def from_representation(cls, representation, base: Incomplete | None = None): ...
    def _scale_operation(self, op, *args, scaled_base: bool = False): ...
