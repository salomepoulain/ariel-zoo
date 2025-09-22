from .base import BaseDifferential as BaseDifferential, BaseRepresentation as BaseRepresentation
from .cartesian import CartesianRepresentation as CartesianRepresentation
from .spherical import PhysicsSphericalRepresentation as PhysicsSphericalRepresentation, _spherical_op_funcs as _spherical_op_funcs
from _typeshed import Incomplete
from astropy.coordinates.angles import Angle as Angle
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED

class CylindricalRepresentation(BaseRepresentation):
    """
    Representation of points in 3D cylindrical coordinates.

    Parameters
    ----------
    rho : `~astropy.units.Quantity`
        The distance from the z axis to the point(s).

    phi : `~astropy.units.Quantity` or str
        The azimuth of the point(s), in angular units, which will be wrapped
        to an angle between 0 and 360 degrees. This can also be instances of
        `~astropy.coordinates.Angle`,

    z : `~astropy.units.Quantity`
        The z coordinate(s) of the point(s)

    differentials : dict, `~astropy.coordinates.CylindricalDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single
        `~astropy.coordinates.CylindricalDifferential` instance, or a dictionary of of differential
        instances with keys set to a string representation of the SI unit with
        which the differential (derivative) is taken. For example, for a
        velocity differential on a positional representation, the key would be
        ``'s'`` for seconds, indicating that the derivative is a time
        derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes: Incomplete
    def __init__(self, rho, phi: Incomplete | None = None, z: Incomplete | None = None, differentials: Incomplete | None = None, copy: bool = True) -> None: ...
    @property
    def rho(self):
        """
        The distance of the point(s) from the z-axis.
        """
    @property
    def phi(self):
        """
        The azimuth of the point(s).
        """
    @property
    def z(self):
        """
        The height of the point(s).
        """
    def unit_vectors(self): ...
    def scale_factors(self): ...
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates to cylindrical polar
        coordinates.
        """
    def to_cartesian(self):
        """
        Converts cylindrical polar coordinates to 3D rectangular cartesian
        coordinates.
        """
    def _scale_operation(self, op, *args): ...
    def represent_as(self, other_class, differential_class: Incomplete | None = None): ...

class CylindricalDifferential(BaseDifferential):
    """Differential(s) of points in cylindrical coordinates.

    Parameters
    ----------
    d_rho : `~astropy.units.Quantity` ['speed']
        The differential cylindrical radius.
    d_phi : `~astropy.units.Quantity` ['angular speed']
        The differential azimuth.
    d_z : `~astropy.units.Quantity` ['speed']
        The differential height.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = CylindricalRepresentation
    def __init__(self, d_rho, d_phi: Incomplete | None = None, d_z: Incomplete | None = None, copy: bool = True) -> None: ...
