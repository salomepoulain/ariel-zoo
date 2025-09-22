from .core import Model
from _typeshed import Incomplete

__all__ = ['RotateCelestial2Native', 'RotateNative2Celestial', 'Rotation2D', 'EulerAngleRotation', 'RotationSequence3D', 'SphericalRotationSequence']

class RotationSequence3D(Model):
    """
    Perform a series of rotations about different axis in 3D space.

    Positive angles represent a counter-clockwise rotation.

    Parameters
    ----------
    angles : array-like
        Angles of rotation in deg in the order of axes_order.
    axes_order : str
        A sequence of 'x', 'y', 'z' corresponding to axis of rotation.

    Examples
    --------
    >>> model = RotationSequence3D([1.1, 2.1, 3.1, 4.1], axes_order='xyzx')

    """
    standard_broadcasting: bool
    _separable: bool
    n_inputs: int
    n_outputs: int
    angles: Incomplete
    axes: Incomplete
    axes_order: Incomplete
    _inputs: Incomplete
    _outputs: Incomplete
    def __init__(self, angles, axes_order, name: Incomplete | None = None) -> None: ...
    @property
    def inverse(self):
        """Inverse rotation."""
    def evaluate(self, x, y, z, angles):
        """
        Apply the rotation to a set of 3D Cartesian coordinates.
        """

class SphericalRotationSequence(RotationSequence3D):
    """
    Perform a sequence of rotations about arbitrary number of axes
    in spherical coordinates.

    Parameters
    ----------
    angles : list
        A sequence of angles (in deg).
    axes_order : str
        A sequence of characters ('x', 'y', or 'z') corresponding to the
        axis of rotation and matching the order in ``angles``.

    """
    _n_inputs: int
    _n_outputs: int
    _inputs: Incomplete
    _outputs: Incomplete
    def __init__(self, angles, axes_order, name: Incomplete | None = None, **kwargs) -> None: ...
    @property
    def n_inputs(self): ...
    @property
    def n_outputs(self): ...
    def evaluate(self, lon, lat, angles): ...

class _EulerRotation:
    """
    Base class which does the actual computation.
    """
    _separable: bool
    def evaluate(self, alpha, delta, phi, theta, psi, axes_order): ...
    _input_units_strict: bool
    _input_units_allow_dimensionless: bool
    @property
    def input_units(self):
        """Input units."""
    @property
    def return_units(self):
        """Output units."""

class EulerAngleRotation(_EulerRotation, Model):
    '''
    Implements Euler angle intrinsic rotations.

    Rotates one coordinate system into another (fixed) coordinate system.
    All coordinate systems are right-handed. The sign of the angles is
    determined by the right-hand rule..

    Parameters
    ----------
    phi, theta, psi : float or `~astropy.units.Quantity` [\'angle\']
        "proper" Euler angles in deg.
        If floats, they should be in deg.
    axes_order : str
        A 3 character string, a combination of \'x\', \'y\' and \'z\',
        where each character denotes an axis in 3D space.
    '''
    n_inputs: int
    n_outputs: int
    phi: Incomplete
    theta: Incomplete
    psi: Incomplete
    axes: Incomplete
    axes_order: Incomplete
    _inputs: Incomplete
    _outputs: Incomplete
    def __init__(self, phi, theta, psi, axes_order, **kwargs) -> None: ...
    @property
    def inverse(self): ...
    def evaluate(self, alpha, delta, phi, theta, psi): ...

class _SkyRotation(_EulerRotation, Model):
    """
    Base class for RotateNative2Celestial and RotateCelestial2Native.
    """
    lon: Incomplete
    lat: Incomplete
    lon_pole: Incomplete
    axes_order: str
    def __init__(self, lon, lat, lon_pole, **kwargs) -> None: ...
    def _evaluate(self, phi, theta, lon, lat, lon_pole): ...

class RotateNative2Celestial(_SkyRotation):
    """
    Transform from Native to Celestial Spherical Coordinates.

    Parameters
    ----------
    lon : float or `~astropy.units.Quantity` ['angle']
        Celestial longitude of the fiducial point.
    lat : float or `~astropy.units.Quantity` ['angle']
        Celestial latitude of the fiducial point.
    lon_pole : float or `~astropy.units.Quantity` ['angle']
        Longitude of the celestial pole in the native system.

    Notes
    -----
    If ``lon``, ``lat`` and ``lon_pole`` are numerical values they
    should be in units of deg. Inputs are angles on the native sphere.
    Outputs are angles on the celestial sphere.
    """
    n_inputs: int
    n_outputs: int
    @property
    def input_units(self):
        """Input units."""
    @property
    def return_units(self):
        """Output units."""
    inputs: Incomplete
    outputs: Incomplete
    def __init__(self, lon, lat, lon_pole, **kwargs) -> None: ...
    def evaluate(self, phi_N, theta_N, lon, lat, lon_pole):
        """
        Parameters
        ----------
        phi_N, theta_N : float or `~astropy.units.Quantity` ['angle']
            Angles in the Native coordinate system.
            it is assumed that numerical only inputs are in degrees.
            If float, assumed in degrees.
        lon, lat, lon_pole : float or `~astropy.units.Quantity` ['angle']
            Parameter values when the model was initialized.
            If float, assumed in degrees.

        Returns
        -------
        alpha_C, delta_C : float or `~astropy.units.Quantity` ['angle']
            Angles on the Celestial sphere.
            If float, in degrees.
        """
    @property
    def inverse(self): ...

class RotateCelestial2Native(_SkyRotation):
    """
    Transform from Celestial to Native Spherical Coordinates.

    Parameters
    ----------
    lon : float or `~astropy.units.Quantity` ['angle']
        Celestial longitude of the fiducial point.
    lat : float or `~astropy.units.Quantity` ['angle']
        Celestial latitude of the fiducial point.
    lon_pole : float or `~astropy.units.Quantity` ['angle']
        Longitude of the celestial pole in the native system.

    Notes
    -----
    If ``lon``, ``lat`` and ``lon_pole`` are numerical values they should be
    in units of deg. Inputs are angles on the celestial sphere.
    Outputs are angles on the native sphere.
    """
    n_inputs: int
    n_outputs: int
    @property
    def input_units(self):
        """Input units."""
    @property
    def return_units(self):
        """Output units."""
    inputs: Incomplete
    outputs: Incomplete
    def __init__(self, lon, lat, lon_pole, **kwargs) -> None: ...
    def evaluate(self, alpha_C, delta_C, lon, lat, lon_pole):
        """
        Parameters
        ----------
        alpha_C, delta_C : float or `~astropy.units.Quantity` ['angle']
            Angles in the Celestial coordinate frame.
            If float, assumed in degrees.
        lon, lat, lon_pole : float or `~astropy.units.Quantity` ['angle']
            Parameter values when the model was initialized.
            If float, assumed in degrees.

        Returns
        -------
        phi_N, theta_N : float or `~astropy.units.Quantity` ['angle']
            Angles on the Native sphere.
            If float, in degrees.

        """
    @property
    def inverse(self): ...

class Rotation2D(Model):
    """
    Perform a 2D rotation given an angle.

    Positive angles represent a counter-clockwise rotation and vice-versa.

    Parameters
    ----------
    angle : float or `~astropy.units.Quantity` ['angle']
        Angle of rotation (if float it should be in deg).
    """
    n_inputs: int
    n_outputs: int
    _separable: bool
    angle: Incomplete
    _inputs: Incomplete
    _outputs: Incomplete
    def __init__(self, angle=..., **kwargs) -> None: ...
    @property
    def inverse(self):
        """Inverse rotation."""
    @classmethod
    def evaluate(cls, x, y, angle):
        """
        Rotate (x, y) about ``angle``.

        Parameters
        ----------
        x, y : array-like
            Input quantities
        angle : float or `~astropy.units.Quantity` ['angle']
            Angle of rotations.
            If float, assumed in degrees.

        """
    @staticmethod
    def _compute_matrix(angle): ...
