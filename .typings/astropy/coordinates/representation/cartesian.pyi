from .base import BaseDifferential as BaseDifferential, BaseRepresentation as BaseRepresentation
from _typeshed import Incomplete

class CartesianRepresentation(BaseRepresentation):
    """
    Representation of points in 3D cartesian coordinates.

    Parameters
    ----------
    x, y, z : `~astropy.units.Quantity` or array
        The x, y, and z coordinates of the point(s). If ``x``, ``y``, and ``z``
        have different shapes, they should be broadcastable. If not quantity,
        ``unit`` should be set.  If only ``x`` is given, it is assumed that it
        contains an array with the 3 coordinates stored along ``xyz_axis``.
    unit : unit-like
        If given, the coordinates will be converted to this unit (or taken to
        be in this unit if not given.
    xyz_axis : int, optional
        The axis along which the coordinates are stored when a single array is
        provided rather than distinct ``x``, ``y``, and ``z`` (default: 0).

    differentials : dict, `~astropy.coordinates.CartesianDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single
        `~astropy.coordinates.CartesianDifferential` instance, or a dictionary of
        `~astropy.coordinates.CartesianDifferential` s with keys set to a string representation of
        the SI unit with which the differential (derivative) is taken. For
        example, for a velocity differential on a positional representation, the
        key would be ``'s'`` for seconds, indicating that the derivative is a
        time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes: Incomplete
    _xyz: Incomplete
    _xyz_axis: Incomplete
    _differentials: Incomplete
    def __init__(self, x, y: Incomplete | None = None, z: Incomplete | None = None, unit: Incomplete | None = None, xyz_axis: Incomplete | None = None, differentials: Incomplete | None = None, copy: bool = True) -> None: ...
    def unit_vectors(self): ...
    def scale_factors(self): ...
    def get_xyz(self, xyz_axis: int = 0):
        """Return a vector array of the x, y, and z coordinates.

        Parameters
        ----------
        xyz_axis : int, optional
            The axis in the final array along which the x, y, z components
            should be stored (default: 0).

        Returns
        -------
        xyz : `~astropy.units.Quantity`
            With dimension 3 along ``xyz_axis``.  Note that, if possible,
            this will be a view.
        """
    xyz: Incomplete
    @classmethod
    def from_cartesian(cls, other): ...
    def to_cartesian(self): ...
    def transform(self, matrix):
        """
        Transform the cartesian coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : ndarray
            A 3x3 transformation matrix, such as a rotation matrix.

        Examples
        --------
        We can start off by creating a cartesian representation object:

            >>> from astropy import units as u
            >>> from astropy.coordinates import CartesianRepresentation
            >>> rep = CartesianRepresentation([1, 2] * u.pc,
            ...                               [2, 3] * u.pc,
            ...                               [3, 4] * u.pc)

        We now create a rotation matrix around the z axis:

            >>> from astropy.coordinates.matrix_utilities import rotation_matrix
            >>> rotation = rotation_matrix(30 * u.deg, axis='z')

        Finally, we can apply this transformation:

            >>> rep_new = rep.transform(rotation)
            >>> rep_new.xyz  # doctest: +FLOAT_CMP
            <Quantity [[ 1.8660254 , 3.23205081],
                       [ 1.23205081, 1.59807621],
                       [ 3.        , 4.        ]] pc>
        """
    def _combine_operation(self, op, other, reverse: bool = False): ...
    def norm(self):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.

        Note that any associated differentials will be dropped during this
        operation.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.
        """
    def mean(self, *args, **kwargs):
        """Vector mean.

        Returns a new CartesianRepresentation instance with the means of the
        x, y, and z components.

        Refer to `~numpy.mean` for full documentation of the arguments, noting
        that ``axis`` is the entry in the ``shape`` of the representation, and
        that the ``out`` argument cannot be used.
        """
    def sum(self, *args, **kwargs):
        """Vector sum.

        Returns a new CartesianRepresentation instance with the sums of the
        x, y, and z components.

        Refer to `~numpy.sum` for full documentation of the arguments, noting
        that ``axis`` is the entry in the ``shape`` of the representation, and
        that the ``out`` argument cannot be used.
        """
    def dot(self, other):
        """Dot product of two representations.

        Note that any associated differentials will be dropped during this
        operation.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            If not already cartesian, it is converted.

        Returns
        -------
        dot_product : `~astropy.units.Quantity`
            The sum of the product of the x, y, and z components of ``self``
            and ``other``.
        """
    def cross(self, other):
        """Cross product of two representations.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            If not already cartesian, it is converted.

        Returns
        -------
        cross_product : `~astropy.coordinates.CartesianRepresentation`
            With vectors perpendicular to both ``self`` and ``other``.
        """

class CartesianDifferential(BaseDifferential):
    """Differentials in of points in 3D cartesian coordinates.

    Parameters
    ----------
    d_x, d_y, d_z : `~astropy.units.Quantity` or array
        The x, y, and z coordinates of the differentials. If ``d_x``, ``d_y``,
        and ``d_z`` have different shapes, they should be broadcastable. If not
        quantities, ``unit`` should be set.  If only ``d_x`` is given, it is
        assumed that it contains an array with the 3 coordinates stored along
        ``xyz_axis``.
    unit : `~astropy.units.Unit` or str
        If given, the differentials will be converted to this unit (or taken to
        be in this unit if not given.
    xyz_axis : int, optional
        The axis along which the coordinates are stored when a single array is
        provided instead of distinct ``d_x``, ``d_y``, and ``d_z`` (default: 0).
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = CartesianRepresentation
    _d_xyz: Incomplete
    _xyz_axis: Incomplete
    def __init__(self, d_x, d_y: Incomplete | None = None, d_z: Incomplete | None = None, unit: Incomplete | None = None, xyz_axis: Incomplete | None = None, copy: bool = True) -> None: ...
    def to_cartesian(self, base: Incomplete | None = None): ...
    @classmethod
    def from_cartesian(cls, other, base: Incomplete | None = None): ...
    def transform(self, matrix, base: Incomplete | None = None, transformed_base: Incomplete | None = None):
        """Transform differentials using a 3x3 matrix in a Cartesian basis.

        This returns a new differential and does not modify the original one.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 (or stack thereof) matrix, such as a rotation matrix.
        base, transformed_base : `~astropy.coordinates.CartesianRepresentation` or None, optional
            Not used in the Cartesian transformation.
        """
    def get_d_xyz(self, xyz_axis: int = 0):
        """Return a vector array of the x, y, and z coordinates.

        Parameters
        ----------
        xyz_axis : int, optional
            The axis in the final array along which the x, y, z components
            should be stored (default: 0).

        Returns
        -------
        d_xyz : `~astropy.units.Quantity`
            With dimension 3 along ``xyz_axis``.  Note that, if possible,
            this will be a view.
        """
    d_xyz: Incomplete
