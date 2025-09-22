from .angles import Angle as Angle
from _typeshed import Incomplete

def matrix_transpose(matrix):
    """Transpose a matrix or stack of matrices by swapping the last two axes.

    This function mostly exists for readability; seeing ``.swapaxes(-2, -1)``
    it is not that obvious that one does a transpose.

    Note that one cannot use `~numpy.ndarray.T`, as this transposes all axes
    and thus does not work for stacks of matrices.  We also avoid
    ``np.matrix_transpose`` (new in numpy 2.0), since it is slower, as it
    first ensures the input is an array, while we ducktype, assuming the
    input has a ``.swapaxes`` method.

    """
def rotation_matrix(angle, axis: str = 'z', unit: Incomplete | None = None):
    """
    Generate matrices for rotation by some angle around some axis.

    Parameters
    ----------
    angle : angle-like
        The amount of rotation the matrices should represent.  Can be an array.
    axis : str or array-like
        Either ``'x'``, ``'y'``, ``'z'``, or a (x,y,z) specifying the axis to
        rotate about. If ``'x'``, ``'y'``, or ``'z'``, the rotation sense is
        counterclockwise looking down the + axis (e.g. positive rotations obey
        left-hand-rule).  If given as an array, the last dimension should be 3;
        it will be broadcast against ``angle``.
    unit : unit-like, optional
        If ``angle`` does not have associated units, they are in this
        unit.  If neither are provided, it is assumed to be degrees.

    Returns
    -------
    rmat : `numpy.matrix`
        A unitary rotation matrix.
    """
def angle_axis(matrix):
    """
    Angle of rotation and rotation axis for a given rotation matrix.

    Parameters
    ----------
    matrix : array-like
        A 3 x 3 unitary rotation matrix (or stack of matrices).

    Returns
    -------
    angle : `~astropy.coordinates.Angle`
        The angle of rotation.
    axis : array
        The (normalized) axis of rotation (with last dimension 3).
    """
def is_O3(matrix, atol: Incomplete | None = None):
    """Check whether a matrix is in the length-preserving group O(3).

    Parameters
    ----------
    matrix : (..., N, N) array-like
        Must have attribute ``.shape`` and method ``.swapaxes()`` and not error
        when using `~numpy.isclose`.
    atol : float, optional
        The allowed absolute difference.
        If `None` it defaults to 1e-15 or 5 * epsilon of the matrix's dtype, if floating.

        .. versionadded:: 5.3

    Returns
    -------
    is_o3 : bool or array of bool
        If the matrix has more than two axes, the O(3) check is performed on
        slices along the last two axes -- (M, N, N) => (M, ) bool array.

    Notes
    -----
    The orthogonal group O(3) preserves lengths, but is not guaranteed to keep
    orientations. Rotations and reflections are in this group.
    For more information, see https://en.wikipedia.org/wiki/Orthogonal_group
    """
def is_rotation(matrix, allow_improper: bool = False, atol: Incomplete | None = None):
    """Check whether a matrix is a rotation, proper or improper.

    Parameters
    ----------
    matrix : (..., N, N) array-like
        Must have attribute ``.shape`` and method ``.swapaxes()`` and not error
        when using `~numpy.isclose` and `~numpy.linalg.det`.
    allow_improper : bool, optional
        Whether to restrict check to the SO(3), the group of proper rotations,
        or also allow improper rotations (with determinant -1).
        The default (False) is only SO(3).
    atol : float, optional
        The allowed absolute difference.
        If `None` it defaults to 1e-15 or 5 * epsilon of the matrix's dtype, if floating.

        .. versionadded:: 5.3

    Returns
    -------
    isrot : bool or array of bool
        If the matrix has more than two axes, the checks are performed on
        slices along the last two axes -- (M, N, N) => (M, ) bool array.

    See Also
    --------
    astopy.coordinates.matrix_utilities.is_O3 :
        For the less restrictive check that a matrix is in the group O(3).

    Notes
    -----
    The group SO(3) is the rotation group. It is O(3), with determinant 1.
    Rotations with determinant -1 are improper rotations, combining both a
    rotation and a reflection.
    For more information, see https://en.wikipedia.org/wiki/Orthogonal_group

    """
