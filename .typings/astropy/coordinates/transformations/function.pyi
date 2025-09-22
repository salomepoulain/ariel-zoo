from _typeshed import Incomplete
from astropy.coordinates.transformations.base import CoordinateTransform

__all__ = ['FunctionTransform', 'FunctionTransformWithFiniteDifference']

class FunctionTransform(CoordinateTransform):
    """
    A coordinate transformation defined by a function that accepts a
    coordinate object and returns the transformed coordinate object.

    Parameters
    ----------
    func : callable
        The transformation function. Should have a call signature
        ``func(formcoord, toframe)``. Note that, unlike
        `CoordinateTransform.__call__`, ``toframe`` is assumed to be of type
        ``tosys`` for this function.
    fromsys : class
        The coordinate frame class to start from.
    tosys : class
        The coordinate frame class to transform into.
    priority : float or int
        The priority if this transform when finding the shortest
        coordinate transform path - large numbers are lower priorities.
    register_graph : `~astropy.coordinates.TransformGraph` or None
        A graph to register this transformation with on creation, or
        `None` to leave it unregistered.

    Raises
    ------
    TypeError
        If ``func`` is not callable.
    ValueError
        If ``func`` cannot accept two arguments.


    """
    func: Incomplete
    def __init__(self, func, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None) -> None: ...
    def __call__(self, fromcoord, toframe): ...

class FunctionTransformWithFiniteDifference(FunctionTransform):
    '''Transformation based on functions using finite difference for velocities.

    A coordinate transformation that works like a
    `~astropy.coordinates.FunctionTransform`, but computes velocity shifts
    based on the finite-difference relative to one of the frame attributes.
    Note that the transform function should *not* change the differential at
    all in this case, as any differentials will be overridden.

    When a differential is in the from coordinate, the finite difference
    calculation has two components. The first part is simple the existing
    differential, but re-orientation (using finite-difference techniques) to
    point in the direction the velocity vector has in the *new* frame. The
    second component is the "induced" velocity.  That is, the velocity
    intrinsic to the frame itself, estimated by shifting the frame using the
    ``finite_difference_frameattr_name`` frame attribute a small amount
    (``finite_difference_dt``) in time and re-calculating the position.

    Parameters
    ----------
    finite_difference_frameattr_name : str or None
        The name of the frame attribute on the frames to use for the finite
        difference.  Both the to and the from frame will be checked for this
        attribute, but only one needs to have it. If None, no velocity
        component induced from the frame itself will be included - only the
        re-orientation of any existing differential.
    finite_difference_dt : `~astropy.units.Quantity` [\'time\'] or callable
        If a quantity, this is the size of the differential used to do the
        finite difference.  If a callable, should accept
        ``(fromcoord, toframe)`` and return the ``dt`` value.
    symmetric_finite_difference : bool
        If True, the finite difference is computed as
        :math:`\\frac{x(t + \\Delta t / 2) - x(t + \\Delta t / 2)}{\\Delta t}`, or
        if False, :math:`\\frac{x(t + \\Delta t) - x(t)}{\\Delta t}`.  The latter
        case has slightly better performance (and more stable finite difference
        behavior).

    All other parameters are identical to the initializer for
    `~astropy.coordinates.FunctionTransform`.

    '''
    finite_difference_dt: Incomplete
    symmetric_finite_difference: Incomplete
    def __init__(self, func, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None, finite_difference_frameattr_name: str = 'obstime', finite_difference_dt=..., symmetric_finite_difference: bool = True) -> None: ...
    @property
    def finite_difference_frameattr_name(self): ...
    _diff_attr_in_fromsys: bool
    _diff_attr_in_tosys: Incomplete
    _finite_difference_frameattr_name: Incomplete
    @finite_difference_frameattr_name.setter
    def finite_difference_frameattr_name(self, value) -> None: ...
    def __call__(self, fromcoord, toframe): ...
