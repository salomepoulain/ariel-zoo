import abc
from _typeshed import Incomplete
from abc import abstractmethod
from astropy.coordinates.transformations.base import CoordinateTransform

__all__ = ['BaseAffineTransform', 'AffineTransform', 'StaticMatrixTransform', 'DynamicMatrixTransform']

class BaseAffineTransform(CoordinateTransform, metaclass=abc.ABCMeta):
    """Base class for common functionality between the ``AffineTransform``-type
    subclasses.

    This base class is needed because `~astropy.coordinates.AffineTransform`
    and the matrix transform classes share the ``__call__()`` method, but
    differ in how they generate the affine parameters.
    `~astropy.coordinates.StaticMatrixTransform` passes in a matrix stored as a
    class attribute, and both of the matrix transforms pass in ``None`` for the
    offset. Hence, user subclasses would likely want to subclass this (rather
    than `~astropy.coordinates.AffineTransform`) if they want to provide
    alternative transformations using this machinery.

    """
    def _apply_transform(self, fromcoord, matrix, offset): ...
    def __call__(self, fromcoord, toframe): ...
    @abstractmethod
    def _affine_params(self, fromcoord, toframe): ...

class AffineTransform(BaseAffineTransform):
    """
    A coordinate transformation specified as a function that yields a 3 x 3
    cartesian transformation matrix and a tuple of displacement vectors.

    See `~astropy.coordinates.Galactocentric` for
    an example.

    Parameters
    ----------
    transform_func : callable
        A callable that has the signature ``transform_func(fromcoord, toframe)``
        and returns: a (3, 3) matrix that operates on ``fromcoord`` in a
        Cartesian representation, and a ``CartesianRepresentation`` with
        (optionally) an attached velocity ``CartesianDifferential`` to represent
        a translation and offset in velocity to apply after the matrix
        operation.
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
        If ``transform_func`` is not callable

    """
    transform_func: Incomplete
    def __init__(self, transform_func, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None) -> None: ...
    def _affine_params(self, fromcoord, toframe): ...

class StaticMatrixTransform(BaseAffineTransform):
    """
    A coordinate transformation defined as a 3 x 3 cartesian
    transformation matrix.

    This is distinct from DynamicMatrixTransform in that this kind of matrix is
    independent of frame attributes.  That is, it depends *only* on the class of
    the frame.

    Parameters
    ----------
    matrix : array-like or callable
        A 3 x 3 matrix for transforming 3-vectors. In most cases will
        be unitary (although this is not strictly required). If a callable,
        will be called *with no arguments* to get the matrix.
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
    ValueError
        If the matrix is not 3 x 3

    """
    matrix: Incomplete
    def __init__(self, matrix, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None) -> None: ...
    def _affine_params(self, fromcoord, toframe): ...

class DynamicMatrixTransform(BaseAffineTransform):
    """
    A coordinate transformation specified as a function that yields a
    3 x 3 cartesian transformation matrix.

    This is similar to, but distinct from StaticMatrixTransform, in that the
    matrix for this class might depend on frame attributes.

    Parameters
    ----------
    matrix_func : callable
        A callable that has the signature ``matrix_func(fromcoord, toframe)`` and
        returns a 3 x 3 matrix that converts ``fromcoord`` in a cartesian
        representation to the new coordinate system.
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
        If ``matrix_func`` is not callable

    """
    matrix_func: Incomplete
    def __init__(self, matrix_func, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None) -> None: ...
    def _affine_params(self, fromcoord, toframe): ...
