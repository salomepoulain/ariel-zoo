from _typeshed import Incomplete
from astropy.coordinates.transformations.base import CoordinateTransform

__all__ = ['CompositeTransform']

class CompositeTransform(CoordinateTransform):
    """
    A transformation constructed by combining together a series of single-step
    transformations.

    Note that the intermediate frame objects are constructed using any frame
    attributes in ``toframe`` or ``fromframe`` that overlap with the intermediate
    frame (``toframe`` favored over ``fromframe`` if there's a conflict).  Any frame
    attributes that are not present use the defaults.

    Parameters
    ----------
    transforms : sequence of `~astropy.coordinates.CoordinateTransform` object
        The sequence of transformations to apply.
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
    collapse_static_mats : bool
        If `True`, consecutive `~astropy.coordinates.StaticMatrixTransform`
        will be collapsed into a single transformation to speed up the
        calculation.

    """
    transforms: Incomplete
    def __init__(self, transforms, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None, collapse_static_mats: bool = True) -> None: ...
    def _combine_statics(self, transforms):
        """
        Combines together sequences of StaticMatrixTransform's into a single
        transform and returns it.
        """
    def __call__(self, fromcoord, toframe): ...
    def _as_single_transform(self):
        """
        Return an encapsulated version of the composite transform so that it appears to
        be a single transform.

        The returned transform internally calls the constituent transforms.  If all of
        the transforms are affine, the merged transform is
        `~astropy.coordinates.DynamicMatrixTransform` (if there are no
        origin shifts) or `~astropy.coordinates.AffineTransform`
        (otherwise).  If at least one of the transforms is not affine, the merged
        transform is
        `~astropy.coordinates.FunctionTransformWithFiniteDifference`.
        """
