import abc
from _typeshed import Incomplete
from matplotlib.transforms import Transform

__all__ = ['CurvedTransform', 'CoordinateTransform', 'World2PixelTransform', 'Pixel2WorldTransform']

class CurvedTransform(Transform, metaclass=abc.ABCMeta):
    """
    Abstract base class for non-affine curved transforms.
    """
    input_dims: int
    output_dims: int
    is_separable: bool
    def transform_path(self, path):
        """
        Transform a Matplotlib Path.

        Parameters
        ----------
        path : :class:`~matplotlib.path.Path`
            The path to transform

        Returns
        -------
        path : :class:`~matplotlib.path.Path`
            The resulting path
        """
    transform_path_non_affine = transform_path
    def transform(self, input) -> None: ...
    def inverted(self) -> None: ...

class CoordinateTransform(CurvedTransform):
    has_inverse: bool
    _input_system_name: Incomplete
    _output_system_name: Incomplete
    input_system: Incomplete
    output_system: Incomplete
    def __init__(self, input_system, output_system) -> None: ...
    @property
    def same_frames(self): ...
    _same_frames: Incomplete
    @same_frames.setter
    def same_frames(self, same_frames) -> None: ...
    def transform(self, input_coords):
        """
        Transform one set of coordinates to another.
        """
    transform_non_affine = transform
    def inverted(self):
        """
        Return the inverse of the transform.
        """

class World2PixelTransform(CurvedTransform, metaclass=abc.ABCMeta):
    """
    Base transformation from world to pixel coordinates.
    """
    has_inverse: bool
    frame_in: Incomplete
    @property
    @abc.abstractmethod
    def input_dims(self):
        """
        The number of input world dimensions.
        """
    @abc.abstractmethod
    def transform(self, world):
        """
        Transform world to pixel coordinates. You should pass in a NxM array
        where N is the number of points to transform, and M is the number of
        dimensions. This then returns the (x, y) pixel coordinates
        as a Nx2 array.
        """
    @abc.abstractmethod
    def inverted(self):
        """
        Return the inverse of the transform.
        """

class Pixel2WorldTransform(CurvedTransform, metaclass=abc.ABCMeta):
    """
    Base transformation from pixel to world coordinates.
    """
    has_inverse: bool
    frame_out: Incomplete
    @property
    @abc.abstractmethod
    def output_dims(self):
        """
        The number of output world dimensions.
        """
    @abc.abstractmethod
    def transform(self, pixel):
        """
        Transform pixel to world coordinates. You should pass in a Nx2 array
        of (x, y) pixel coordinates to transform to world coordinates. This
        will then return an NxM array where M is the number of dimensions.
        """
    @abc.abstractmethod
    def inverted(self):
        """
        Return the inverse of the transform.
        """
