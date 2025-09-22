import abc
from _typeshed import Incomplete
from collections import OrderedDict

__all__ = ['RectangularFrame1D', 'Spine', 'BaseFrame', 'RectangularFrame', 'EllipticalFrame']

class Spine:
    """
    A single side of an axes.

    This does not need to be a straight line, but represents a 'side' when
    determining which part of the frame to put labels and ticks on.

    Parameters
    ----------
    parent_axes : `~astropy.visualization.wcsaxes.WCSAxes`
        The parent axes
    transform : `~matplotlib.transforms.Transform`
        The transform from data to world
    data_func : callable
        If not ``None``, it should be a function that returns the appropriate spine
        data when called with this object as the sole argument.  If ``None``, the
        spine data must be manually updated in ``update_spines()``.
    """
    parent_axes: Incomplete
    transform: Incomplete
    data_func: Incomplete
    _data: Incomplete
    _world: Incomplete
    def __init__(self, parent_axes, transform, *, data_func: Incomplete | None = None) -> None: ...
    @property
    def data(self): ...
    @data.setter
    def data(self, value) -> None: ...
    def _get_pixel(self): ...
    @property
    def pixel(self): ...
    @pixel.setter
    def pixel(self, value) -> None: ...
    @property
    def world(self): ...
    _pixel: Incomplete
    @world.setter
    def world(self, value) -> None: ...
    normal_angle: Incomplete
    def _update_normal(self) -> None: ...
    def _halfway_x_y_angle(self):
        """
        Return the x, y, normal_angle values halfway along the spine.
        """

class SpineXAligned(Spine):
    """
    A single side of an axes, aligned with the X data axis.

    This does not need to be a straight line, but represents a 'side' when
    determining which part of the frame to put labels and ticks on.
    """
    @property
    def data(self): ...
    _data: Incomplete
    _world: Incomplete
    @data.setter
    def data(self, value) -> None: ...

class BaseFrame(OrderedDict, metaclass=abc.ABCMeta):
    """
    Base class for frames, which are collections of
    :class:`~astropy.visualization.wcsaxes.frame.Spine` instances.
    """
    spine_class = Spine
    parent_axes: Incomplete
    _transform: Incomplete
    _linewidth: Incomplete
    _color: Incomplete
    _path: Incomplete
    def __init__(self, parent_axes, transform, path: Incomplete | None = None) -> None: ...
    @property
    def origin(self): ...
    @property
    def transform(self): ...
    @transform.setter
    def transform(self, value) -> None: ...
    def _update_patch_path(self) -> None: ...
    @property
    def patch(self): ...
    def draw(self, renderer) -> None: ...
    def sample(self, n_samples): ...
    def set_color(self, color) -> None:
        """
        Sets the color of the frame.

        Parameters
        ----------
        color : str
            The color of the frame.
        """
    def get_color(self): ...
    def set_linewidth(self, linewidth) -> None:
        """
        Sets the linewidth of the frame.

        Parameters
        ----------
        linewidth : float
            The linewidth of the frame in points.
        """
    def get_linewidth(self): ...
    def update_spines(self) -> None: ...

class RectangularFrame1D(BaseFrame):
    """
    A classic rectangular frame.
    """
    spine_names: str
    _spine_auto_position_order: str
    spine_class = SpineXAligned
    def update_spines(self) -> None: ...
    _path: Incomplete
    def _update_patch_path(self) -> None: ...
    def draw(self, renderer) -> None: ...

class RectangularFrame(BaseFrame):
    """
    A classic rectangular frame.
    """
    spine_names: str
    _spine_auto_position_order: str
    def update_spines(self) -> None: ...

class EllipticalFrame(BaseFrame):
    """
    An elliptical frame.
    """
    spine_names: str
    _spine_auto_position_order: str
    def update_spines(self) -> None: ...
    _path: Incomplete
    def _update_patch_path(self) -> None:
        """Override path patch to include only the outer ellipse,
        not the major and minor axes in the middle.
        """
    def draw(self, renderer) -> None:
        """Override to draw only the outer ellipse,
        not the major and minor axes in the middle.

        FIXME: we may want to add a general method to give the user control
        over which spines are drawn.
        """
