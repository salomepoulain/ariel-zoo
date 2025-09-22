from .frame import RectangularFrame as RectangularFrame
from _typeshed import Incomplete
from astropy.utils.decorators import deprecated_renamed_argument as deprecated_renamed_argument
from astropy.utils.exceptions import AstropyDeprecationWarning as AstropyDeprecationWarning
from matplotlib.text import Text

def sort_using(X, Y): ...

class TickLabels(Text):
    _frame: Incomplete
    _exclude_overlapping: bool
    _simplify: bool
    _axis_bboxes: Incomplete
    _stale: bool
    def __init__(self, frame, *args, **kwargs) -> None: ...
    world: Incomplete
    data: Incomplete
    angle: Incomplete
    text: Incomplete
    disp: Incomplete
    def clear(self) -> None: ...
    def add(self, axis: Incomplete | None = None, world: Incomplete | None = None, pixel: Incomplete | None = None, angle: Incomplete | None = None, text: Incomplete | None = None, axis_displacement: Incomplete | None = None, data: Incomplete | None = None) -> None:
        """
        Add a label.

        Parameters
        ----------
        axis : str
            Axis to add label to.
        world : Quantity
            Coordinate value along this axis.
        pixel : [float, float]
            Pixel coordinates of the label. Deprecated and no longer used.
        angle : float
            Angle of the label.
        text : str
            Label text.
        axis_displacement : float
            Displacement from axis.
        data : [float, float]
            Data coordinates of the label.
        """
    def sort(self) -> None:
        """
        Sort by axis displacement, which allows us to figure out which parts
        of labels to not repeat.
        """
    def simplify_labels(self) -> None:
        """
        Figure out which parts of labels can be dropped to avoid repetition.
        """
    _pad: Incomplete
    def set_pad(self, value) -> None: ...
    def get_pad(self): ...
    _visible_axes: Incomplete
    def set_visible_axes(self, visible_axes) -> None: ...
    def get_visible_axes(self): ...
    def set_exclude_overlapping(self, exclude_overlapping) -> None: ...
    def set_simplify(self, simplify) -> None: ...
    xy: Incomplete
    ha: Incomplete
    va: Incomplete
    def _set_xy_alignments(self, renderer) -> None:
        """
        Compute and set the x, y positions and the horizontal/vertical alignment of
        each label.
        """
    def _get_bb(self, axis, i, renderer):
        """
        Get the bounding box of an individual label. n.b. _set_xy_alignment()
        must be called before this method.
        """
    @property
    def _all_bboxes(self): ...
    _existing_bboxes: Incomplete
    def _set_existing_bboxes(self, bboxes) -> None: ...
    def draw(self, renderer, bboxes: Incomplete | None = None, ticklabels_bbox: Incomplete | None = None, tick_out_size: Incomplete | None = None) -> None: ...
