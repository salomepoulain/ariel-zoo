from _typeshed import Incomplete

__all__ = ['CoordinateHelper']

class CoordinateHelper:
    """
    Helper class to control one of the coordinates in the
    :class:`~astropy.visualization.wcsaxes.WCSAxes`.

    Parameters
    ----------
    parent_axes : :class:`~astropy.visualization.wcsaxes.WCSAxes`
        The axes the coordinate helper belongs to.
    parent_map : :class:`~astropy.visualization.wcsaxes.CoordinatesMap`
        The :class:`~astropy.visualization.wcsaxes.CoordinatesMap` object this
        coordinate belongs to.
    transform : `~matplotlib.transforms.Transform`
        The transform corresponding to this coordinate system.
    coord_index : int
        The index of this coordinate in the
        :class:`~astropy.visualization.wcsaxes.CoordinatesMap`.
    coord_type : {'longitude', 'latitude', 'scalar'}
        The type of this coordinate, which is used to determine the wrapping and
        boundary behavior of coordinates. Longitudes wrap at ``coord_wrap``,
        latitudes have to be in the range -90 to 90, and scalars are unbounded
        and do not wrap.
    coord_unit : `~astropy.units.Unit`
        The unit that this coordinate is in given the output of transform.
    format_unit : `~astropy.units.Unit`, optional
        The unit to use to display the coordinates.
    coord_wrap : `astropy.units.Quantity`
        The angle at which the longitude wraps (defaults to 360 degrees).
    frame : `~astropy.visualization.wcsaxes.frame.BaseFrame`
        The frame of the :class:`~astropy.visualization.wcsaxes.WCSAxes`.
    default_label : str, optional
        The axis label to show by default if none is set later.
    """
    _parent_axes: Incomplete
    _parent_map: Incomplete
    _transform: Incomplete
    _coord_index: Incomplete
    _coord_unit: Incomplete
    _format_unit: Incomplete
    _frame: Incomplete
    _default_label: Incomplete
    _auto_axislabel: bool
    _axislabel_set: bool
    _custom_formatter: Incomplete
    dpi_transform: Incomplete
    offset_transform: Incomplete
    _ticks: Incomplete
    _ticklabels: Incomplete
    _minor_frequency: int
    _axislabels: Incomplete
    _grid_lines: Incomplete
    _grid_lines_kwargs: Incomplete
    def __init__(self, parent_axes: Incomplete | None = None, parent_map: Incomplete | None = None, transform: Incomplete | None = None, coord_index: Incomplete | None = None, coord_type: str = 'scalar', coord_unit: Incomplete | None = None, coord_wrap: Incomplete | None = None, frame: Incomplete | None = None, format_unit: Incomplete | None = None, default_label: Incomplete | None = None) -> None: ...
    @property
    def parent_axes(self):
        """
        The axes the coordinate helper belongs to.
        """
    @parent_axes.setter
    def parent_axes(self, value) -> None: ...
    @property
    def parent_map(self):
        """
        The :class:`~astropy.visualization.wcsaxes.CoordinatesMap` object this
        coordinate belongs to.
        """
    @parent_map.setter
    def parent_map(self, value) -> None: ...
    @property
    def transform(self):
        """
        The transform corresponding to this coordinate system.
        """
    @transform.setter
    def transform(self, value) -> None: ...
    @property
    def coord_index(self):
        """
        The index of this coordinate in the
        :class:`~astropy.visualization.wcsaxes.CoordinatesMap`.
        """
    @coord_index.setter
    def coord_index(self, value) -> None: ...
    @property
    def coord_type(self):
        """
        The type of this coordinate (e.g., ``'longitude'``)
        """
    _coord_type: Incomplete
    @coord_type.setter
    def coord_type(self, value) -> None: ...
    @property
    def coord_unit(self):
        """
        The unit that this coordinate is in given the output of transform.
        """
    @coord_unit.setter
    def coord_unit(self, value) -> None: ...
    @property
    def coord_wrap(self):
        """
        The angle at which the longitude wraps (defaults to 360 degrees).
        """
    _coord_wrap: Incomplete
    @coord_wrap.setter
    def coord_wrap(self, value) -> None: ...
    @property
    def frame(self):
        """
        The frame of the :class:`~astropy.visualization.wcsaxes.WCSAxes`.
        """
    @frame.setter
    def frame(self, value) -> None: ...
    @property
    def default_label(self):
        """
        The axis label to show by default if none is set later.
        """
    @default_label.setter
    def default_label(self, value) -> None: ...
    @property
    def ticks(self): ...
    @ticks.setter
    def ticks(self, value) -> None: ...
    @property
    def ticklabels(self): ...
    @ticklabels.setter
    def ticklabels(self, value) -> None: ...
    @property
    def axislabels(self): ...
    @axislabels.setter
    def axislabels(self, value) -> None: ...
    _grid_type: Incomplete
    def grid(self, draw_grid: bool = True, grid_type: Incomplete | None = None, **kwargs) -> None:
        """
        Plot grid lines for this coordinate.

        Standard matplotlib appearance options (color, alpha, etc.) can be
        passed as keyword arguments.

        Parameters
        ----------
        draw_grid : bool
            Whether to show the gridlines
        grid_type : {'lines', 'contours'}
            Whether to plot the contours by determining the grid lines in
            world coordinates and then plotting them in world coordinates
            (``'lines'``) or by determining the world coordinates at many
            positions in the image and then drawing contours
            (``'contours'``). The first is recommended for 2-d images, while
            for 3-d (or higher dimensional) cubes, the ``'contours'`` option
            is recommended. By default, 'lines' is used if the transform has
            an inverse, otherwise 'contours' is used.
        """
    _coord_scale_to_deg: Incomplete
    _formatter_locator: Incomplete
    def set_coord_type(self, coord_type, coord_wrap: Incomplete | None = None) -> None:
        """
        Set the coordinate type for the axis.

        Parameters
        ----------
        coord_type : str
            One of 'longitude', 'latitude' or 'scalar'
        coord_wrap : `~astropy.units.Quantity`, optional
            The value to wrap at for angular coordinates.
        """
    def set_major_formatter(self, formatter) -> None:
        """
        Set the format string to use for the major tick labels.

        See :ref:`tick_label_format` for accepted format strings and examples.

        Parameters
        ----------
        formatter : str or callable
            The format string to use, or a callable (for advanced use cases).
            If specified as a callable, this should take a
            `~astropy.units.Quantity` (which could be scalar or array) of tick
            world coordinates as well as an optional ``spacing`` keyword
            argument, which gives (also as a `~astropy.units.Quantity`) the
            spacing between ticks, and returns an iterable of strings
            containing the labels.
        """
    def format_coord(self, value, format: str = 'auto'):
        """
        Given the value of a coordinate, will format it according to the
        format of the formatter_locator.

        Parameters
        ----------
        value : float
            The value to format
        format : {'auto', 'ascii', 'latex'}, optional
            The format to use - by default the formatting will be adjusted
            depending on whether Matplotlib is using LaTeX or MathTex. To
            get plain ASCII strings, use format='ascii'.
        """
    def set_separator(self, separator) -> None:
        """
        Set the separator to use for the angle major tick labels.

        Parameters
        ----------
        separator : str or tuple or None
            The separator between numbers in sexagesimal representation. Can be
            either a string or a tuple (or `None` for default).
        """
    def set_format_unit(self, unit, decimal: Incomplete | None = None, show_decimal_unit: bool = True) -> None:
        """
        Set the unit for the major tick labels.

        Parameters
        ----------
        unit : class:`~astropy.units.Unit`
            The unit to which the tick labels should be converted to.
        decimal : bool, optional
            Whether to use decimal formatting. By default this is `False`
            for degrees or hours (which therefore use sexagesimal formatting)
            and `True` for all other units.
        show_decimal_unit : bool, optional
            Whether to include units when in decimal mode.
        """
    def get_format_unit(self):
        """
        Get the unit for the major tick labels.
        """
    def set_ticks(self, values: Incomplete | None = None, spacing: Incomplete | None = None, number: Incomplete | None = None, size: Incomplete | None = None, width: Incomplete | None = None, color: Incomplete | None = None, alpha: Incomplete | None = None, direction: Incomplete | None = None, exclude_overlapping: Incomplete | None = None) -> None:
        """
        Set the location and properties of the ticks.

        At most one of the options from ``values``, ``spacing``, or
        ``number`` can be specified.

        Parameters
        ----------
        values : iterable, optional
            The coordinate values at which to show the ticks.
        spacing : float, optional
            The spacing between ticks.
        number : float, optional
            The approximate number of ticks shown.
        size : float, optional
            The length of the ticks in points
        color : str or tuple, optional
            A valid Matplotlib color for the ticks
        alpha : float, optional
            The alpha value (transparency) for the ticks.
        direction : {'in','out'}, optional
            Whether the ticks should point inwards or outwards.
        """
    def set_ticks_position(self, position) -> None:
        """
        Set where ticks should appear.

        Parameters
        ----------
        position : str or list
            The axes on which the ticks for this coordinate should appear.
            Should be a sequence containing zero or more of ``'b'``, ``'t'``,
            ``'l'``, ``'r'``. For example, ``'lb'`` will lead the ticks to be
            shown on the left and bottom axis. In addition, if ``'#'`` is
            included in the sequence, the position will be considered dynamic and
            will be updated at draw-time in order to show the ticks on the same
            axes as the tick labels are shown.
        """
    def get_ticks_position(self):
        """
        Get where tick labels will appear.
        """
    def set_ticks_visible(self, visible) -> None:
        """
        Set whether ticks are visible or not.

        Parameters
        ----------
        visible : bool
            The visibility of ticks. Setting as ``False`` will hide ticks
            along this coordinate.
        """
    def set_ticklabel(self, color: Incomplete | None = None, size: Incomplete | None = None, pad: Incomplete | None = None, exclude_overlapping: Incomplete | None = None, *, simplify: bool = True, **kwargs) -> None:
        """
        Set the visual properties for the tick labels.

        Parameters
        ----------
        size : float, optional
            The size of the ticks labels in points
        color : str or tuple, optional
            A valid Matplotlib color for the tick labels
        pad : float, optional
            Distance in points between tick and label.
        exclude_overlapping : bool, optional
            Whether to exclude tick labels that overlap over each other.
        simplify : bool, optional
            Whether to remove repeated parts of tick labels.
        **kwargs
            Other keyword arguments are passed to :class:`matplotlib.text.Text`.
        """
    def set_ticklabel_position(self, position) -> None:
        """
        Set where tick labels should appear.

        Parameters
        ----------
        position : str or list
            The axes on which the tick labels for this coordinate should
            appear. Should be a sequence containing zero or more of ``'b'``,
            ``'t'``, ``'l'``, ``'r'``. For example, ``'lb'`` will lead the
            tick labels to be shown on the left and bottom axis. In addition,
            if ``'#'`` is included in the sequence, the position will be
            considered dynamic and will be updated at draw-time in order to
            attempt to optimize the layout of all the coordinates.
        """
    def get_ticklabel_position(self):
        """
        Get where tick labels will appear.
        """
    def set_ticklabel_visible(self, visible) -> None:
        """
        Set whether the tick labels are visible or not.

        Parameters
        ----------
        visible : bool
            The visibility of ticks. Setting as ``False`` will hide this
            coordinate's tick labels.
        """
    def set_axislabel(self, text, minpad: int = 1, **kwargs) -> None:
        """
        Set the text and optionally visual properties for the axis label.

        Parameters
        ----------
        text : str
            The axis label text.
        minpad : float, optional
            The padding for the label in terms of axis label font size.
        **kwargs
            Keywords are passed to :class:`matplotlib.text.Text`. These
            can include keywords to set the ``color``, ``size``, ``weight``, and
            other text properties.
        """
    def get_axislabel(self):
        """
        Get the text for the axis label.

        Returns
        -------
        label : str
            The axis label
        """
    def set_auto_axislabel(self, auto_label) -> None:
        """
        Render default axis labels if no explicit label is provided.

        Parameters
        ----------
        auto_label : `bool`
            `True` if default labels will be rendered.
        """
    def get_auto_axislabel(self):
        """
        Render default axis labels if no explicit label is provided.

        Returns
        -------
        auto_axislabel : `bool`
            `True` if default labels will be rendered.
        """
    def _get_default_axislabel(self): ...
    def set_axislabel_position(self, position) -> None:
        """
        Set where axis labels should appear.

        Parameters
        ----------
        position : str or list
            The axes on which the axis label for this coordinate should
            appear. Should be a sequence containing zero or more of ``'b'``,
            ``'t'``, ``'l'``, ``'r'``. For example, ``'lb'`` will lead the
            axis label to be shown on the left and bottom axis. In addition, if
            ``'#'`` is included in the sequence, the position will be considered
            dynamic and will be updated at draw-time in order to show the axis
            label on the same axes as the tick labels are shown.
        """
    def get_axislabel_position(self):
        """
        Get where axis labels will appear.
        """
    def set_axislabel_visibility_rule(self, rule) -> None:
        """
        Set the rule used to determine when the axis label is drawn.

        Parameters
        ----------
        rule : str
            If the rule is 'always' axis labels will always be drawn on the
            axis. If the rule is 'ticks' the label will only be drawn if ticks
            were drawn on that axis. If the rule is 'labels' the axis label
            will only be drawn if tick labels were drawn on that axis.
        """
    def get_axislabel_visibility_rule(self, rule):
        """
        Get the rule used to determine when the axis label is drawn.
        """
    @property
    def locator(self): ...
    @property
    def formatter(self): ...
    def _draw_grid(self, renderer) -> None: ...
    def _draw_ticks(self, renderer, existing_bboxes) -> None:
        """
        Draw all ticks and ticklabels.

        Parameters
        ----------
        existing_bboxes : list[Bbox]
            All bboxes for ticks that have already been drawn by other
            coordinates.
        """
    def _draw_axislabels(self, renderer, bboxes, ticklabels_bbox, visible_ticks) -> None: ...
    _lblinfo: Incomplete
    _lbl_world: Incomplete
    def _update_ticks(self) -> None: ...
    def _compute_ticks(self, tick_world_coordinates, spine, axis, w1, w2, tick_angle, ticks: str = 'major') -> None: ...
    def display_minor_ticks(self, display_minor_ticks) -> None:
        """
        Display minor ticks for this coordinate.

        Parameters
        ----------
        display_minor_ticks : bool
            Whether or not to display minor ticks.
        """
    def get_minor_frequency(self): ...
    def set_minor_frequency(self, frequency) -> None:
        """
        Set the frequency of minor ticks per major ticks.

        Parameters
        ----------
        frequency : int
            The number of minor ticks per major ticks.
        """
    def _update_grid_lines_1d(self) -> None: ...
    def _update_grid_lines(self) -> None: ...
    def add_tickable_gridline(self, name, constant):
        """
        Define a gridline that can be used for ticks and labels.

        This gridline is not itself drawn, but instead can be specified in calls to
        methods such as
        :meth:`~astropy.visualization.wcsaxes.coordinate_helpers.CoordinateHelper.set_ticklabel_position`
        for drawing ticks and labels.  Since the gridline has a constant value in this
        coordinate, and thus would not have any ticks or labels for the same coordinate,
        the call to
        :meth:`~astropy.visualization.wcsaxes.coordinate_helpers.CoordinateHelper.set_ticklabel_position`
        would typically be made on the complementary coordinate.

        Parameters
        ----------
        name : str
            The name for the gridline, usually a single character, but can be longer
        constant : `~astropy.units.Quantity`
            The constant coordinate value of the gridline

        Notes
        -----
        A limitation is that the tickable part of the gridline must be contiguous.  If
        the gridline consists of more than one disconnected segment within the plot
        extent, only one of those segments will be made tickable.
        """
    def _get_gridline(self, xy_world, pixel, xy_world_round): ...
    def _clear_grid_contour(self) -> None: ...
    _grid: Incomplete
    def _update_grid_contour(self) -> None: ...
    def tick_params(self, which: str = 'both', **kwargs) -> None:
        """
        Method to set the tick and tick label parameters in the same way as the
        :meth:`~matplotlib.axes.Axes.tick_params` method in Matplotlib.

        This is provided for convenience, but the recommended API is to use
        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticks`,
        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticklabel`,
        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticks_position`,
        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticklabel_position`,
        and :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.grid`.

        Parameters
        ----------
        which : {'both', 'major', 'minor'}, optional
            Which ticks to apply the settings to. By default, setting are
            applied to both major and minor ticks. Note that if ``'minor'`` is
            specified, only the length of the ticks can be set currently.
        direction : {'in', 'out'}, optional
            Puts ticks inside the axes, or outside the axes.
        length : float, optional
            Tick length in points.
        width : float, optional
            Tick width in points.
        color : color, optional
            Tick color (accepts any valid Matplotlib color)
        pad : float, optional
            Distance in points between tick and label.
        labelsize : float or str, optional
            Tick label font size in points or as a string (e.g., 'large').
        labelcolor : color, optional
            Tick label color (accepts any valid Matplotlib color)
        colors : color, optional
            Changes the tick color and the label color to the same value
             (accepts any valid Matplotlib color).
        bottom, top, left, right : bool, optional
            Where to draw the ticks. Note that this will not work correctly if
            the frame is not rectangular.
        labelbottom, labeltop, labelleft, labelright : bool, optional
            Where to draw the tick labels. Note that this will not work
            correctly if the frame is not rectangular.
        grid_color : color, optional
            The color of the grid lines (accepts any valid Matplotlib color).
        grid_alpha : float, optional
            Transparency of grid lines: 0 (transparent) to 1 (opaque).
        grid_linewidth : float, optional
            Width of grid lines in points.
        grid_linestyle : str, optional
            The style of the grid lines (accepts any valid Matplotlib line
            style).
        """
