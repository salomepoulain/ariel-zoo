from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['TransformGraph']

class TransformGraph:
    """
    A graph representing the paths between coordinate frames.
    """
    _graph: Incomplete
    def __init__(self) -> None: ...
    def _cached_names(self): ...
    _cached_frame_set: Incomplete
    @property
    def frame_set(self):
        """
        A `set` of all the frame classes present in this TransformGraph.
        """
    def frame_attributes(self):
        """
        A `dict` of all the attributes of all frame classes in this TransformGraph.
        """
    @property
    def frame_component_names(self):
        """
        A `set` of all component names every defined within any frame class in
        this TransformGraph.
        """
    _shortestpaths: Incomplete
    _composite_cache: Incomplete
    def invalidate_cache(self) -> None:
        """
        Invalidates the cache that stores optimizations for traversing the
        transform graph.  This is called automatically when transforms
        are added or removed, but will need to be called manually if
        weights on transforms are modified inplace.
        """
    def add_transform(self, fromsys, tosys, transform) -> None:
        """Add a new coordinate transformation to the graph.

        Parameters
        ----------
        fromsys : class
            The coordinate frame class to start from.
        tosys : class
            The coordinate frame class to transform into.
        transform : `~astropy.coordinates.CoordinateTransform`
            The transformation object. Typically a
            `~astropy.coordinates.CoordinateTransform` object, although it may
            be some other callable that is called with the same signature.

        Raises
        ------
        TypeError
            If ``fromsys`` or ``tosys`` are not classes or ``transform`` is
            not callable.

        """
    def remove_transform(self, fromsys, tosys, transform) -> None:
        """
        Removes a coordinate transform from the graph.

        Parameters
        ----------
        fromsys : class or None
            The coordinate frame *class* to start from. If `None`,
            ``transform`` will be searched for and removed (``tosys`` must
            also be `None`).
        tosys : class or None
            The coordinate frame *class* to transform into. If `None`,
            ``transform`` will be searched for and removed (``fromsys`` must
            also be `None`).
        transform : callable or None
            The transformation object to be removed or `None`.  If `None`
            and ``tosys`` and ``fromsys`` are supplied, there will be no
            check to ensure the correct object is removed.
        """
    def find_shortest_path(self, fromsys, tosys):
        """
        Computes the shortest distance along the transform graph from
        one system to another.

        Parameters
        ----------
        fromsys : class
            The coordinate frame class to start from.
        tosys : class
            The coordinate frame class to transform into.

        Returns
        -------
        path : list of class or None
            The path from ``fromsys`` to ``tosys`` as an in-order sequence
            of classes.  This list includes *both* ``fromsys`` and
            ``tosys``. Is `None` if there is no possible path.
        distance : float or int
            The total distance/priority from ``fromsys`` to ``tosys``.  If
            priorities are not set this is the number of transforms
            needed. Is ``inf`` if there is no possible path.
        """
    def get_transform(self, fromsys, tosys):
        """Generates and returns the CompositeTransform for a transformation
        between two coordinate systems.

        Parameters
        ----------
        fromsys : class
            The coordinate frame class to start from.
        tosys : class
            The coordinate frame class to transform into.

        Returns
        -------
        trans : `~astropy.coordinates.CompositeTransform` or None
            If there is a path from ``fromsys`` to ``tosys``, this is a
            transform object for that path.   If no path could be found, this is
            `None`.

        Notes
        -----
        A `~astropy.coordinates.CompositeTransform` is always returned, because
        `~astropy.coordinates.CompositeTransform` is slightly more adaptable in
        the way it can be called than other transform classes. Specifically, it
        takes care of intermediate steps of transformations in a way that is
        consistent with 1-hop transformations.

        """
    def lookup_name(self, name):
        """
        Tries to locate the coordinate class with the provided alias.

        Parameters
        ----------
        name : str
            The alias to look up.

        Returns
        -------
        `BaseCoordinateFrame` subclass
            The coordinate class corresponding to the ``name`` or `None` if
            no such class exists.
        """
    def get_names(self):
        """
        Returns all available transform names. They will all be
        valid arguments to `lookup_name`.

        Returns
        -------
        nms : list
            The aliases for coordinate systems.
        """
    def to_dot_graph(self, priorities: bool = True, addnodes=[], savefn: Incomplete | None = None, savelayout: str = 'plain', saveformat: Incomplete | None = None, color_edges: bool = True):
        '''
        Converts this transform graph to the graphviz_ DOT format.

        Optionally saves it (requires `graphviz`_ be installed and on your path).

        .. _graphviz: http://www.graphviz.org/

        Parameters
        ----------
        priorities : bool
            If `True`, show the priority values for each transform.  Otherwise,
            the will not be included in the graph.
        addnodes : sequence of str
            Additional coordinate systems to add (this can include systems
            already in the transform graph, but they will only appear once).
        savefn : None or str
            The file name to save this graph to or `None` to not save
            to a file.
        savelayout : {"plain", "dot", "neato", "fdp", "sfdp", "circo", "twopi", "nop", "nop2", "osage", "patchwork"}
            The graphviz program to use to layout the graph (see
            graphviz_ for details) or \'plain\' to just save the DOT graph
            content. Ignored if ``savefn`` is `None`.
        saveformat : str
            The graphviz output format. (e.g. the ``-Txxx`` option for
            the command line program - see graphviz docs for details).
            Ignored if ``savefn`` is `None`.
        color_edges : bool
            Color the edges between two nodes (frames) based on the type of
            transform. ``FunctionTransform``: red, ``StaticMatrixTransform``:
            blue, ``DynamicMatrixTransform``: green.

        Returns
        -------
        dotgraph : str
            A string with the DOT format graph.
        '''
    def to_networkx_graph(self):
        """
        Converts this transform graph into a networkx graph.

        .. note::
            You must have the `networkx <https://networkx.github.io/>`_
            package installed for this to work.

        Returns
        -------
        nxgraph : ``networkx.Graph``
            This `~astropy.coordinates.TransformGraph` as a
            `networkx.Graph <https://networkx.github.io/documentation/stable/reference/classes/graph.html>`_.
        """
    def transform(self, transcls, fromsys, tosys, priority: int = 1, **kwargs):
        """A function decorator for defining transformations.

        .. note::
            If decorating a static method of a class, ``@staticmethod``
            should be  added *above* this decorator.

        Parameters
        ----------
        transcls : class
            The class of the transformation object to create.
        fromsys : class
            The coordinate frame class to start from.
        tosys : class
            The coordinate frame class to transform into.
        priority : float or int
            The priority if this transform when finding the shortest
            coordinate transform path - large numbers are lower priorities.

        Additional keyword arguments are passed into the ``transcls``
        constructor.

        Returns
        -------
        deco : function
            A function that can be called on another function as a decorator
            (see example).

        Notes
        -----
        This decorator assumes the first argument of the ``transcls``
        initializer accepts a callable, and that the second and third are
        ``fromsys`` and ``tosys``. If this is not true, you should just
        initialize the class manually and use
        `~astropy.coordinates.TransformGraph.add_transform` instead of this
        decorator.

        Examples
        --------
        ::

            graph = TransformGraph()

            class Frame1(BaseCoordinateFrame):
               ...

            class Frame2(BaseCoordinateFrame):
                ...

            @graph.transform(FunctionTransform, Frame1, Frame2)
            def f1_to_f2(f1_obj):
                ... do something with f1_obj ...
                return f2_obj

        """
    def _add_merged_transform(self, fromsys, tosys, *furthersys, priority: int = 1) -> None:
        """
        Add a single-step transform that encapsulates a multi-step transformation path,
        using the transforms that already exist in the graph.

        The created transform internally calls the existing transforms.  If all of the
        transforms are affine, the merged transform is
        `~astropy.coordinates.DynamicMatrixTransform` (if there are no
        origin shifts) or `~astropy.coordinates.AffineTransform`
        (otherwise).  If at least one of the transforms is not affine, the merged
        transform is
        `~astropy.coordinates.FunctionTransformWithFiniteDifference`.

        This method is primarily useful for defining loopback transformations
        (i.e., where ``fromsys`` and the final ``tosys`` are the same).

        Parameters
        ----------
        fromsys : class
            The coordinate frame class to start from.
        tosys : class
            The coordinate frame class to transform to.
        *furthersys : class
            Additional coordinate frame classes to transform to in order.
        priority : number
            The priority of this transform when finding the shortest
            coordinate transform path - large numbers are lower priorities.

        Notes
        -----
        Even though the created transform is a single step in the graph, it
        will still internally call the constituent transforms.  Thus, there is
        no performance benefit for using this created transform.

        For Astropy's built-in frames, loopback transformations typically use
        `~astropy.coordinates.ICRS` to be safe.  Transforming through an inertial
        frame ensures that changes in observation time and observer
        location/velocity are properly accounted for.

        An error will be raised if a direct transform between ``fromsys`` and
        ``tosys`` already exist.
        """
    def impose_finite_difference_dt(self, dt) -> Generator[None]:
        """
        Context manager to impose a finite-difference time step on all applicable transformations.

        For each transformation in this transformation graph that has the attribute
        ``finite_difference_dt``, that attribute is set to the provided value.  The only standard
        transformation with this attribute is
        `~astropy.coordinates.FunctionTransformWithFiniteDifference`.

        Parameters
        ----------
        dt : `~astropy.units.Quantity` ['time'] or callable
            If a quantity, this is the size of the differential used to do the finite difference.
            If a callable, should accept ``(fromcoord, toframe)`` and return the ``dt`` value.
        """
