from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod

__all__ = ['CoordinateTransform']

class CoordinateTransform(metaclass=ABCMeta):
    """
    An object that transforms a coordinate from one system to another.
    Subclasses must implement `__call__` with the provided signature.
    They should also call this superclass's ``__init__`` in their
    ``__init__``.

    Parameters
    ----------
    fromsys : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The coordinate frame class to start from.
    tosys : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The coordinate frame class to transform into.
    priority : float or int
        The priority if this transform when finding the shortest
        coordinate transform path - large numbers are lower priorities.
    register_graph : `~astropy.coordinates.TransformGraph` or None
        A graph to register this transformation with on creation, or
        `None` to leave it unregistered.
    """
    fromsys: Incomplete
    tosys: Incomplete
    priority: Incomplete
    overlapping_frame_attr_names: Incomplete
    def __init__(self, fromsys, tosys, priority: int = 1, register_graph: Incomplete | None = None) -> None: ...
    def register(self, graph) -> None:
        """
        Add this transformation to the requested Transformation graph,
        replacing anything already connecting these two coordinates.

        Parameters
        ----------
        graph : `~astropy.coordinates.TransformGraph` object
            The graph to register this transformation with.
        """
    def unregister(self, graph) -> None:
        """
        Remove this transformation from the requested transformation
        graph.

        Parameters
        ----------
        graph : a TransformGraph object
            The graph to unregister this transformation from.

        Raises
        ------
        ValueError
            If this is not currently in the transform graph.
        """
    @abstractmethod
    def __call__(self, fromcoord, toframe):
        """
        Does the actual coordinate transformation from the ``fromsys`` class to
        the ``tosys`` class.

        Parameters
        ----------
        fromcoord : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            An object of class matching ``fromsys`` that is to be transformed.
        toframe : object
            An object that has the attributes necessary to fully specify the
            frame.  That is, it must have attributes with names that match the
            keys of the dictionary ``tosys.frame_attributes``.
            Typically this is of class ``tosys``, but it *might* be
            some other class as long as it has the appropriate attributes.

        Returns
        -------
        tocoord : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            The new coordinate after the transform has been applied.
        """
