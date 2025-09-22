from .angles import Angle
from _typeshed import Incomplete
from astropy.coordinates import Latitude, Longitude, SkyCoord
from astropy.table import QTable
from astropy.units import Unit
from astropy.utils.data_info import MixinInfo
from astropy.utils.masked import MaskableShapedLikeNDArray
from typing import Literal, NamedTuple

__all__ = ['BaseCoordinateFrame', 'CoordinateFrameInfo', 'frame_transform_graph', 'GenericFrame', 'RepresentationMapping']

frame_transform_graph: Incomplete

class RepresentationMapping(NamedTuple):
    """
    This :class:`~typing.NamedTuple` is used with the
    ``frame_specific_representation_info`` attribute to tell frames what
    attribute names (and default units) to use for a particular representation.
    ``reprname`` and ``framename`` should be strings, while ``defaultunit`` can
    be either an astropy unit, the string ``'recommended'`` (which is degrees
    for Angles, nothing otherwise), or None (to indicate that no unit mapping
    should be done).
    """
    reprname: str
    framename: str
    defaultunit: str | Unit = ...

class CoordinateFrameInfo(MixinInfo):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    attrs_from_parent: Incomplete
    _supports_indexing: bool
    mask_val: Incomplete
    @staticmethod
    def default_format(val): ...
    @property
    def unit(self): ...
    @property
    def _repr_data(self): ...
    def _represent_as_dict(self): ...
    def new_like(self, coords, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        '''A new consistent coordinate instance with the given length.

        Return a new SkyCoord or BaseCoordinateFrame instance which is
        consistent with the input coordinate objects ``coords`` and has
        ``length`` rows.  Being "consistent" is defined as being able to set an
        item from one to each of the rest without any exception being raised.

        This is intended for creating a new coordinate instance whose elements
        can be set in-place for table operations like join or vstack.  This is
        used when a coordinate object is used as a mixin column in an astropy
        Table.

        The data values are not predictable and it is expected that the consumer
        of the object will fill in all values.

        Parameters
        ----------
        coords : list
            List of input SkyCoord or BaseCoordinateFrame objects
        length : int
            Length of the output SkyCoord or BaseCoordinateFrame object
        metadata_conflicts : str (\'warn\'|\'error\'|\'silent\')
            How to handle metadata conflicts
        name : str
            Output name (sets output coord.info.name)

        Returns
        -------
        coord : |SkyCoord|, |BaseFrame|
            Instance of this class consistent with ``coords``

        '''
    def _insert(self, obj, values, axis: int = 0):
        """
        Make a copy with coordinate values inserted before the given indices.

        The values to be inserted must conform to the rules for in-place setting
        of the object.

        The API signature matches the ``np.insert`` API, but is more limited.
        The specification of insert index ``obj`` must be a single integer,
        and the ``axis`` must be ``0`` for simple insertion before the index.

        Parameters
        ----------
        obj : int
            Integer index before which ``values`` is inserted.
        values : array-like
            Value(s) to insert.  If the type of ``values`` is different
            from that of quantity, ``values`` is converted to the matching type.
        axis : int, optional
            Axis along which to insert ``values``.  Default is 0, which is the
            only allowed value and will insert a row.

        Returns
        -------
        coord : |SkyCoord|, |BaseFrame|
            Copy of instance with new values inserted.
        """

class BaseCoordinateFrame(MaskableShapedLikeNDArray):
    """
    The base class for coordinate frames.

    This class is intended to be subclassed to create instances of specific
    systems.  Subclasses can implement the following attributes:

    * `default_representation`
        A subclass of `~astropy.coordinates.BaseRepresentation` that will be
        treated as the default representation of this frame.  This is the
        representation assumed by default when the frame is created.

    * `default_differential`
        A subclass of `~astropy.coordinates.BaseDifferential` that will be
        treated as the default differential class of this frame.  This is the
        differential class assumed by default when the frame is created.

    * `~astropy.coordinates.Attribute` class attributes
       Frame attributes such as ``FK4.equinox`` or ``FK4.obstime`` are defined
       using a descriptor class.  See the narrative documentation or
       built-in classes code for details.

    * `frame_specific_representation_info`
        A dictionary mapping the name or class of a representation to a list of
        `~astropy.coordinates.RepresentationMapping` objects that tell what
        names and default units should be used on this frame for the components
        of that representation.

    Unless overridden via `frame_specific_representation_info`, velocity name
    defaults are:

      * ``pm_{lon}_cos{lat}``, ``pm_{lat}`` for `~astropy.coordinates.SphericalCosLatDifferential` velocity components
      * ``pm_{lon}``, ``pm_{lat}`` for `~astropy.coordinates.SphericalDifferential` velocity components
      * ``radial_velocity`` for any ``d_distance`` component
      * ``v_{x,y,z}`` for `~astropy.coordinates.CartesianDifferential` velocity components

    where ``{lon}`` and ``{lat}`` are the frame names of the angular components.
    """
    default_representation: Incomplete
    default_differential: Incomplete
    frame_specific_representation_info: Incomplete
    frame_attributes: Incomplete
    info: Incomplete
    def __init_subclass__(cls, **kwargs) -> None: ...
    _attr_names_with_defaults: Incomplete
    _representation: Incomplete
    _shape: Incomplete
    _data: Incomplete
    def __init__(self, *args, copy: bool = True, representation_type: Incomplete | None = None, differential_type: Incomplete | None = None, **kwargs) -> None: ...
    def _infer_representation(self, representation_type, differential_type): ...
    def _infer_data(self, args, copy, kwargs): ...
    @classmethod
    def _infer_repr_info(cls, repr_info): ...
    @classmethod
    def _create_readonly_property(cls, attr_name, value, doc: Incomplete | None = None): ...
    def cache(self):
        """Cache for this frame, a dict.

        It stores anything that should be computed from the coordinate data (*not* from
        the frame attributes). This can be used in functions to store anything that
        might be expensive to compute but might be re-used by some other function.
        E.g.::

            if 'user_data' in myframe.cache:
                data = myframe.cache['user_data']
            else:
                myframe.cache['user_data'] = data = expensive_func(myframe.lat)

        If in-place modifications are made to the frame data, the cache should
        be cleared::

            myframe.cache.clear()

        """
    @property
    def data(self):
        """
        The coordinate data for this object.  If this frame has no data, an
        `ValueError` will be raised.  Use `has_data` to
        check if data is present on this frame object.
        """
    @property
    def has_data(self):
        """
        True if this frame has `data`, False otherwise.
        """
    @property
    def shape(self): ...
    def __bool__(self) -> bool: ...
    @property
    def size(self): ...
    @property
    def masked(self):
        """Whether the underlying data is masked.

        Raises
        ------
        ValueError
            If the frame has no associated data.
        """
    def get_mask(self, *attrs):
        '''Get the mask associated with these coordinates.

        Parameters
        ----------
        *attrs : str
            Attributes from which to get the masks to combine. Items can be
            dotted, like ``"data.lon", "data.lat"``. By default, get the
            combined mask of all components (including from differentials),
            ignoring possible masks of attributes.

        Returns
        -------
        mask : ~numpy.ndarray of bool
            The combined, read-only mask. If the instance is not masked, it
            is an array of `False` with the correct shape.

        Raises
        ------
        ValueError
            If the coordinate frame has no associated data.

        '''
    mask: Incomplete
    @classmethod
    def get_frame_attr_defaults(cls):
        """Return a dict with the defaults for each frame attribute."""
    def get_representation_cls(self, which: str = 'base'):
        """The class used for part of this frame's data.

        Parameters
        ----------
        which : ('base', 's', `None`)
            The class of which part to return.  'base' means the class used to
            represent the coordinates; 's' the first derivative to time, i.e.,
            the class representing the proper motion and/or radial velocity.
            If `None`, return a dict with both.

        Returns
        -------
        representation : `~astropy.coordinates.BaseRepresentation` or `~astropy.coordinates.BaseDifferential`.
        """
    def set_representation_cls(self, base: Incomplete | None = None, s: str = 'base') -> None:
        """Set representation and/or differential class for this frame's data.

        Parameters
        ----------
        base : str, `~astropy.coordinates.BaseRepresentation` subclass, optional
            The name or subclass to use to represent the coordinate data.
        s : `~astropy.coordinates.BaseDifferential` subclass, optional
            The differential subclass to use to represent any velocities,
            such as proper motion and radial velocity.  If equal to 'base',
            which is the default, it will be inferred from the representation.
            If `None`, the representation will drop any differentials.
        """
    representation_type: Incomplete
    @property
    def differential_type(self):
        """
        The differential used for this frame's data.

        This will be a subclass from `~astropy.coordinates.BaseDifferential`.
        For simultaneous setting of representation and differentials, see the
        ``set_representation_cls`` method.
        """
    @differential_type.setter
    def differential_type(self, value) -> None: ...
    @classmethod
    def _get_representation_info(cls): ...
    def representation_info(self):
        """
        A dictionary with the information of what attribute names for this frame
        apply to particular representations.
        """
    def get_representation_component_names(self, which: str = 'base'): ...
    def get_representation_component_units(self, which: str = 'base'): ...
    representation_component_names: Incomplete
    representation_component_units: Incomplete
    def _replicate(self, data, copy: bool = False, **kwargs):
        """Base for replicating a frame, with possibly different attributes.

        Produces a new instance of the frame using the attributes of the old
        frame (unless overridden) and with the data given.

        Parameters
        ----------
        data : `~astropy.coordinates.BaseRepresentation` or None
            Data to use in the new frame instance.  If `None`, it will be
            a data-less frame.
        copy : bool, optional
            Whether data and the attributes on the old frame should be copied
            (default), or passed on by reference.
        **kwargs
            Any attributes that should be overridden.
        """
    def replicate(self, copy: bool = False, **kwargs):
        """
        Return a replica of the frame, optionally with new frame attributes.

        The replica is a new frame object that has the same data as this frame
        object and with frame attributes overridden if they are provided as extra
        keyword arguments to this method. If ``copy`` is set to `True` then a
        copy of the internal arrays will be made.  Otherwise the replica will
        use a reference to the original arrays when possible to save memory. The
        internal arrays are normally not changeable by the user so in most cases
        it should not be necessary to set ``copy`` to `True`.

        Parameters
        ----------
        copy : bool, optional
            If True, the resulting object is a copy of the data.  When False,
            references are used where  possible. This rule also applies to the
            frame attributes.
        **kwargs
            Any additional keywords are treated as frame attributes to be set on the
            new frame object.

        Returns
        -------
        frameobj : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            Replica of this object, but possibly with new frame attributes.
        """
    def replicate_without_data(self, copy: bool = False, **kwargs):
        """
        Return a replica without data, optionally with new frame attributes.

        The replica is a new frame object without data but with the same frame
        attributes as this object, except where overridden by extra keyword
        arguments to this method.  The ``copy`` keyword determines if the frame
        attributes are truly copied vs being references (which saves memory for
        cases where frame attributes are large).

        This method is essentially the converse of `realize_frame`.

        Parameters
        ----------
        copy : bool, optional
            If True, the resulting object has copies of the frame attributes.
            When False, references are used where  possible.
        **kwargs
            Any additional keywords are treated as frame attributes to be set on the
            new frame object.

        Returns
        -------
        frameobj : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            Replica of this object, but without data and possibly with new frame
            attributes.
        """
    def realize_frame(self, data, **kwargs):
        """
        Generates a new frame with new data from another frame (which may or
        may not have data). Roughly speaking, the converse of
        `replicate_without_data`.

        Parameters
        ----------
        data : `~astropy.coordinates.BaseRepresentation`
            The representation to use as the data for the new frame.
        **kwargs
            Any additional keywords are treated as frame attributes to be set on the
            new frame object. In particular, `representation_type` can be specified.

        Returns
        -------
        frameobj : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            A new object in *this* frame, with the same frame attributes as
            this one, but with the ``data`` as the coordinate data.

        """
    def represent_as(self, base, s: str = 'base', in_frame_units: bool = False):
        """
        Generate and return a new representation of this frame's `data`
        as a Representation object.

        Note: In order to make an in-place change of the representation
        of a Frame or SkyCoord object, set the ``representation``
        attribute of that object to the desired new representation, or
        use the ``set_representation_cls`` method to also set the differential.

        Parameters
        ----------
        base : subclass of BaseRepresentation or string
            The type of representation to generate.  Must be a *class*
            (not an instance), or the string name of the representation
            class.
        s : subclass of `~astropy.coordinates.BaseDifferential`, str, optional
            Class in which any velocities should be represented. Must be
            a *class* (not an instance), or the string name of the
            differential class.  If equal to 'base' (default), inferred from
            the base class.  If `None`, all velocity information is dropped.
        in_frame_units : bool, keyword-only
            Force the representation units to match the specified units
            particular to this frame

        Returns
        -------
        newrep : BaseRepresentation-derived object
            A new representation object of this frame's `data`.

        Raises
        ------
        AttributeError
            If this object had no `data`

        Examples
        --------
        >>> from astropy import units as u
        >>> from astropy.coordinates import SkyCoord, CartesianRepresentation
        >>> coord = SkyCoord(0*u.deg, 0*u.deg)
        >>> coord.represent_as(CartesianRepresentation)  # doctest: +FLOAT_CMP
        <CartesianRepresentation (x, y, z) [dimensionless]
                (1., 0., 0.)>

        >>> coord.representation_type = CartesianRepresentation
        >>> coord  # doctest: +FLOAT_CMP
        <SkyCoord (ICRS): (x, y, z) [dimensionless]
            (1., 0., 0.)>
        """
    def transform_to(self, new_frame):
        """
        Transform this object's coordinate data to a new frame.

        Parameters
        ----------
        new_frame : coordinate-like
            The frame to transform this coordinate frame into.

        Returns
        -------
        transframe : coordinate-like
            A new object with the coordinate data represented in the
            ``newframe`` system.

        Raises
        ------
        ValueError
            If there is no possible transformation route.
        """
    def is_transformable_to(self, new_frame):
        """
        Determines if this coordinate frame can be transformed to another
        given frame.

        Parameters
        ----------
        new_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass or instance
            The proposed frame to transform into.

        Returns
        -------
        transformable : bool or str
            `True` if this can be transformed to ``new_frame``, `False` if
            not, or the string 'same' if ``new_frame`` is the same system as
            this object but no transformation is defined.

        Notes
        -----
        A return value of 'same' means the transformation will work, but it will
        just give back a copy of this object.  The intended usage is::

            if coord.is_transformable_to(some_unknown_frame):
                coord2 = coord.transform_to(some_unknown_frame)

        This will work even if ``some_unknown_frame``  turns out to be the same
        frame class as ``coord``.  This is intended for cases where the frame
        is the same regardless of the frame attributes (e.g. ICRS), but be
        aware that it *might* also indicate that someone forgot to define the
        transformation between two objects of the same frame class but with
        different attributes.
        """
    def is_frame_attr_default(self, attrnm):
        """
        Determine whether or not a frame attribute has its value because it's
        the default value, or because this frame was created with that value
        explicitly requested.

        Parameters
        ----------
        attrnm : str
            The name of the attribute to check.

        Returns
        -------
        isdefault : bool
            True if the attribute ``attrnm`` has its value by default, False if
            it was specified at creation of this frame.
        """
    @staticmethod
    def _frameattr_equiv(left_fattr, right_fattr):
        """
        Determine if two frame attributes are equivalent.  Implemented as a
        staticmethod mainly as a convenient location, although conceivable it
        might be desirable for subclasses to override this behavior.

        Primary purpose is to check for equality of representations.
        Secondary purpose is to check for equality of coordinate attributes,
        which first checks whether they themselves are in equivalent frames
        before checking for equality in the normal fashion.  This is because
        checking for equality with non-equivalent frames raises an error.
        """
    def is_equivalent_frame(self, other):
        """
        Checks if this object is the same frame as the ``other`` object.

        To be the same frame, two objects must be the same frame class and have
        the same frame attributes.  Note that it does *not* matter what, if any,
        data either object has.

        Parameters
        ----------
        other : :class:`~astropy.coordinates.BaseCoordinateFrame`
            the other frame to check

        Returns
        -------
        isequiv : bool
            True if the frames are the same, False if not.

        Raises
        ------
        TypeError
            If ``other`` isn't a `~astropy.coordinates.BaseCoordinateFrame` or subclass.
        """
    def __repr__(self) -> str: ...
    def _data_repr(self):
        """Returns a string representation of the coordinate data."""
    def _frame_attrs_repr(self):
        """
        Returns a string representation of the frame's attributes, if any.
        """
    def _apply(self, method, *args, **kwargs):
        """Create a new instance, applying a method to the underlying data.

        In typical usage, the method is any of the shape-changing methods for
        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those
        picking particular elements (``__getitem__``, ``take``, etc.), which
        are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be
        applied to the underlying arrays in the representation (e.g., ``x``,
        ``y``, and ``z`` for `~astropy.coordinates.CartesianRepresentation`),
        as well as to any frame attributes that have a shape, with the results
        used to create a new instance.

        Internally, it is also used to apply functions to the above parts
        (in particular, `~numpy.broadcast_to`).

        Parameters
        ----------
        method : str or callable
            If str, it is the name of a method that is applied to the internal
            ``components``. If callable, the function is applied.
        *args : tuple
            Any positional arguments for ``method``.
        **kwargs : dict
            Any keyword arguments for ``method``.
        """
    def __setitem__(self, item, value) -> None: ...
    def insert(self, obj, values, axis: int = 0): ...
    def __dir__(self):
        """
        Override the builtin `dir` behavior to include representation
        names.

        TODO: dynamic representation transforms (i.e. include cylindrical et al.).
        """
    def __getattr__(self, attr):
        """
        Allow access to attributes on the representation and differential as
        found via ``self.get_representation_component_names``.

        TODO: We should handle dynamic representation transforms here (e.g.,
        `.cylindrical`) instead of defining properties as below.
        """
    def __setattr__(self, attr, value) -> None: ...
    def __eq__(self, value):
        """Equality operator for frame.

        This implements strict equality and requires that the frames are
        equivalent and that the representation data are exactly equal.
        """
    def __ne__(self, value): ...
    def _prepare_unit_sphere_coords(self, other: BaseCoordinateFrame | SkyCoord, origin_mismatch: Literal['ignore', 'warn', 'error']) -> tuple[Longitude, Latitude, Longitude, Latitude]: ...
    def position_angle(self, other: BaseCoordinateFrame | SkyCoord) -> Angle:
        '''Compute the on-sky position angle to another coordinate.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`
            The other coordinate to compute the position angle to.  It is
            treated as the "head" of the vector of the position angle.

        Returns
        -------
        `~astropy.coordinates.Angle`
            The (positive) position angle of the vector pointing from ``self``
            to ``other``, measured East from North.  If either ``self`` or
            ``other`` contain arrays, this will be an array following the
            appropriate `numpy` broadcasting rules.

        Examples
        --------
        >>> from astropy import units as u
        >>> from astropy.coordinates import ICRS, SkyCoord
        >>> c1 = SkyCoord(0*u.deg, 0*u.deg)
        >>> c2 = ICRS(1*u.deg, 0*u.deg)
        >>> c1.position_angle(c2).to(u.deg)
        <Angle 90. deg>
        >>> c2.position_angle(c1).to(u.deg)
        <Angle 270. deg>
        >>> c3 = SkyCoord(1*u.deg, 1*u.deg)
        >>> c1.position_angle(c3).to(u.deg)  # doctest: +FLOAT_CMP
        <Angle 44.995636455344844 deg>
        '''
    def separation(self, other: BaseCoordinateFrame | SkyCoord, *, origin_mismatch: Literal['ignore', 'warn', 'error'] = 'warn') -> Angle:
        '''
        Computes on-sky separation between this coordinate and another.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`
            The coordinate to get the separation to.
        origin_mismatch : {"warn", "ignore", "error"}, keyword-only
            If the ``other`` coordinates are in a different frame then they
            will have to be transformed, and if the transformation is not a
            pure rotation then ``self.separation(other)`` can be
            different from ``other.separation(self)``. With
            ``origin_mismatch="warn"`` (default) the transformation is
            always performed, but a warning is emitted if it is not a
            pure rotation. If ``origin_mismatch="ignore"`` then the
            required transformation is always performed without warnings.
            If ``origin_mismatch="error"`` then only transformations
            that are pure rotations are allowed.

        Returns
        -------
        sep : `~astropy.coordinates.Angle`
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance

        '''
    def separation_3d(self, other):
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`
            The coordinate system to get the distance to.

        Returns
        -------
        sep : `~astropy.coordinates.Distance`
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.
        """
    @property
    def cartesian(self):
        """
        Shorthand for a cartesian representation of the coordinates in this
        object.
        """
    @property
    def cylindrical(self):
        """
        Shorthand for a cylindrical representation of the coordinates in this
        object.
        """
    @property
    def spherical(self):
        """
        Shorthand for a spherical representation of the coordinates in this
        object.
        """
    @property
    def sphericalcoslat(self):
        """
        Shorthand for a spherical representation of the positional data and a
        `~astropy.coordinates.SphericalCosLatDifferential` for the velocity
        data in this object.
        """
    @property
    def velocity(self):
        """
        Shorthand for retrieving the Cartesian space-motion as a
        `~astropy.coordinates.CartesianDifferential` object.

        This is equivalent to calling ``self.cartesian.differentials['s']``.
        """
    @property
    def proper_motion(self):
        """
        Shorthand for the two-dimensional proper motion as a
        `~astropy.units.Quantity` object with angular velocity units. In the
        returned `~astropy.units.Quantity`, ``axis=0`` is the longitude/latitude
        dimension so that ``.proper_motion[0]`` is the longitudinal proper
        motion and ``.proper_motion[1]`` is latitudinal. The longitudinal proper
        motion already includes the cos(latitude) term.
        """
    @property
    def radial_velocity(self):
        """
        Shorthand for the radial or line-of-sight velocity as a
        `~astropy.units.Quantity` object.
        """
    def to_table(self) -> QTable:
        """
        Convert this |BaseFrame| to a |QTable|.

        Any attributes that have the same length as the |BaseFrame| will be
        converted to columns of the |QTable|. All other attributes will be
        recorded as metadata.

        Returns
        -------
        `~astropy.table.QTable`
            A |QTable| containing the data of this |BaseFrame|.

        Examples
        --------
        >>> from astropy.coordinates import ICRS
        >>> coord = ICRS(ra=[40, 70]*u.deg, dec=[0, -20]*u.deg)
        >>> t =  coord.to_table()
        >>> t
        <QTable length=2>
           ra     dec
          deg     deg
        float64 float64
        ------- -------
           40.0     0.0
           70.0   -20.0
        >>> t.meta
        {'representation_type': 'spherical'}
        """

class GenericFrame(BaseCoordinateFrame):
    """
    A frame object that can't store data but can hold any arbitrary frame
    attributes. Mostly useful as a utility for the high-level class to store
    intermediate frame attributes.

    Parameters
    ----------
    frame_attrs : dict
        A dictionary of attributes to be used as the frame attributes for this
        frame.
    """
    name: Incomplete
    frame_attributes: Incomplete
    def __init__(self, frame_attrs) -> None: ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, value) -> None: ...
