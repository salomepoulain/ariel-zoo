import abc
from _typeshed import Incomplete
from astropy.coordinates.angles import Angle as Angle
from astropy.utils import classproperty as classproperty
from astropy.utils.data_info import MixinInfo as MixinInfo
from astropy.utils.exceptions import DuplicateRepresentationWarning as DuplicateRepresentationWarning
from astropy.utils.masked import MaskableShapedLikeNDArray as MaskableShapedLikeNDArray, Masked as Masked, combine_masks as combine_masks

REPRESENTATION_CLASSES: Incomplete
DIFFERENTIAL_CLASSES: Incomplete
DUPLICATE_REPRESENTATIONS: Incomplete

def _fqn_class(cls):
    """Get the fully qualified name of a class."""
def get_reprdiff_cls_hash():
    """
    Returns a hash value that should be invariable if the
    `REPRESENTATION_CLASSES` and `DIFFERENTIAL_CLASSES` dictionaries have not
    changed.
    """

class BaseRepresentationOrDifferentialInfo(MixinInfo):
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
    def _represent_as_dict_attrs(self): ...
    @property
    def unit(self): ...
    def new_like(self, reps, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new instance like ``reps`` with ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        reps : list
            List of input representations or differentials.
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : `~astropy.coordinates.BaseRepresentation` or `~astropy.coordinates.BaseDifferential` subclass instance
            Empty instance of this class consistent with ``cols``

        """

class BaseRepresentationOrDifferential(MaskableShapedLikeNDArray, metaclass=abc.ABCMeta):
    """3D coordinate representations and differentials.

    Parameters
    ----------
    comp1, comp2, comp3 : `~astropy.units.Quantity` or subclass
        The components of the 3D point or differential.  The names are the
        keys and the subclasses the values of the ``attr_classes`` attribute.
    copy : bool, optional
        If `True` (default), arrays will be copied; if `False`, they will be
        broadcast together but not use new memory.
    """
    __array_priority__: int
    info: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_name(cls):
        """Name of the representation or differential.

        In lower case, with any trailing 'representation' or 'differential'
        removed. (E.g., 'spherical' for
        `~astropy.coordinates.SphericalRepresentation` or
        `~astropy.coordinates.SphericalDifferential`.)
        """
    @classmethod
    @abc.abstractmethod
    def from_cartesian(cls, other):
        """Create a representation of this class from a supplied Cartesian one.

        Parameters
        ----------
        other : `~astropy.coordinates.CartesianRepresentation`
            The representation to turn into this class

        Returns
        -------
        representation : `~astropy.coordinates.BaseRepresentation` subclass instance
            A new representation of this class's type.
        """
    @abc.abstractmethod
    def to_cartesian(self):
        """Convert the representation to its Cartesian form.

        Note that any differentials get dropped.
        Also note that orientation information at the origin is *not* preserved by
        conversions through Cartesian coordinates. For example, transforming
        an angular position defined at distance=0 through cartesian coordinates
        and back will lose the original angular coordinates::

            >>> import astropy.units as u
            >>> import astropy.coordinates as coord
            >>> rep = coord.SphericalRepresentation(
            ...     lon=15*u.deg,
            ...     lat=-11*u.deg,
            ...     distance=0*u.pc)
            >>> rep.to_cartesian().represent_as(coord.SphericalRepresentation)
            <SphericalRepresentation (lon, lat, distance) in (rad, rad, pc)
                (0., 0., 0.)>

        Returns
        -------
        cartrepr : `~astropy.coordinates.CartesianRepresentation`
            The representation in Cartesian form.
        """
    @property
    def components(self):
        """A tuple with the in-order names of the coordinate components."""
    def __eq__(self, value):
        """Equality operator.

        This implements strict equality and requires that the representation
        classes are identical and that the representation data are exactly equal.
        """
    def __ne__(self, value): ...
    def _apply(self, method, *args, **kwargs):
        """Create a new representation or differential with ``method`` applied
        to the component data.

        In typical usage, the method is any of the shape-changing methods for
        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those
        picking particular elements (``__getitem__``, ``take``, etc.), which
        are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be
        applied to the underlying arrays (e.g., ``x``, ``y``, and ``z`` for
        `~astropy.coordinates.CartesianRepresentation`), with the results used
        to create a new instance.

        Internally, it is also used to apply functions to the components
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
    @property
    def shape(self):
        """The shape of the instance and underlying arrays.

        Like `~numpy.ndarray.shape`, can be set to a new shape by assigning a
        tuple.  Note that if different instances share some but not all
        underlying data, setting the shape of one instance can make the other
        instance unusable.  Hence, it is strongly recommended to get new,
        reshaped instances with the ``reshape`` method.

        Raises
        ------
        ValueError
            If the new shape has the wrong total number of elements.
        AttributeError
            If the shape of any of the components cannot be changed without the
            arrays being copied.  For these cases, use the ``reshape`` method
            (which copies any arrays that cannot be reshaped in-place).
        """
    @shape.setter
    def shape(self, shape) -> None: ...
    @property
    def masked(self): ...
    def _ensure_masked(self) -> None:
        """Ensure Masked components."""
    def get_mask(self, *attrs):
        """Calculate the mask, by combining masks from the given attributes.

        Parameters
        ----------
        *attrs : str
            Attributes from which to get the masks to combine. If not given,
            use all components of the class.

        Returns
        -------
        mask : ~numpy.ndarray of bool
            The combined, read-only mask. If the instance is not masked, it
            is an array of `False` with the correct shape.
        """
    mask: Incomplete
    @abc.abstractmethod
    def _scale_operation(self, op, *args): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __neg__(self): ...
    def __pos__(self): ...
    @abc.abstractmethod
    def _combine_operation(self, op, other, reverse: bool = False): ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    @property
    def _values(self):
        """Turn the coordinates into a record array with the coordinate values.

        The record array fields will have the component names.
        """
    @property
    def _units(self):
        """Return a dictionary with the units of the coordinate components."""
    @property
    def _unitstr(self): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class RepresentationInfo(BaseRepresentationOrDifferentialInfo):
    @property
    def _represent_as_dict_attrs(self): ...
    def _represent_as_dict(self, attrs: Incomplete | None = None): ...
    def _construct_from_dict(self, map): ...

class BaseRepresentation(BaseRepresentationOrDifferential, metaclass=abc.ABCMeta):
    """Base for representing a point in a 3D coordinate system.

    Parameters
    ----------
    comp1, comp2, comp3 : `~astropy.units.Quantity` or subclass
        The components of the 3D points.  The names are the keys and the
        subclasses the values of the ``attr_classes`` attribute.
    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        subclass instance, or a dictionary with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.

    Notes
    -----
    All representation classes should subclass this base representation class,
    and define an ``attr_classes`` attribute, a `dict`
    which maps component names to the class that creates them. They must also
    define a ``to_cartesian`` method and a ``from_cartesian`` class method. By
    default, transformations are done via the cartesian system, but classes
    that want to define a smarter transformation path can overload the
    ``represent_as`` method. If one wants to use an associated differential
    class, one should also define ``unit_vectors`` and ``scale_factors``
    methods (see those methods for details).
    """
    info: Incomplete
    _differentials: Incomplete
    def __init_subclass__(cls, **kwargs): ...
    def __init__(self, *args, differentials: Incomplete | None = None, **kwargs) -> None: ...
    def _ensure_masked(self) -> None: ...
    def _validate_differentials(self, differentials):
        """
        Validate that the provided differentials are appropriate for this
        representation and recast/reshape as necessary and then return.

        Note that this does *not* set the differentials on
        ``self._differentials``, but rather leaves that for the caller.
        """
    def _raise_if_has_differentials(self, op_name) -> None:
        """
        Used to raise a consistent exception for any operation that is not
        supported when a representation has differentials attached.
        """
    def _compatible_differentials(cls): ...
    @property
    def differentials(self):
        """A dictionary of differential class instances.

        The keys of this dictionary must be a string representation of the SI
        unit with which the differential (derivative) is taken. For example, for
        a velocity differential on a positional representation, the key would be
        ``'s'`` for seconds, indicating that the derivative is a time
        derivative.
        """
    def unit_vectors(self) -> None:
        """Cartesian unit vectors in the direction of each component.

        Given unit vectors :math:`\\hat{e}_c` and scale factors :math:`f_c`,
        a change in one component of :math:`\\delta c` corresponds to a change
        in representation of :math:`\\delta c \\times f_c \\times \\hat{e}_c`.

        Returns
        -------
        unit_vectors : dict of `~astropy.coordinates.CartesianRepresentation`
            The keys are the component names.
        """
    def scale_factors(self) -> None:
        """Scale factors for each component's direction.

        Given unit vectors :math:`\\hat{e}_c` and scale factors :math:`f_c`,
        a change in one component of :math:`\\delta c` corresponds to a change
        in representation of :math:`\\delta c \\times f_c \\times \\hat{e}_c`.

        Returns
        -------
        scale_factors : dict of `~astropy.units.Quantity`
            The keys are the component names.
        """
    def _re_represent_differentials(self, new_rep, differential_class):
        """Re-represent the differentials to the specified classes.

        This returns a new dictionary with the same keys but with the
        attached differentials converted to the new differential classes.
        """
    def represent_as(self, other_class, differential_class: Incomplete | None = None):
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via Cartesian coordinates.
        Also note that orientation information at the origin is *not* preserved by
        conversions through Cartesian coordinates. See the docstring for
        :meth:`~astropy.coordinates.BaseRepresentationOrDifferential.to_cartesian`
        for an example.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseRepresentation` subclass
            The type of representation to turn the coordinates into.
        differential_class : dict of `~astropy.coordinates.BaseDifferential`, optional
            Classes in which the differentials should be represented.
            Can be a single class if only a single differential is attached,
            otherwise it should be a `dict` keyed by the same keys as the
            differentials.
        """
    def transform(self, matrix):
        """Transform coordinates using a 3x3 matrix in a Cartesian basis.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 (or stack thereof) matrix, such as a rotation matrix.

        """
    def with_differentials(self, differentials):
        """
        Create a new representation with the same positions as this
        representation, but with these new differentials.

        Differential keys that already exist in this object's differential dict
        are overwritten.

        Parameters
        ----------
        differentials : sequence of `~astropy.coordinates.BaseDifferential` subclass instance
            The differentials for the new representation to have.

        Returns
        -------
        `~astropy.coordinates.BaseRepresentation` subclass instance
            A copy of this representation, but with the ``differentials`` as
            its differentials.
        """
    def without_differentials(self):
        """Return a copy of the representation without attached differentials.

        Returns
        -------
        `~astropy.coordinates.BaseRepresentation` subclass instance
            A shallow copy of this representation, without any differentials.
            If no differentials were present, no copy is made.
        """
    @classmethod
    def from_representation(cls, representation):
        """Create a new instance of this representation from another one.

        Parameters
        ----------
        representation : `~astropy.coordinates.BaseRepresentation` instance
            The presentation that should be converted to this class.
        """
    def __eq__(self, value):
        """Equality operator for BaseRepresentation.

        This implements strict equality and requires that the representation
        classes are identical, the differentials are identical, and that the
        representation data are exactly equal.
        """
    def __ne__(self, value): ...
    def _apply(self, method, *args, **kwargs):
        """Create a new representation with ``method`` applied to the component
        data.

        This is not a simple inherit from ``BaseRepresentationOrDifferential``
        because we need to call ``._apply()`` on any associated differential
        classes.

        See docstring for `BaseRepresentationOrDifferential._apply`.

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
    def _scale_operation(self, op, *args):
        """Scale all non-angular components, leaving angular ones unchanged.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.mul`, `~operator.neg`, etc.
        *args
            Any arguments required for the operator (typically, what is to
            be multiplied with, divided by).
        """
    def _combine_operation(self, op, other, reverse: bool = False):
        """Combine two representation.

        By default, operate on the cartesian representations of both.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            The other representation.
        reverse : bool
            Whether the operands should be reversed (e.g., as we got here via
            ``self.__rsub__`` because ``self`` is a subclass of ``other``).
        """
    @BaseRepresentationOrDifferential.shape.setter
    def shape(self, shape) -> None: ...
    def norm(self):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.

        Note that any associated differentials will be dropped during this
        operation.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.
        """
    def mean(self, *args, **kwargs):
        """Vector mean.

        Averaging is done by converting the representation to cartesian, and
        taking the mean of the x, y, and z components. The result is converted
        back to the same representation as the input.

        Refer to `~numpy.mean` for full documentation of the arguments, noting
        that ``axis`` is the entry in the ``shape`` of the representation, and
        that the ``out`` argument cannot be used.

        Returns
        -------
        mean : `~astropy.coordinates.BaseRepresentation` subclass instance
            Vector mean, in the same representation as that of the input.
        """
    def sum(self, *args, **kwargs):
        """Vector sum.

        Adding is done by converting the representation to cartesian, and
        summing the x, y, and z components. The result is converted back to the
        same representation as the input.

        Refer to `~numpy.sum` for full documentation of the arguments, noting
        that ``axis`` is the entry in the ``shape`` of the representation, and
        that the ``out`` argument cannot be used.

        Returns
        -------
        sum : `~astropy.coordinates.BaseRepresentation` subclass instance
            Vector sum, in the same representation as that of the input.
        """
    def dot(self, other):
        """Dot product of two representations.

        The calculation is done by converting both ``self`` and ``other``
        to `~astropy.coordinates.CartesianRepresentation`.

        Note that any associated differentials will be dropped during this
        operation.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseRepresentation`
            The representation to take the dot product with.

        Returns
        -------
        dot_product : `~astropy.units.Quantity`
            The sum of the product of the x, y, and z components of the
            cartesian representations of ``self`` and ``other``.
        """
    def cross(self, other):
        """Vector cross product of two representations.

        The calculation is done by converting both ``self`` and ``other``
        to `~astropy.coordinates.CartesianRepresentation`, and converting the
        result back to the type of representation of ``self``.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            The representation to take the cross product with.

        Returns
        -------
        cross_product : `~astropy.coordinates.BaseRepresentation` subclass instance
            With vectors perpendicular to both ``self`` and ``other``, in the
            same type of representation as ``self``.
        """

class BaseDifferential(BaseRepresentationOrDifferential):
    """A base class representing differentials of representations.

    These represent differences or derivatives along each component.
    E.g., for physics spherical coordinates, these would be
    :math:`\\delta r, \\delta \\theta, \\delta \\phi`.

    Parameters
    ----------
    d_comp1, d_comp2, d_comp3 : `~astropy.units.Quantity` or subclass
        The components of the 3D differentials.  The names are the keys and the
        subclasses the values of the ``attr_classes`` attribute.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.

    Notes
    -----
    All differential representation classes should subclass this base class,
    and define an ``base_representation`` attribute with the class of the
    regular `~astropy.coordinates.BaseRepresentation` for which differential
    coordinates are provided. This will set up a default ``attr_classes``
    instance with names equal to the base component names prefixed by ``d_``,
    and all classes set to `~astropy.units.Quantity`, plus properties to access
    those, and a default ``__init__`` for initialization.
    """
    def __init_subclass__(cls, **kwargs):
        """Set default ``attr_classes`` and component getters on a Differential.

        For these, the components are those of the base representation prefixed
        by 'd_', and the class is `~astropy.units.Quantity`.
        """
    @classmethod
    def _check_base(cls, base) -> None: ...
    def _get_deriv_key(self, base):
        """Given a base (representation instance), determine the unit of the
        derivative by removing the representation unit from the component units
        of this differential.
        """
    @classmethod
    def _get_base_vectors(cls, base):
        """Get unit vectors and scale factors from base.

        Parameters
        ----------
        base : instance of ``self.base_representation``
            The points for which the unit vectors and scale factors should be
            retrieved.

        Returns
        -------
        unit_vectors : dict of `~astropy.coordinates.CartesianRepresentation`
            In the directions of the coordinates of base.
        scale_factors : dict of `~astropy.units.Quantity`
            Scale factors for each of the coordinates

        Raises
        ------
        TypeError : if the base is not of the correct type
        """
    def to_cartesian(self, base):
        """Convert the differential to 3D rectangular cartesian coordinates.

        Parameters
        ----------
        base : instance of ``self.base_representation``
            The points for which the differentials are to be converted: each of
            the components is multiplied by its unit vectors and scale factors.

        Returns
        -------
        `~astropy.coordinates.CartesianDifferential`
            This object, converted.

        """
    @classmethod
    def from_cartesian(cls, other, base):
        """Convert the differential from 3D rectangular cartesian coordinates to
        the desired class.

        Parameters
        ----------
        other
            The object to convert into this differential.
        base : `~astropy.coordinates.BaseRepresentation`
            The points for which the differentials are to be converted: each of
            the components is multiplied by its unit vectors and scale factors.
            Will be converted to ``cls.base_representation`` if needed.

        Returns
        -------
        `~astropy.coordinates.BaseDifferential` subclass instance
            A new differential object that is this class' type.
        """
    def represent_as(self, other_class, base):
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via cartesian coordinates.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseRepresentation` subclass
            The type of representation to turn the coordinates into.
        base : instance of ``self.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        """
    @classmethod
    def from_representation(cls, representation, base):
        """Create a new instance of this representation from another one.

        Parameters
        ----------
        representation : `~astropy.coordinates.BaseRepresentation` instance
            The presentation that should be converted to this class.
        base : instance of ``cls.base_representation``
            The base relative to which the differentials will be defined. If
            the representation is a differential itself, the base will be
            converted to its ``base_representation`` to help convert it.
        """
    def transform(self, matrix, base, transformed_base):
        """Transform differential using a 3x3 matrix in a Cartesian basis.

        This returns a new differential and does not modify the original one.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 (or stack thereof) matrix, such as a rotation matrix.
        base : instance of ``cls.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        transformed_base : instance of ``cls.base_representation``
            Base relative to which the transformed differentials are defined.
            If the other class is a differential representation, the base will
            be converted to its ``base_representation``.
        """
    def _scale_operation(self, op, *args, scaled_base: bool = False):
        """Scale all components.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.mul`, `~operator.neg`, etc.
        *args
            Any arguments required for the operator (typically, what is to
            be multiplied with, divided by).
        scaled_base : bool, optional
            Whether the base was scaled the same way. This affects whether
            differential components should be scaled. For instance, a differential
            in longitude should not be scaled if its spherical base is scaled
            in radius.
        """
    def _combine_operation(self, op, other, reverse: bool = False):
        """Combine two differentials, or a differential with a representation.

        If ``other`` is of the same differential type as ``self``, the
        components will simply be combined.  If ``other`` is a representation,
        it will be used as a base for which to evaluate the differential,
        and the result is a new representation.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.
        other : `~astropy.coordinates.BaseRepresentation` subclass instance
            The other differential or representation.
        reverse : bool
            Whether the operands should be reversed (e.g., as we got here via
            ``self.__rsub__`` because ``self`` is a subclass of ``other``).
        """
    def __sub__(self, other): ...
    def norm(self, base: Incomplete | None = None):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.

        Parameters
        ----------
        base : instance of ``self.base_representation``
            Base relative to which the differentials are defined. This is
            required to calculate the physical size of the differential for
            all but Cartesian differentials or radial differentials.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.
        """
