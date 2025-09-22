from _typeshed import Incomplete

__all__ = ['Parameter', 'InputParameterError', 'ParameterError']

class ParameterError(Exception):
    """Generic exception class for all exceptions pertaining to Parameters."""
class InputParameterError(ValueError, ParameterError):
    """Used for incorrect input parameter values and definitions."""
class ParameterDefinitionError(ParameterError):
    """Exception in declaration of class-level Parameters."""

class Parameter:
    """
    Wraps individual parameters.

    Since 4.0 Parameters are no longer descriptors and are based on a new
    implementation of the Parameter class. Parameters now  (as of 4.0) store
    values locally (as instead previously in the associated model)

    This class represents a model's parameter (in a somewhat broad sense). It
    serves a number of purposes:

    #. A type to be recognized by models and treated specially at class
       initialization (i.e., if it is found that there is a class definition
       of a Parameter, the model initializer makes a copy at the instance level).

    #. Managing the handling of allowable parameter values and once defined,
       ensuring updates are consistent with the Parameter definition. This
       includes the optional use of units and quantities as well as transforming
       values to an internally consistent representation (e.g., from degrees to
       radians through the use of getters and setters).

    #. Holding attributes of parameters relevant to fitting, such as whether
       the parameter may be varied in fitting, or whether there are constraints
       that must be satisfied.

    See :ref:`astropy:modeling-parameters` for more details.

    Parameters
    ----------
    name : str
        parameter name

        .. warning::

            The fact that `Parameter` accepts ``name`` as an argument is an
            implementation detail, and should not be used directly.  When
            defining a new `Model` class, parameter names are always
            automatically defined by the class attribute they're assigned to.
    description : str
        parameter description
    default : float or array
        default value to use for this parameter
    unit : `~astropy.units.Unit`
        if specified, the parameter will be in these units, and when the
        parameter is updated in future, it should be set to a
        :class:`~astropy.units.Quantity` that has equivalent units.
    getter : callable or `None`, optional
        A function that wraps the raw (internal) value of the parameter
        when returning the value through the parameter proxy (e.g., a
        parameter may be stored internally as radians but returned to
        the user as degrees). The internal value is what is used for
        computations while the proxy value is what users will interact
        with (passing and viewing). If ``getter`` is not `None`, then a
        ``setter`` must also be input.
    setter : callable or `None`, optional
        A function that wraps any values assigned to this parameter; should
        be the inverse of ``getter``.  If ``setter`` is not `None`, then a
        ``getter`` must also be input.
    fixed : bool
        if True the parameter is not varied during fitting
    tied : callable or False
        if callable is supplied it provides a way to link the value of this
        parameter to another parameter (or some other arbitrary function)
    min : float
        the lower bound of a parameter
    max : float
        the upper bound of a parameter
    bounds : tuple
        specify min and max as a single tuple--bounds may not be specified
        simultaneously with min or max
    mag : bool
        Specify if the unit of the parameter can be a Magnitude unit or not
    """
    constraints: Incomplete
    _model: Incomplete
    _model_required: bool
    _setter: Incomplete
    _getter: Incomplete
    _name: Incomplete
    __doc__: Incomplete
    _default: Incomplete
    _mag: Incomplete
    _internal_unit: Incomplete
    _value: Incomplete
    _fixed: Incomplete
    _tied: Incomplete
    _bounds: Incomplete
    _order: Incomplete
    _validator: Incomplete
    _prior: Incomplete
    _posterior: Incomplete
    _std: Incomplete
    def __init__(self, name: str = '', description: str = '', default: Incomplete | None = None, unit: Incomplete | None = None, getter: Incomplete | None = None, setter: Incomplete | None = None, fixed: bool = False, tied: bool = False, min: Incomplete | None = None, max: Incomplete | None = None, bounds: Incomplete | None = None, prior: Incomplete | None = None, posterior: Incomplete | None = None, mag: bool = False) -> None: ...
    def __set_name__(self, owner, name) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def name(self):
        """Parameter name."""
    @property
    def default(self):
        """Parameter default value."""
    @property
    def value(self):
        """The unadorned value proxied by this parameter."""
    _internal_value: Incomplete
    @value.setter
    def value(self, value) -> None: ...
    @property
    def unit(self):
        """
        The unit attached to this parameter, if any.

        On unbound parameters (i.e. parameters accessed through the
        model class, rather than a model instance) this is the required/
        default unit for the parameter.
        """
    @unit.setter
    def unit(self, unit) -> None: ...
    _unit: Incomplete
    def _set_unit(self, unit, force: bool = False) -> None: ...
    @property
    def internal_unit(self):
        """
        Return the internal unit the parameter uses for the internal value stored.
        """
    @internal_unit.setter
    def internal_unit(self, internal_unit) -> None:
        """
        Set the unit the parameter will convert the supplied value to the
        representation used internally.
        """
    @property
    def input_unit(self):
        """Unit for the input value."""
    @property
    def quantity(self):
        """
        This parameter, as a :class:`~astropy.units.Quantity` instance.
        """
    @quantity.setter
    def quantity(self, quantity) -> None: ...
    @property
    def shape(self):
        """The shape of this parameter's value array."""
    @shape.setter
    def shape(self, value) -> None: ...
    @property
    def size(self):
        """The size of this parameter's value array."""
    @property
    def std(self):
        """Standard deviation, if available from fit."""
    @std.setter
    def std(self, value) -> None: ...
    @property
    def prior(self): ...
    @prior.setter
    def prior(self, val) -> None: ...
    @property
    def posterior(self): ...
    @posterior.setter
    def posterior(self, val) -> None: ...
    @property
    def fixed(self):
        """
        Boolean indicating if the parameter is kept fixed during fitting.
        """
    @fixed.setter
    def fixed(self, value) -> None:
        """Fix a parameter."""
    @property
    def tied(self):
        """
        Indicates that this parameter is linked to another one.

        A callable which provides the relationship of the two parameters.
        """
    @tied.setter
    def tied(self, value) -> None:
        """Tie a parameter."""
    @property
    def bounds(self):
        """The minimum and maximum values of a parameter as a tuple."""
    @bounds.setter
    def bounds(self, value) -> None:
        """Set the minimum and maximum values of a parameter from a tuple."""
    @property
    def min(self):
        """A value used as a lower bound when fitting a parameter."""
    @min.setter
    def min(self, value) -> None:
        """Set a minimum value of a parameter."""
    @property
    def max(self):
        """A value used as an upper bound when fitting a parameter."""
    @max.setter
    def max(self, value) -> None:
        """Set a maximum value of a parameter."""
    @property
    def validator(self):
        """
        Used as a decorator to set the validator method for a `Parameter`.
        The validator method validates any value set for that parameter.
        It takes two arguments--``self``, which refers to the `Model`
        instance (remember, this is a method defined on a `Model`), and
        the value being set for this parameter.  The validator method's
        return value is ignored, but it may raise an exception if the value
        set on the parameter is invalid (typically an `InputParameterError`
        should be raised, though this is not currently a requirement).

        Note: Using this method as a decorator will cause problems with
        pickling the model. An alternative is to assign the actual validator
        function to ``Parameter._validator`` (see examples in modeling).

        """
    def validate(self, value) -> None:
        """Run the validator on this parameter."""
    def copy(self, name: Incomplete | None = None, description: Incomplete | None = None, default: Incomplete | None = None, unit: Incomplete | None = None, getter: Incomplete | None = None, setter: Incomplete | None = None, fixed: bool = False, tied: bool = False, min: Incomplete | None = None, max: Incomplete | None = None, bounds: Incomplete | None = None, prior: Incomplete | None = None, posterior: Incomplete | None = None):
        """
        Make a copy of this `Parameter`, overriding any of its core attributes
        in the process (or an exact copy).

        The arguments to this method are the same as those for the `Parameter`
        initializer.  This simply returns a new `Parameter` instance with any
        or all of the attributes overridden, and so returns the equivalent of:

        .. code:: python

            Parameter(self.name, self.description, ...)

        """
    @property
    def model(self):
        """Return the model this  parameter is associated with."""
    @model.setter
    def model(self, value) -> None: ...
    @property
    def _raw_value(self):
        """
        Currently for internal use only.

        Like Parameter.value but does not pass the result through
        Parameter.getter.  By design this should only be used from bound
        parameters.

        This will probably be removed are retweaked at some point in the
        process of rethinking how parameter values are stored/updated.
        """
    def _create_value_wrapper(self, wrapper, model):
        """Wraps a getter/setter function to support optionally passing in
        a reference to the model object as the second argument.
        If a model is tied to this parameter and its getter/setter supports
        a second argument then this creates a partial function using the model
        instance as the second argument.
        """
    def __array__(self, dtype: Incomplete | None = None, copy=...): ...
    def __bool__(self) -> bool: ...
    __add__: Incomplete
    __radd__: Incomplete
    __sub__: Incomplete
    __rsub__: Incomplete
    __mul__: Incomplete
    __rmul__: Incomplete
    __pow__: Incomplete
    __rpow__: Incomplete
    __truediv__: Incomplete
    __rtruediv__: Incomplete
    __eq__: Incomplete
    __ne__: Incomplete
    __lt__: Incomplete
    __gt__: Incomplete
    __le__: Incomplete
    __ge__: Incomplete
    __neg__: Incomplete
    __abs__: Incomplete
