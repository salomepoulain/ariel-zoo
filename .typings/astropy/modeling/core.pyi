import abc
from _typeshed import Incomplete

__all__ = ['Model', 'FittableModel', 'Fittable1DModel', 'Fittable2DModel', 'CompoundModel', 'fix_inputs', 'custom_model', 'ModelDefinitionError', 'bind_bounding_box', 'bind_compound_bounding_box']

class ModelDefinitionError(TypeError):
    """Used for incorrect models definitions."""

class _ModelMeta(abc.ABCMeta):
    """
    Metaclass for Model.

    Currently just handles auto-generating the param_names list based on
    Parameter descriptors declared at the class-level of Model subclasses.
    """
    _is_dynamic: bool
    def __new__(mcls, name, bases, members, **kwds): ...
    def __init__(cls, name, bases, members, **kwds) -> None: ...
    def __repr__(cls) -> str:
        """
        Custom repr for Model subclasses.
        """
    def _repr_pretty_(cls, p, cycle) -> None:
        '''
        Repr for IPython\'s pretty printer.

        By default IPython "pretty prints" classes, so we need to implement
        this so that IPython displays the custom repr for Models.
        '''
    def __reduce__(cls): ...
    @property
    def name(cls):
        """
        The name of this model class--equivalent to ``cls.__name__``.

        This attribute is provided for symmetry with the
        `~astropy.modeling.Model.name` attribute of model instances.
        """
    @property
    def _is_concrete(cls):
        """
        A class-level property that determines whether the class is a concrete
        implementation of a Model--i.e. it is not some abstract base class or
        internal implementation detail (i.e. begins with '_').
        """
    def rename(cls, name: Incomplete | None = None, inputs: Incomplete | None = None, outputs: Incomplete | None = None):
        """
        Creates a copy of this model class with a new name, inputs or outputs.

        The new class is technically a subclass of the original class, so that
        instance and type checks will still work.  For example::

            >>> from astropy.modeling.models import Rotation2D
            >>> SkyRotation = Rotation2D.rename('SkyRotation')
            >>> SkyRotation
            <class 'astropy.modeling.core.SkyRotation'>
            Name: SkyRotation (Rotation2D)
            N_inputs: 2
            N_outputs: 2
            Fittable parameters: ('angle',)
            >>> issubclass(SkyRotation, Rotation2D)
            True
            >>> r = SkyRotation(90)
            >>> isinstance(r, Rotation2D)
            True
        """
    def _create_inverse_property(cls, members) -> None: ...
    def _create_bounding_box_property(cls, members) -> None:
        """
        Takes any bounding_box defined on a concrete Model subclass (either
        as a fixed tuple or a property or method) and wraps it in the generic
        getter/setter interface for the bounding_box attribute.
        """
    def _create_bounding_box_subclass(cls, func, sig):
        """
        For Models that take optional arguments for defining their bounding
        box, we create a subclass of ModelBoundingBox with a ``__call__`` method
        that supports those additional arguments.

        Takes the function's Signature as an argument since that is already
        computed in _create_bounding_box_property, so no need to duplicate that
        effort.
        """
    def _handle_special_methods(cls, members, pdict): ...
    __add__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __truediv__: Incomplete
    __pow__: Incomplete
    __or__: Incomplete
    __and__: Incomplete
    _fix_inputs: Incomplete
    def _format_cls_repr(cls, keywords=[]):
        """
        Internal implementation of ``__repr__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__repr__`` while keeping the same basic
        formatting.
        """

class Model(metaclass=_ModelMeta):
    '''
    Base class for all models.

    This is an abstract class and should not be instantiated directly.

    The following initialization arguments apply to the majority of Model
    subclasses by default (exceptions include specialized utility models
    like `~astropy.modeling.mappings.Mapping`).  Parametric models take all
    their parameters as arguments, followed by any of the following optional
    keyword arguments:

    Parameters
    ----------
    name : str, optional
        A human-friendly name associated with this model instance
        (particularly useful for identifying the individual components of a
        compound model).

    meta : dict, optional
        An optional dict of user-defined metadata to attach to this model.
        How this is used and interpreted is up to the user or individual use
        case.

    n_models : int, optional
        If given an integer greater than 1, a *model set* is instantiated
        instead of a single model.  This affects how the parameter arguments
        are interpreted.  In this case each parameter must be given as a list
        or array--elements of this array are taken along the first axis (or
        ``model_set_axis`` if specified), such that the Nth element is the
        value of that parameter for the Nth model in the set.

        See the section on model sets in the documentation for more details.

    model_set_axis : int, optional
        This argument only applies when creating a model set (i.e. ``n_models >
        1``).  It changes how parameter values are interpreted.  Normally the
        first axis of each input parameter array (properly the 0th axis) is
        taken as the axis corresponding to the model sets.  However, any axis
        of an input array may be taken as this "model set axis".  This accepts
        negative integers as well--for example use ``model_set_axis=-1`` if the
        last (most rapidly changing) axis should be associated with the model
        sets. Also, ``model_set_axis=False`` can be used to tell that a given
        input should be used to evaluate all the models in the model set.

    fixed : dict, optional
        Dictionary ``{parameter_name: bool}`` setting the fixed constraint
        for one or more parameters.  `True` means the parameter is held fixed
        during fitting and is prevented from updates once an instance of the
        model has been created.

        Alternatively the `~astropy.modeling.Parameter.fixed` property of a
        parameter may be used to lock or unlock individual parameters.

    tied : dict, optional
        Dictionary ``{parameter_name: callable}`` of parameters which are
        linked to some other parameter. The dictionary values are callables
        providing the linking relationship.

        Alternatively the `~astropy.modeling.Parameter.tied` property of a
        parameter may be used to set the ``tied`` constraint on individual
        parameters.

    bounds : dict, optional
        A dictionary ``{parameter_name: value}`` of lower and upper bounds of
        parameters. Keys are parameter names. Values are a list or a tuple
        of length 2 giving the desired range for the parameter.

        Alternatively the `~astropy.modeling.Parameter.min` and
        `~astropy.modeling.Parameter.max` or
        ~astropy.modeling.Parameter.bounds` properties of a parameter may be
        used to set bounds on individual parameters.

    eqcons : list, optional
        List of functions of length n such that ``eqcons[j](x0, *args) == 0.0``
        in a successfully optimized problem.

    ineqcons : list, optional
        List of functions of length n such that ``ieqcons[j](x0, *args) >=
        0.0`` is a successfully optimized problem.

    Examples
    --------
    >>> from astropy.modeling import models
    >>> def tie_center(model):
    ...         mean = 50 * model.stddev
    ...         return mean
    >>> tied_parameters = {\'mean\': tie_center}

    Specify that ``\'mean\'`` is a tied parameter in one of two ways:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        tied=tied_parameters)

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.mean.tied
    False
    >>> g1.mean.tied = tie_center
    >>> g1.mean.tied
    <function tie_center at 0x...>

    Fixed parameters:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        fixed={\'stddev\': True})
    >>> g1.stddev.fixed
    True

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.stddev.fixed
    False
    >>> g1.stddev.fixed = True
    >>> g1.stddev.fixed
    True
    '''
    parameter_constraints: Incomplete
    model_constraints: Incomplete
    param_names: Incomplete
    standard_broadcasting: bool
    fittable: bool
    linear: bool
    _separable: Incomplete
    meta: Incomplete
    _inverse: Incomplete
    _user_inverse: Incomplete
    _bounding_box: Incomplete
    _user_bounding_box: Incomplete
    _has_inverse_bounding_box: bool
    _n_models: int
    _input_units_strict: bool
    _input_units_allow_dimensionless: bool
    input_units_equivalencies: Incomplete
    _cov_matrix: Incomplete
    _stds: Incomplete
    def __init_subclass__(cls, **kwargs) -> None: ...
    _name: Incomplete
    _constraints_cache: Incomplete
    def __init__(self, *args, meta: Incomplete | None = None, name: Incomplete | None = None, **kwargs) -> None: ...
    _inputs: Incomplete
    _outputs: Incomplete
    def _default_inputs_outputs(self) -> None: ...
    def _initialize_setters(self, kwargs):
        """
        This exists to inject defaults for settable properties for models
        originating from `~astropy.modeling.custom_model`.
        """
    @property
    def inputs(self): ...
    @inputs.setter
    def inputs(self, val) -> None: ...
    @property
    def outputs(self): ...
    @outputs.setter
    def outputs(self, val) -> None: ...
    @property
    def n_inputs(self) -> int:
        """The number of inputs."""
    @property
    def n_outputs(self) -> int:
        """The number of outputs."""
    def _calculate_separability_matrix(self):
        """
        This is a hook which customises the behavior of modeling.separable.

        This allows complex subclasses to customise the separability matrix.
        If it returns `NotImplemented` the default behavior is used.
        """
    def _initialize_unit_support(self) -> None:
        """
        Convert self._input_units_strict and
        self.input_units_allow_dimensionless to dictionaries
        mapping input name to a boolean value.
        """
    @property
    def input_units_strict(self):
        """
        Enforce strict units on inputs to evaluate. If this is set to True,
        input values to evaluate will be in the exact units specified by
        input_units. If the input quantities are convertible to input_units,
        they are converted. If this is a dictionary then it should map input
        name to a bool to set strict input units for that parameter.
        """
    @property
    def input_units_allow_dimensionless(self):
        """
        Allow dimensionless input (and corresponding output). If this is True,
        input values to evaluate will gain the units specified in input_units. If
        this is a dictionary then it should map input name to a bool to allow
        dimensionless numbers for that input.
        Only has an effect if input_units is defined.
        """
    @property
    def uses_quantity(self):
        """
        True if this model has been created with `~astropy.units.Quantity`
        objects or if there are no parameters.

        This can be used to determine if this model should be evaluated with
        `~astropy.units.Quantity` or regular floats.
        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    @staticmethod
    def _strip_ones(intup): ...
    def __setattr__(self, attr, value) -> None: ...
    def _pre_evaluate(self, *args, **kwargs):
        """
        Model specific input setup that needs to occur prior to model evaluation.
        """
    def get_bounding_box(self, with_bbox: bool = True):
        """
        Return the ``bounding_box`` of a model if it exists or ``None``
        otherwise.

        Parameters
        ----------
        with_bbox :
            The value of the ``with_bounding_box`` keyword argument
            when calling the model. Default is `True` for usage when
            looking up the model's ``bounding_box`` without risk of error.
        """
    @property
    def _argnames(self):
        """The inputs used to determine input_shape for bounding_box evaluation."""
    def _validate_input_shape(self, _input, idx, argnames, model_set_axis, check_model_set_axis):
        """Perform basic validation of a single model input's shape.

        The shape has the minimum dimensions for the given model_set_axis.

        Returns the shape of the input if validation succeeds.
        """
    def _validate_input_shapes(self, inputs, argnames, model_set_axis):
        """
        Perform basic validation of model inputs
            --that they are mutually broadcastable and that they have
            the minimum dimensions for the given model_set_axis.

        If validation succeeds, returns the total shape that will result from
        broadcasting the input arrays with each other.
        """
    def input_shape(self, inputs):
        """Get input shape for bounding_box evaluation."""
    def _generic_evaluate(self, evaluate, _inputs, fill_value, with_bbox):
        """Generic model evaluation routine.

        Selects and evaluates model with or without bounding_box enforcement.
        """
    def _post_evaluate(self, inputs, outputs, broadcasted_shapes, with_bbox, **kwargs):
        """
        Model specific post evaluation processing of outputs.
        """
    @property
    def bbox_with_units(self): ...
    def __call__(self, *args, **kwargs):
        """
        Evaluate this model using the given input(s) and the parameter values
        that were specified when the model was instantiated.
        """
    def _get_renamed_inputs_as_positional(self, *args, **kwargs): ...
    @property
    def name(self):
        """User-provided name for this model instance."""
    @name.setter
    def name(self, val) -> None:
        """Assign a (new) name to this model."""
    @property
    def model_set_axis(self):
        """
        The index of the model set axis--that is the axis of a parameter array
        that pertains to which model a parameter value pertains to--as
        specified when the model was initialized.

        See the documentation on :ref:`astropy:modeling-model-sets`
        for more details.
        """
    @property
    def param_sets(self):
        """
        Return parameters as a pset.

        This is a list with one item per parameter set, which is an array of
        that parameter's values across all parameter sets, with the last axis
        associated with the parameter set.
        """
    @property
    def parameters(self):
        """
        A flattened array of all parameter values in all parameter sets.

        Fittable parameters maintain this list and fitters modify it.
        """
    @parameters.setter
    def parameters(self, value) -> None:
        """
        Assigning to this attribute updates the parameters array rather than
        replacing it.
        """
    _sync_constraints: bool
    @property
    def sync_constraints(self):
        """
        This is a boolean property that indicates whether or not accessing constraints
        automatically check the constituent models current values. It defaults to True
        on creation of a model, but for fitting purposes it should be set to False
        for performance reasons.
        """
    @sync_constraints.setter
    def sync_constraints(self, value) -> None: ...
    @property
    def fixed(self):
        """
        A ``dict`` mapping parameter names to their fixed constraint.
        """
    @property
    def bounds(self):
        """
        A ``dict`` mapping parameter names to their upper and lower bounds as
        ``(min, max)`` tuples or ``[min, max]`` lists.
        """
    @property
    def tied(self):
        """
        A ``dict`` mapping parameter names to their tied constraint.
        """
    @property
    def has_fixed(self):
        """
        Whether the model has any fixed constraints.
        """
    @property
    def has_bounds(self):
        """
        Whether the model has any bounds constraints.
        """
    @property
    def has_tied(self):
        """
        Whether the model has any tied constraints.
        """
    @property
    def eqcons(self):
        """List of parameter equality constraints."""
    @property
    def ineqcons(self):
        """List of parameter inequality constraints."""
    def has_inverse(self):
        """
        Returns True if the model has an analytic or user
        inverse defined.
        """
    @property
    def inverse(self):
        """
        Returns a new `~astropy.modeling.Model` instance which performs the
        inverse transform, if an analytic inverse is defined for this model.

        Even on models that don't have an inverse defined, this property can be
        set with a manually-defined inverse, such a pre-computed or
        experimentally determined inverse (often given as a
        `~astropy.modeling.polynomial.PolynomialModel`, but not by
        requirement).

        A custom inverse can be deleted with ``del model.inverse``.  In this
        case the model's inverse is reset to its default, if a default exists
        (otherwise the default is to raise `NotImplementedError`).

        Note to authors of `~astropy.modeling.Model` subclasses:  To define an
        inverse for a model simply override this property to return the
        appropriate model representing the inverse.  The machinery that will
        make the inverse manually-overridable is added automatically by the
        base class.
        """
    @inverse.setter
    def inverse(self, value) -> None: ...
    @inverse.deleter
    def inverse(self) -> None:
        """
        Resets the model's inverse to its default (if one exists, otherwise
        the model will have no inverse).
        """
    @property
    def has_user_inverse(self):
        """
        A flag indicating whether or not a custom inverse model has been
        assigned to this model by a user, via assignment to ``model.inverse``.
        """
    @property
    def bounding_box(self):
        """
        A `tuple` of length `n_inputs` defining the bounding box limits, or
        raise `NotImplementedError` for no bounding_box.

        The default limits are given by a ``bounding_box`` property or method
        defined in the class body of a specific model.  If not defined then
        this property just raises `NotImplementedError` by default (but may be
        assigned a custom value by a user).  ``bounding_box`` can be set
        manually to an array-like object of shape ``(model.n_inputs, 2)``. For
        further usage, see :ref:`astropy:bounding-boxes`

        The limits are ordered according to the `numpy` ``'C'`` indexing
        convention, and are the reverse of the model input order,
        e.g. for inputs ``('x', 'y', 'z')``, ``bounding_box`` is defined:

        * for 1D: ``(x_low, x_high)``
        * for 2D: ``((y_low, y_high), (x_low, x_high))``
        * for 3D: ``((z_low, z_high), (y_low, y_high), (x_low, x_high))``

        Examples
        --------
        Setting the ``bounding_box`` limits for a 1D and 2D model:

        >>> from astropy.modeling.models import Gaussian1D, Gaussian2D
        >>> model_1d = Gaussian1D()
        >>> model_2d = Gaussian2D(x_stddev=1, y_stddev=1)
        >>> model_1d.bounding_box = (-5, 5)
        >>> model_2d.bounding_box = ((-6, 6), (-5, 5))

        Setting the bounding_box limits for a user-defined 3D
        `~astropy.modeling.custom_model`:

        >>> from astropy.modeling.models import custom_model
        >>> def const3d(x, y, z, amp=1):
        ...    return amp
        ...
        >>> Const3D = custom_model(const3d)
        >>> model_3d = Const3D()
        >>> model_3d.bounding_box = ((-6, 6), (-5, 5), (-4, 4))

        To reset ``bounding_box`` to its default limits just delete the
        user-defined value--this will reset it back to the default defined
        on the class:

        >>> del model_1d.bounding_box

        To disable the bounding box entirely (including the default),
        set ``bounding_box`` to `None`:

        >>> model_1d.bounding_box = None
        >>> model_1d.bounding_box  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        NotImplementedError: No bounding box is defined for this model
        (note: the bounding box was explicitly disabled for this model;
        use `del model.bounding_box` to restore the default bounding box,
        if one is defined for this model).
        """
    @bounding_box.setter
    def bounding_box(self, bounding_box) -> None:
        """
        Assigns the bounding box limits.
        """
    def set_slice_args(self, *args) -> None: ...
    @bounding_box.deleter
    def bounding_box(self) -> None: ...
    @property
    def has_user_bounding_box(self):
        """
        A flag indicating whether or not a custom bounding_box has been
        assigned to this model by a user, via assignment to
        ``model.bounding_box``.
        """
    @property
    def cov_matrix(self):
        """
        Fitter should set covariance matrix, if available.
        """
    @cov_matrix.setter
    def cov_matrix(self, cov) -> None: ...
    @property
    def stds(self):
        """
        Standard deviation of parameters, if covariance matrix is available.
        """
    @stds.setter
    def stds(self, stds) -> None: ...
    @property
    def separable(self):
        """A flag indicating whether a model is separable."""
    def without_units_for_data(self, **kwargs):
        """
        Return an instance of the model for which the parameter values have
        been converted to the right units for the data, then the units have
        been stripped away.

        The input and output Quantity objects should be given as keyword
        arguments.

        Notes
        -----
        This method is needed in order to be able to fit models with units in
        the parameters, since we need to temporarily strip away the units from
        the model during the fitting (which might be done by e.g. scipy
        functions).

        The units that the parameters should be converted to are not
        necessarily the units of the input data, but are derived from them.
        Model subclasses that want fitting to work in the presence of
        quantities need to define a ``_parameter_units_for_data_units`` method
        that takes the input and output units (as two dictionaries) and
        returns a dictionary giving the target units for each parameter.

        """
    def output_units(self, **kwargs):
        """
        Return a dictionary of output units for this model given a dictionary
        of fitting inputs and outputs.

        The input and output Quantity objects should be given as keyword
        arguments.

        Notes
        -----
        This method is needed in order to be able to fit models with units in
        the parameters, since we need to temporarily strip away the units from
        the model during the fitting (which might be done by e.g. scipy
        functions).

        This method will force extra model evaluations, which maybe computationally
        expensive. To avoid this, one can add a return_units property to the model,
        see :ref:`astropy:models_return_units`.
        """
    def strip_units_from_tree(self) -> None: ...
    def with_units_from_data(self, **kwargs):
        """
        Return an instance of the model which has units for which the parameter
        values are compatible with the data units specified.

        The input and output Quantity objects should be given as keyword
        arguments.

        Notes
        -----
        This method is needed in order to be able to fit models with units in
        the parameters, since we need to temporarily strip away the units from
        the model during the fitting (which might be done by e.g. scipy
        functions).

        The units that the parameters will gain are not necessarily the units
        of the input data, but are derived from them. Model subclasses that
        want fitting to work in the presence of quantities need to define a
        ``_parameter_units_for_data_units`` method that takes the input and output
        units (as two dictionaries) and returns a dictionary giving the target
        units for each parameter.
        """
    @property
    def _has_units(self): ...
    @property
    def _supports_unit_fitting(self): ...
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the model on some input variables."""
    def sum_of_implicit_terms(self, *args, **kwargs) -> None:
        """
        Evaluate the sum of any implicit model terms on some input variables.
        This includes any fixed terms used in evaluating a linear model that
        do not have corresponding parameters exposed to the user. The
        prototypical case is `astropy.modeling.functional_models.Shift`, which
        corresponds to a function y = a + bx, where b=1 is intrinsically fixed
        by the type of model, such that sum_of_implicit_terms(x) == x. This
        method is needed by linear fitters to correct the dependent variable
        for the implicit term(s) when solving for the remaining terms
        (ie. a = y - bx).
        """
    def render(self, out: Incomplete | None = None, coords: Incomplete | None = None):
        """
        Evaluate a model at fixed positions, respecting the ``bounding_box``.

        The key difference relative to evaluating the model directly is that
        this method is limited to a bounding box if the
        `~astropy.modeling.Model.bounding_box` attribute is set.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            An array that the evaluated model will be added to.  If this is not
            given (or given as ``None``), a new array will be created.
        coords : array-like, optional
            An array to be used to translate from the model's input coordinates
            to the ``out`` array. It should have the property that
            ``self(coords)`` yields the same shape as ``out``.  If ``out`` is
            not specified, ``coords`` will be used to determine the shape of
            the returned array. If this is not provided (or None), the model
            will be evaluated on a grid determined by
            `~astropy.modeling.Model.bounding_box`.

        Returns
        -------
        out : `numpy.ndarray`
            The model added to ``out`` if ``out`` is not ``None``,
            or else a new array from evaluating the model
            over ``coords``. If ``out`` and ``coords`` are
            both `None`, the returned array is limited to the
            `~astropy.modeling.Model.bounding_box` limits. If
            `~astropy.modeling.Model.bounding_box` is `None`, ``arr`` or
            ``coords`` must be passed.

        Raises
        ------
        ValueError
            If ``coords`` are not given and the
            `~astropy.modeling.Model.bounding_box` of this model is not
            set.

        Examples
        --------
        :ref:`astropy:bounding-boxes`
        """
    @property
    def input_units(self):
        """
        This property is used to indicate what units or sets of units the
        evaluate method expects, and returns a dictionary mapping inputs to
        units (or `None` if any units are accepted).

        Model sub-classes can also use function annotations in evaluate to
        indicate valid input units, in which case this property should
        not be overridden since it will return the input units based on the
        annotations.
        """
    @property
    def return_units(self):
        """
        This property is used to indicate what units or sets of units the
        output of evaluate should be in, and returns a dictionary mapping
        outputs to units (or `None` if any units are accepted).

        Model sub-classes can also use function annotations in evaluate to
        indicate valid output units, in which case this property should not be
        overridden since it will return the return units based on the
        annotations.
        """
    def _prepare_inputs_single_model(self, params, inputs, **kwargs): ...
    @staticmethod
    def _remove_axes_from_shape(shape, axis):
        """
        Given a shape tuple as the first input, construct a new one by  removing
        that particular axis from the shape and all preceding axes. Negative axis
        numbers are permittted, where the axis is relative to the last axis.
        """
    def _prepare_inputs_model_set(self, params, inputs, model_set_axis_input, **kwargs): ...
    def prepare_inputs(self, *inputs, model_set_axis: Incomplete | None = None, equivalencies: Incomplete | None = None, **kwargs):
        """
        This method is used in `~astropy.modeling.Model.__call__` to ensure
        that all the inputs to the model can be broadcast into compatible
        shapes (if one or both of them are input as arrays), particularly if
        there are more than one parameter sets. This also makes sure that (if
        applicable) the units of the input will be compatible with the evaluate
        method.
        """
    def _validate_input_units(self, inputs, equivalencies: Incomplete | None = None, inputs_map: Incomplete | None = None): ...
    def _process_output_units(self, inputs, outputs): ...
    @staticmethod
    def _prepare_output_single_model(output, broadcast_shape): ...
    def _prepare_outputs_single_model(self, outputs, broadcasted_shapes): ...
    def _prepare_outputs_model_set(self, outputs, broadcasted_shapes, model_set_axis): ...
    def prepare_outputs(self, broadcasted_shapes, *outputs, **kwargs): ...
    def copy(self):
        """
        Return a copy of this model.

        Uses a deep copy so that all model attributes, including parameter
        values, are copied as well.
        """
    def deepcopy(self):
        """
        Return a deep copy of this model.

        """
    def rename(self, name):
        """
        Return a copy of this model with a new name.
        """
    def coerce_units(self, input_units: Incomplete | None = None, return_units: Incomplete | None = None, input_units_equivalencies: Incomplete | None = None, input_units_allow_dimensionless: bool = False):
        """
        Attach units to this (unitless) model.

        Parameters
        ----------
        input_units : dict or tuple, optional
            Input units to attach.  If dict, each key is the name of a model input,
            and the value is the unit to attach.  If tuple, the elements are units
            to attach in order corresponding to `~astropy.modeling.Model.inputs`.
        return_units : dict or tuple, optional
            Output units to attach.  If dict, each key is the name of a model output,
            and the value is the unit to attach.  If tuple, the elements are units
            to attach in order corresponding to `~astropy.modeling.Model.outputs`.
        input_units_equivalencies : dict, optional
            Default equivalencies to apply to input values.  If set, this should be a
            dictionary where each key is a string that corresponds to one of the
            model inputs.
        input_units_allow_dimensionless : bool or dict, optional
            Allow dimensionless input. If this is True, input values to evaluate will
            gain the units specified in input_units. If this is a dictionary then it
            should map input name to a bool to allow dimensionless numbers for that
            input.

        Returns
        -------
        `~astropy.modeling.CompoundModel`
            A `~astropy.modeling.CompoundModel` composed of the current
            model plus `~astropy.modeling.mappings.UnitsMapping`
            model(s) that attach the units.

        Raises
        ------
        ValueError
            If the current model already has units.

        Examples
        --------
        Wrapping a unitless model to require and convert units:

        >>> from astropy.modeling.models import Polynomial1D
        >>> from astropy import units as u
        >>> poly = Polynomial1D(1, c0=1, c1=2)
        >>> model = poly.coerce_units((u.m,), (u.s,))
        >>> model(u.Quantity(10, u.m))  # doctest: +FLOAT_CMP
        <Quantity 21. s>
        >>> model(u.Quantity(1000, u.cm))  # doctest: +FLOAT_CMP
        <Quantity 21. s>
        >>> model(u.Quantity(10, u.cm))  # doctest: +FLOAT_CMP
        <Quantity 1.2 s>

        Wrapping a unitless model but still permitting unitless input:

        >>> from astropy.modeling.models import Polynomial1D
        >>> from astropy import units as u
        >>> poly = Polynomial1D(1, c0=1, c1=2)
        >>> model = poly.coerce_units((u.m,), (u.s,), input_units_allow_dimensionless=True)
        >>> model(u.Quantity(10, u.m))  # doctest: +FLOAT_CMP
        <Quantity 21. s>
        >>> model(10)  # doctest: +FLOAT_CMP
        <Quantity 21. s>
        """
    @property
    def n_submodels(self):
        """
        Return the number of components in a single model, which is
        obviously 1.
        """
    _mconstraints: Incomplete
    def _initialize_constraints(self, kwargs) -> None:
        """
        Pop parameter constraint values off the keyword arguments passed
        to `~astropy.modeling.Model.__init__` and store them in private
        instance attributes.
        """
    def _reset_parameters(self, *args, **kwargs) -> None:
        """
        Reset parameters on the models to those specified.

        Parameters can be specified either as positional arguments or keyword
        arguments, as in the model initializer. Any parameters not specified
        will be reset to their default values.
        """
    _model_set_axis: Incomplete
    _param_metrics: Incomplete
    def _initialize_parameters(self, args, kwargs) -> None:
        """
        Initialize the _parameters array that stores raw parameter values for
        all parameter sets for use with vectorized fitting algorithms; on
        FittableModels the _param_name attributes actually just reference
        slices of this array.
        """
    def _initialize_parameter_value(self, param_name, value) -> None:
        """Mostly deals with consistency checks and determining unit issues."""
    _parameters: Incomplete
    def _initialize_slices(self) -> None: ...
    def _parameters_to_array(self) -> None: ...
    def _array_to_parameters(self) -> None: ...
    def _check_param_broadcast(self, max_ndim) -> None:
        """
        This subroutine checks that all parameter arrays can be broadcast
        against each other, and determines the shapes parameters must have in
        order to broadcast correctly.

        If model_set_axis is None this merely checks that the parameters
        broadcast and returns an empty dict if so.  This mode is only used for
        single model sets.
        """
    def _param_sets(self, raw: bool = False, units: bool = False):
        """
        Implementation of the Model.param_sets property.

        This internal implementation has a ``raw`` argument which controls
        whether or not to return the raw parameter values (i.e. the values that
        are actually stored in the ._parameters array, as opposed to the values
        displayed to users.  In most cases these are one in the same but there
        are currently a few exceptions.

        Note: This is notably an overcomplicated device and may be removed
        entirely in the near future.
        """
    def _format_repr(self, args=[], kwargs={}, defaults={}):
        """
        Internal implementation of ``__repr__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__repr__`` while keeping the same basic
        formatting.
        """
    def _format_str(self, keywords=[], defaults={}):
        """
        Internal implementation of ``__str__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__str__`` while keeping the same basic
        formatting.
        """

class FittableModel(Model, metaclass=abc.ABCMeta):
    """
    Base class for models that can be fitted using the built-in fitting
    algorithms.
    """
    linear: bool
    fit_deriv: Incomplete
    col_fit_deriv: bool
    fittable: bool

class Fittable1DModel(FittableModel, metaclass=abc.ABCMeta):
    """
    Base class for one-dimensional fittable models.

    This class provides an easier interface to defining new models.
    Examples can be found in `astropy.modeling.functional_models`.
    """
    n_inputs: int
    n_outputs: int
    _separable: bool

class Fittable2DModel(FittableModel, metaclass=abc.ABCMeta):
    """
    Base class for two-dimensional fittable models.

    This class provides an easier interface to defining new models.
    Examples can be found in `astropy.modeling.functional_models`.
    """
    n_inputs: int
    n_outputs: int

class CompoundModel(Model):
    """
    Base class for compound models.

    While it can be used directly, the recommended way
    to combine models is through the model operators.
    """
    _n_submodels: Incomplete
    op: Incomplete
    left: Incomplete
    right: Incomplete
    _bounding_box: Incomplete
    _user_bounding_box: Incomplete
    _leaflist: Incomplete
    _tdict: Incomplete
    _parameters: Incomplete
    _parameters_: Incomplete
    _param_metrics: Incomplete
    _n_models: Incomplete
    _model_set_axis: Incomplete
    inputs: Incomplete
    outputs: Incomplete
    _outputs: Incomplete
    bounding_box: Incomplete
    name: Incomplete
    _fittable: Incomplete
    linear: Incomplete
    n_left_params: Incomplete
    _constraints_cache: Incomplete
    def __init__(self, op, left, right, name: Incomplete | None = None) -> None: ...
    def _get_left_inputs_from_args(self, args): ...
    def _get_right_inputs_from_args(self, args): ...
    def _get_left_params_from_args(self, args): ...
    def _get_right_params_from_args(self, args): ...
    def _get_kwarg_model_parameters_as_positional(self, args, kwargs): ...
    def _apply_operators_to_value_lists(self, leftval, rightval, **kw): ...
    def evaluate(self, *args, **kw): ...
    @property
    def fit_deriv(self): ...
    @property
    def col_fit_deriv(self): ...
    @property
    def n_submodels(self): ...
    @property
    def submodel_names(self):
        """Return the names of submodels in a ``CompoundModel``."""
    def _pre_evaluate(self, *args, **kwargs):
        """
        CompoundModel specific input setup that needs to occur prior to
            model evaluation.

        Note
        ----
            All of the _pre_evaluate for each component model will be
            performed at the time that the individual model is evaluated.
        """
    @property
    def _argnames(self):
        """
        No inputs should be used to determine input_shape when handling compound models.
        """
    def _post_evaluate(self, inputs, outputs, broadcasted_shapes, with_bbox, **kwargs):
        """
        CompoundModel specific post evaluation processing of outputs.

        Note
        ----
            All of the _post_evaluate for each component model will be
            performed at the time that the individual model is evaluated.
        """
    def _evaluate(self, *args, **kw): ...
    @property
    def param_names(self):
        """An ordered list of parameter names."""
    def _make_leaflist(self) -> None: ...
    def __getattr__(self, name):
        """
        If someone accesses an attribute not already defined, map the
        parameters, and then see if the requested attribute is one of
        the parameters.
        """
    def __getitem__(self, index): ...
    def _str_index_to_int(self, str_index): ...
    @property
    def n_inputs(self):
        """The number of inputs of a model."""
    _n_inputs: Incomplete
    @n_inputs.setter
    def n_inputs(self, value) -> None: ...
    @property
    def n_outputs(self):
        """The number of outputs of a model."""
    _n_outputs: Incomplete
    @n_outputs.setter
    def n_outputs(self, value) -> None: ...
    @property
    def eqcons(self): ...
    _eqcons: Incomplete
    @eqcons.setter
    def eqcons(self, value) -> None: ...
    @property
    def ineqcons(self): ...
    @ineqcons.setter
    def ineqcons(self, value) -> None: ...
    def traverse_postorder(self, include_operator: bool = False):
        """Postorder traversal of the CompoundModel tree."""
    def _format_expression(self, format_leaf: Incomplete | None = None): ...
    def _format_components(self): ...
    def __str__(self) -> str: ...
    def rename(self, name): ...
    @property
    def isleaf(self): ...
    @property
    def inverse(self): ...
    @property
    def fittable(self):
        """Set the fittable attribute on a compound model."""
    __add__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __truediv__: Incomplete
    __pow__: Incomplete
    __or__: Incomplete
    __and__: Incomplete
    _param_names: Incomplete
    _param_map: Incomplete
    _param_map_inverse: Incomplete
    def _map_parameters(self) -> None:
        """
        Map all the constituent model parameters to the compound object,
        renaming as necessary by appending a suffix number.

        This can be an expensive operation, particularly for a complex
        expression tree.

        All the corresponding parameter attributes are created that one
        expects for the Model class.

        The parameter objects that the attributes point to are the same
        objects as in the constiutent models. Changes made to parameter
        values to either are seen by both.

        Prior to calling this, none of the associated attributes will
        exist. This method must be called to make the model usable by
        fitting engines.

        If oldnames=True, then parameters are named as in the original
        implementation of compound models.
        """
    @staticmethod
    def _recursive_lookup(branch, adict, key): ...
    def inputs_map(self):
        """
        Map the names of the inputs to this ExpressionTree to the inputs to the leaf models.
        """
    def _parameter_units_for_data_units(self, input_units, output_units): ...
    @property
    def input_units(self): ...
    @property
    def input_units_equivalencies(self): ...
    @property
    def input_units_allow_dimensionless(self): ...
    @property
    def input_units_strict(self): ...
    @property
    def return_units(self): ...
    def outputs_map(self):
        """
        Map the names of the outputs to this ExpressionTree to the outputs to the leaf models.
        """
    @property
    def has_user_bounding_box(self):
        """
        A flag indicating whether or not a custom bounding_box has been
        assigned to this model by a user, via assignment to
        ``model.bounding_box``.
        """
    def render(self, out: Incomplete | None = None, coords: Incomplete | None = None):
        """
        Evaluate a model at fixed positions, respecting the ``bounding_box``.

        The key difference relative to evaluating the model directly
        is that this method is limited to a bounding box if the
        `~astropy.modeling.Model.bounding_box` attribute is set.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            An array that the evaluated model will be added to.  If this is not
            given (or given as ``None``), a new array will be created.
        coords : array-like, optional
            An array to be used to translate from the model's input coordinates
            to the ``out`` array. It should have the property that
            ``self(coords)`` yields the same shape as ``out``.  If ``out`` is
            not specified, ``coords`` will be used to determine the shape of
            the returned array. If this is not provided (or None), the model
            will be evaluated on a grid determined by
            `~astropy.modeling.Model.bounding_box`.

        Returns
        -------
        out : `numpy.ndarray`
            The model added to ``out`` if  ``out`` is not ``None``, or else a
            new array from evaluating the model over ``coords``.
            If ``out`` and ``coords`` are both `None`, the returned array is
            limited to the `~astropy.modeling.Model.bounding_box`
            limits. If `~astropy.modeling.Model.bounding_box` is `None`,
            ``arr`` or ``coords`` must be passed.

        Raises
        ------
        ValueError
            If ``coords`` are not given and the
            `~astropy.modeling.Model.bounding_box` of this model is not
            set.

        Examples
        --------
        :ref:`astropy:bounding-boxes`
        """
    def replace_submodel(self, name, model):
        """
        Construct a new `~astropy.modeling.CompoundModel` instance from an
        existing CompoundModel, replacing the named submodel with a new model.

        In order to ensure that inverses and names are kept/reconstructed, it's
        necessary to rebuild the CompoundModel from the replaced node all the
        way back to the base. The original CompoundModel is left untouched.

        Parameters
        ----------
        name : str
            name of submodel to be replaced
        model : `~astropy.modeling.Model`
            replacement model
        """
    def without_units_for_data(self, **kwargs):
        """
        See `~astropy.modeling.Model.without_units_for_data` for overview
        of this method.

        Notes
        -----
        This modifies the behavior of the base method to account for the
        case where the sub-models of a compound model have different output
        units. This is only valid for compound * and / compound models as
        in that case it is reasonable to mix the output units. It does this
        by modifying the output units of each sub model by using the output
        units of the other sub model so that we can apply the original function
        and get the desired result.

        Additional data has to be output in the mixed output unit case
        so that the units can be properly rebuilt by
        `~astropy.modeling.CompoundModel.with_units_from_data`.

        Outside the mixed output units, this method is identical to the
        base method.
        """
    def with_units_from_data(self, **kwargs):
        """
        See `~astropy.modeling.Model.with_units_from_data` for overview
        of this method.

        Notes
        -----
        This modifies the behavior of the base method to account for the
        case where the sub-models of a compound model have different output
        units. This is only valid for compound * and / compound models as
        in that case it is reasonable to mix the output units. In order to
        do this it requires some additional information output by
        `~astropy.modeling.CompoundModel.without_units_for_data` passed as
        keyword arguments under the keywords ``_left_kwargs`` and ``_right_kwargs``.

        Outside the mixed output units, this method is identical to the
        base method.
        """

def fix_inputs(modelinstance, values, bounding_boxes: Incomplete | None = None, selector_args: Incomplete | None = None):
    """
    This function creates a compound model with one or more of the input
    values of the input model assigned fixed values (scalar or array).

    Parameters
    ----------
    modelinstance : `~astropy.modeling.Model` instance
        This is the model that one or more of the
        model input values will be fixed to some constant value.
    values : dict
        A dictionary where the key identifies which input to fix
        and its value is the value to fix it at. The key may either be the
        name of the input or a number reflecting its order in the inputs.

    Examples
    --------
    >>> from astropy.modeling.models import Gaussian2D
    >>> g = Gaussian2D(1, 2, 3, 4, 5)
    >>> gv = fix_inputs(g, {0: 2.5})

    Results in a 1D function equivalent to Gaussian2D(1, 2, 3, 4, 5)(x=2.5, y)
    """
def bind_bounding_box(modelinstance, bounding_box, ignored: Incomplete | None = None, order: str = 'C') -> None:
    """
    Set a validated bounding box to a model instance.

    Parameters
    ----------
    modelinstance : `~astropy.modeling.Model` instance
        This is the model that the validated bounding box will be set on.
    bounding_box : tuple
        A bounding box tuple, see :ref:`astropy:bounding-boxes` for details
    ignored : list
        List of the inputs to be ignored by the bounding box.
    order : str, optional
        The ordering of the bounding box tuple, can be either ``'C'`` or
        ``'F'``.
    """
def bind_compound_bounding_box(modelinstance, bounding_boxes, selector_args, create_selector: Incomplete | None = None, ignored: Incomplete | None = None, order: str = 'C') -> None:
    """
    Add a validated compound bounding box to a model instance.

    Parameters
    ----------
    modelinstance : `~astropy.modeling.Model` instance
        This is the model that the validated compound bounding box will be set on.
    bounding_boxes : dict
        A dictionary of bounding box tuples, see :ref:`astropy:bounding-boxes`
        for details.
    selector_args : list
        List of selector argument tuples to define selection for compound
        bounding box, see :ref:`astropy:bounding-boxes` for details.
    create_selector : callable, optional
        An optional callable with interface (selector_value, model) which
        can generate a bounding box based on a selector value and model if
        there is no bounding box in the compound bounding box listed under
        that selector value. Default is ``None``, meaning new bounding
        box entries will not be automatically generated.
    ignored : list
        List of the inputs to be ignored by the bounding box.
    order : str, optional
        The ordering of the bounding box tuple, can be either ``'C'`` or
        ``'F'``.
    """
def custom_model(*args, fit_deriv: Incomplete | None = None):
    '''
    Create a model from a user defined function. The inputs and parameters of
    the model will be inferred from the arguments of the function.

    This can be used either as a function or as a decorator.  See below for
    examples of both usages.

    The model is separable only if there is a single input.

    .. note::

        All model parameters have to be defined as keyword arguments with
        default values in the model function.  Use `None` as a default argument
        value if you do not want to have a default value for that parameter.

        The standard settable model properties can be configured by default
        using keyword arguments matching the name of the property; however,
        these values are not set as model "parameters". Moreover, users
        cannot use keyword arguments matching non-settable model properties,
        with the exception of ``n_outputs`` which should be set to the number of
        outputs of your function.

    Parameters
    ----------
    func : function
        Function which defines the model.  It should take N positional
        arguments where ``N`` is dimensions of the model (the number of
        independent variable in the model), and any number of keyword arguments
        (the parameters).  It must return the value of the model (typically as
        an array, but can also be a scalar for scalar inputs).  This
        corresponds to the `~astropy.modeling.Model.evaluate` method.
    fit_deriv : function, optional
        Function which defines the Jacobian derivative of the model. I.e., the
        derivative with respect to the *parameters* of the model.  It should
        have the same argument signature as ``func``, but should return a
        sequence where each element of the sequence is the derivative
        with respect to the corresponding argument. This corresponds to the
        :meth:`~astropy.modeling.FittableModel.fit_deriv` method.

    Examples
    --------
    Define a sinusoidal model function as a custom 1D model::

        >>> from astropy.modeling.models import custom_model
        >>> import numpy as np
        >>> def sine_model(x, amplitude=1., frequency=1.):
        ...     return amplitude * np.sin(2 * np.pi * frequency * x)
        >>> def sine_deriv(x, amplitude=1., frequency=1.):
        ...     return 2 * np.pi * amplitude * np.cos(2 * np.pi * frequency * x)
        >>> SineModel = custom_model(sine_model, fit_deriv=sine_deriv)

    Create an instance of the custom model and evaluate it::

        >>> model = SineModel()
        >>> model(0.25)  # doctest: +FLOAT_CMP
        1.0

    This model instance can now be used like a usual astropy model.

    The next example demonstrates a 2D Moffat function model, and also
    demonstrates the support for docstrings (this example could also include
    a derivative, but it has been omitted for simplicity)::

        >>> @custom_model
        ... def Moffat2D(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, gamma=1.0,
        ...            alpha=1.0):
        ...     """Two dimensional Moffat function."""
        ...     rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2
        ...     return amplitude * (1 + rr_gg) ** (-alpha)
        ...
        >>> print(Moffat2D.__doc__)
        Two dimensional Moffat function.
        >>> model = Moffat2D()
        >>> model(1, 1)  # doctest: +FLOAT_CMP
        0.3333333333333333
    '''
