import abc
import numpy as np
from _typeshed import Incomplete
from astropy.units import UnitBase
from collections.abc import Callable
from typing import Any, NamedTuple, Self

__all__ = ['ModelBoundingBox', 'CompoundBoundingBox']

class _BaseInterval(NamedTuple):
    lower: float
    upper: float

class _Interval(_BaseInterval):
    """
    A single input's bounding box interval.

    Parameters
    ----------
    lower : float
        The lower bound of the interval

    upper : float
        The upper bound of the interval

    Methods
    -------
    validate :
        Constructs a valid interval

    outside :
        Determine which parts of an input array are outside the interval.

    domain :
        Constructs a discretization of the points inside the interval.
    """
    def __repr__(self) -> str: ...
    def copy(self): ...
    @staticmethod
    def _validate_shape(interval) -> None:
        """Validate the shape of an interval representation."""
    @classmethod
    def _validate_bounds(cls, lower, upper):
        """Validate the bounds are reasonable and construct an interval from them."""
    @classmethod
    def validate(cls, interval):
        """
        Construct and validate an interval.

        Parameters
        ----------
        interval : iterable
            A representation of the interval.

        Returns
        -------
        A validated interval.
        """
    def outside(self, _input: np.ndarray):
        """
        Parameters
        ----------
        _input : np.ndarray
            The evaluation input in the form of an array.

        Returns
        -------
        Boolean array indicating which parts of _input are outside the interval:
            True  -> position outside interval
            False -> position inside  interval
        """
    def domain(self, resolution): ...

class _BoundingDomain(abc.ABC, metaclass=abc.ABCMeta):
    """
    Base class for ModelBoundingBox and CompoundBoundingBox.
        This is where all the `~astropy.modeling.core.Model` evaluation
        code for evaluating with a bounding box is because it is common
        to both types of bounding box.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The Model this bounding domain is for.

    prepare_inputs :
        Generates the necessary input information so that model can
        be evaluated only for input points entirely inside bounding_box.
        This needs to be implemented by a subclass. Note that most of
        the implementation is in ModelBoundingBox.

    prepare_outputs :
        Fills the output values in for any input points outside the
        bounding_box.

    evaluate :
        Performs a complete model evaluation while enforcing the bounds
        on the inputs and returns a complete output.
    """
    _model: Incomplete
    _ignored: Incomplete
    _order: Incomplete
    def __init__(self, model, ignored: list[int] | None = None, order: str = 'C') -> None: ...
    @property
    def model(self): ...
    @property
    def order(self) -> str: ...
    @property
    def ignored(self) -> list[int]: ...
    def _get_order(self, order: str | None = None) -> str:
        """
        Get if bounding_box is C/python ordered or Fortran/mathematically
        ordered.
        """
    def _get_index(self, key) -> int:
        """
        Get the input index corresponding to the given key.
            Can pass in either:
                the string name of the input or
                the input index itself.
        """
    def _get_name(self, index: int):
        """Get the input name corresponding to the input index."""
    @property
    def ignored_inputs(self) -> list[str]: ...
    def _validate_ignored(self, ignored: list) -> list[int]: ...
    def __call__(self, *args, **kwargs) -> None: ...
    @abc.abstractmethod
    def fix_inputs(self, model, fixed_inputs: dict):
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : dict
            Dictionary of inputs which have been fixed by this bounding box.
        """
    @abc.abstractmethod
    def prepare_inputs(self, input_shape, inputs) -> tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """
    @staticmethod
    def _base_output(input_shape, fill_value):
        """
        Create a baseline output, assuming that the entire input is outside
        the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        An array of the correct shape containing all fill_value
        """
    def _all_out_output(self, input_shape, fill_value):
        """
        Create output if all inputs are outside the domain.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        A full set of outputs for case that all inputs are outside domain.
        """
    def _modify_output(self, valid_output, valid_index, input_shape, fill_value):
        """
        For a single output fill in all the parts corresponding to inputs
        outside the bounding box.

        Parameters
        ----------
        valid_output : numpy array
            The output from the model corresponding to inputs inside the
            bounding box
        valid_index : numpy array
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        An output array with all the indices corresponding to inputs
        outside the bounding box filled in by fill_value
        """
    def _prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        """
        Fill in all the outputs of the model corresponding to inputs
        outside the bounding_box.

        Parameters
        ----------
        valid_outputs : list of numpy array
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        valid_index : numpy array
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        List of filled in output arrays.
        """
    def prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        """
        Fill in all the outputs of the model corresponding to inputs
        outside the bounding_box, adjusting any single output model so that
        its output becomes a list of containing that output.

        Parameters
        ----------
        valid_outputs : list
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        valid_index : array_like
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        """
    @staticmethod
    def _get_valid_outputs_unit(valid_outputs, with_units: bool) -> UnitBase | None:
        """
        Get the unit for outputs if one is required.

        Parameters
        ----------
        valid_outputs : list of numpy array
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        with_units : bool
            whether or not a unit is required
        """
    def _evaluate_model(self, evaluate: Callable, valid_inputs, valid_index, input_shape, fill_value, with_units: bool):
        """
        Evaluate the model using the given evaluate routine.

        Parameters
        ----------
        evaluate : Callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list of numpy arrays
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : numpy array
            array of all indices inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        with_units : bool
            whether or not a unit is required

        Returns
        -------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs
        """
    def _evaluate(self, evaluate: Callable, inputs, input_shape, fill_value, with_units: bool):
        """Evaluate model with steps: prepare_inputs -> evaluate -> prepare_outputs.

        Parameters
        ----------
        evaluate : Callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list of numpy arrays
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : numpy array
            array of all indices inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        with_units : bool
            whether or not a unit is required

        Returns
        -------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs
        """
    @staticmethod
    def _set_outputs_unit(outputs, valid_outputs_unit):
        """
        Set the units on the outputs
            prepare_inputs -> evaluate -> prepare_outputs -> set output units.

        Parameters
        ----------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs

        Returns
        -------
        List containing filled in output values and units
        """
    def evaluate(self, evaluate: Callable, inputs, fill_value):
        """
        Perform full model evaluation steps:
            prepare_inputs -> evaluate -> prepare_outputs -> set output units.

        Parameters
        ----------
        evaluate : callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        """

class ModelBoundingBox(_BoundingDomain):
    """
    A model's bounding box.

    Parameters
    ----------
    intervals : dict
        A dictionary containing all the intervals for each model input
            keys   -> input index
            values -> interval for that index

    model : `~astropy.modeling.Model`
        The Model this bounding_box is for.

    ignored : list
        A list containing all the inputs (index) which will not be
        checked for whether or not their elements are in/out of an interval.

    order : optional, str
        The ordering that is assumed for the tuple representation of this
        bounding_box. Options: 'C': C/Python order, e.g. z, y, x.
        (default), 'F': Fortran/mathematical notation order, e.g. x, y, z.
    """
    _intervals: Incomplete
    def __init__(self, intervals: dict[int, _Interval], model, ignored: list[int] | None = None, order: str = 'C') -> None: ...
    def copy(self, ignored: Incomplete | None = None): ...
    @property
    def intervals(self) -> dict[int, _Interval]:
        """Return bounding_box labeled using input positions."""
    @property
    def named_intervals(self) -> dict[str, _Interval]:
        """Return bounding_box labeled using input names."""
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __contains__(self, key) -> bool: ...
    def has_interval(self, key): ...
    def __getitem__(self, key):
        """Get bounding_box entries by either input name or input index."""
    def bounding_box(self, order: str | None = None) -> tuple[float, float] | tuple[tuple[float, float], ...]:
        """
        Return the old tuple of tuples representation of the bounding_box
            order='C' corresponds to the old bounding_box ordering
            order='F' corresponds to the gwcs bounding_box ordering.
        """
    def __eq__(self, value):
        """Note equality can be either with old representation or new one."""
    def __setitem__(self, key, value) -> None:
        """Validate and store interval under key (input index or input name)."""
    def __delitem__(self, key) -> None:
        """Delete stored interval."""
    def _validate_dict(self, bounding_box: dict):
        """Validate passing dictionary of intervals and setting them."""
    @property
    def _available_input_index(self): ...
    def _validate_sequence(self, bounding_box, order: str | None = None):
        """
        Validate passing tuple of tuples representation (or related) and setting them.
        """
    @property
    def _n_inputs(self) -> int: ...
    def _validate_iterable(self, bounding_box, order: str | None = None):
        """Validate and set any iterable representation."""
    def _validate(self, bounding_box, order: str | None = None):
        """Validate and set any representation."""
    @classmethod
    def validate(cls, model, bounding_box, ignored: list | None = None, order: str = 'C', _preserve_ignore: bool = False, **kwargs) -> Self:
        """
        Construct a valid bounding box for a model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be a bounding_box
        bounding_box : dict, tuple
            A possible representation of the bounding box
        order : optional, str
            The order that a tuple representation will be assumed to be
                Default: 'C'
        """
    def fix_inputs(self, model, fixed_inputs: dict, _keep_ignored: bool = False) -> Self:
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : dict
            Dictionary of inputs which have been fixed by this bounding box.
        keep_ignored : bool
            Keep the ignored inputs of the bounding box (internal argument only)
        """
    @property
    def dimension(self): ...
    def domain(self, resolution, order: str | None = None) -> list[np.ndarray]: ...
    def _outside(self, input_shape, inputs):
        """
        Get all the input positions which are outside the bounding_box,
        so that the corresponding outputs can be filled with the fill
        value (default NaN).

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        outside_index : bool-numpy array
            True  -> position outside bounding_box
            False -> position inside  bounding_box
        all_out : bool
            if all of the inputs are outside the bounding_box
        """
    def _valid_index(self, input_shape, inputs):
        """
        Get the indices of all the inputs inside the bounding_box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_index : numpy array
            array of all indices inside the bounding box
        all_out : bool
            if all of the inputs are outside the bounding_box
        """
    def prepare_inputs(self, input_shape, inputs) -> tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """

class _BaseSelectorArgument(NamedTuple):
    index: int
    ignore: bool

class _SelectorArgument(_BaseSelectorArgument):
    """
    Contains a single CompoundBoundingBox slicing input.

    Parameters
    ----------
    index : int
        The index of the input in the input list

    ignore : bool
        Whether or not this input will be ignored by the bounding box.

    Methods
    -------
    validate :
        Returns a valid SelectorArgument for a given model.

    get_selector :
        Returns the value of the input for use in finding the correct
        bounding_box.

    get_fixed_value :
        Gets the slicing value from a fix_inputs set of values.
    """
    def __new__(cls, index, ignore): ...
    @classmethod
    def validate(cls, model, argument, ignored: bool = True) -> Self:
        """
        Construct a valid selector argument for a CompoundBoundingBox.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be an argument for.
        argument : int or str
            A representation of which evaluation input to use
        ignored : optional, bool
            Whether or not to ignore this argument in the ModelBoundingBox.

        Returns
        -------
        Validated selector_argument
        """
    def get_selector(self, *inputs):
        """
        Get the selector value corresponding to this argument.

        Parameters
        ----------
        *inputs :
            All the processed model evaluation inputs.
        """
    def name(self, model) -> str:
        """
        Get the name of the input described by this selector argument.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.
        """
    def pretty_repr(self, model):
        """
        Get a pretty-print representation of this object.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.
        """
    def get_fixed_value(self, model, values: dict):
        """
        Gets the value fixed input corresponding to this argument.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.

        values : dict
            Dictionary of fixed inputs.
        """
    def is_argument(self, model, argument) -> bool:
        """
        Determine if passed argument is described by this selector argument.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.

        argument : int or str
            A representation of which evaluation input is being used
        """
    def named_tuple(self, model):
        """
        Get a tuple representation of this argument using the input
        name from the model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.
        """

class _SelectorArguments(tuple):
    """
    Contains the CompoundBoundingBox slicing description.

    Parameters
    ----------
    input_ :
        The SelectorArgument values

    Methods
    -------
    validate :
        Returns a valid SelectorArguments for its model.

    get_selector :
        Returns the selector a set of inputs corresponds to.

    is_selector :
        Determines if a selector is correctly formatted for this CompoundBoundingBox.

    get_fixed_value :
        Gets the selector from a fix_inputs set of values.
    """
    _kept_ignore: Incomplete
    def __new__(cls, input_: tuple[_SelectorArgument], kept_ignore: list | None = None) -> Self: ...
    def pretty_repr(self, model):
        """
        Get a pretty-print representation of this object.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.
        """
    @property
    def ignore(self):
        """Get the list of ignored inputs."""
    @property
    def kept_ignore(self):
        """The arguments to persist in ignoring."""
    @classmethod
    def validate(cls, model, arguments, kept_ignore: list | None = None) -> Self:
        """
        Construct a valid Selector description for a CompoundBoundingBox.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        arguments :
            The individual argument information

        kept_ignore :
            Arguments to persist as ignored
        """
    def get_selector(self, *inputs):
        """
        Get the selector corresponding to these inputs.

        Parameters
        ----------
        *inputs :
            All the processed model evaluation inputs.
        """
    def is_selector(self, _selector):
        """
        Determine if this is a reasonable selector.

        Parameters
        ----------
        _selector : tuple
            The selector to check
        """
    def get_fixed_values(self, model, values: dict):
        """
        Gets the value fixed input corresponding to this argument.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        values : dict
            Dictionary of fixed inputs.
        """
    def is_argument(self, model, argument) -> bool:
        """
        Determine if passed argument is one of the selector arguments.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which evaluation input is being used
        """
    def selector_index(self, model, argument):
        """
        Get the index of the argument passed in the selector tuples.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which argument is being used
        """
    def reduce(self, model, argument):
        """
        Reduce the selector arguments by the argument given.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which argument is being used
        """
    def add_ignore(self, model, argument):
        """
        Add argument to the kept_ignore list.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which argument is being used
        """
    def named_tuple(self, model):
        """
        Get a tuple of selector argument tuples using input names.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.
        """

class CompoundBoundingBox(_BoundingDomain):
    """
    A model's compound bounding box.

    Parameters
    ----------
    bounding_boxes : dict
        A dictionary containing all the ModelBoundingBoxes that are possible
            keys   -> _selector (extracted from model inputs)
            values -> ModelBoundingBox

    model : `~astropy.modeling.Model`
        The Model this compound bounding_box is for.

    selector_args : _SelectorArguments
        A description of how to extract the selectors from model inputs.

    create_selector : optional
        A method which takes in the selector and the model to return a
        valid bounding corresponding to that selector. This can be used
        to construct new bounding_boxes for previously undefined selectors.
        These new boxes are then stored for future lookups.

    order : optional, str
        The ordering that is assumed for the tuple representation of the
        bounding_boxes.
    """
    _create_selector: Incomplete
    _selector_args: Incomplete
    _bounding_boxes: Incomplete
    def __init__(self, bounding_boxes: dict[Any, ModelBoundingBox], model, selector_args: _SelectorArguments, create_selector: Callable | None = None, ignored: list[int] | None = None, order: str = 'C') -> None: ...
    def copy(self): ...
    def __repr__(self) -> str: ...
    @property
    def bounding_boxes(self) -> dict[Any, ModelBoundingBox]: ...
    @property
    def selector_args(self) -> _SelectorArguments: ...
    @selector_args.setter
    def selector_args(self, value) -> None: ...
    @property
    def named_selector_tuple(self) -> tuple: ...
    @property
    def create_selector(self): ...
    @staticmethod
    def _get_selector_key(key): ...
    def __setitem__(self, key, value) -> None: ...
    def _validate(self, bounding_boxes: dict): ...
    def __eq__(self, value): ...
    @classmethod
    def validate(cls, model, bounding_box: dict, selector_args: Incomplete | None = None, create_selector: Incomplete | None = None, ignored: list | None = None, order: str = 'C', _preserve_ignore: bool = False, **kwarg) -> Self:
        """
        Construct a valid compound bounding box for a model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be a bounding_box
        bounding_box : dict
            Dictionary of possible bounding_box representations
        selector_args : optional
            Description of the selector arguments
        create_selector : optional, callable
            Method for generating new selectors
        order : optional, str
            The order that a tuple representation will be assumed to be
                Default: 'C'
        """
    def __contains__(self, key) -> bool: ...
    def _create_bounding_box(self, _selector): ...
    def __getitem__(self, key): ...
    def _select_bounding_box(self, inputs) -> ModelBoundingBox: ...
    def prepare_inputs(self, input_shape, inputs) -> tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """
    def _matching_bounding_boxes(self, argument, value) -> dict[Any, ModelBoundingBox]: ...
    def _fix_input_selector_arg(self, argument, value): ...
    def _fix_input_bbox_arg(self, argument, value): ...
    def fix_inputs(self, model, fixed_inputs: dict) -> Self:
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : dict
            Dictionary of inputs which have been fixed by this bounding box.
        """
