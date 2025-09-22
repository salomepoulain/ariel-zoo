from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod

__all__ = ['MissingDataAssociationException', 'IncompatibleUncertaintiesException', 'NDUncertainty', 'StdDevUncertainty', 'UnknownUncertainty', 'VarianceUncertainty', 'InverseVariance']

class IncompatibleUncertaintiesException(Exception):
    """This exception should be used to indicate cases in which uncertainties
    with two different classes can not be propagated.
    """
class MissingDataAssociationException(Exception):
    """This exception should be used to indicate that an uncertainty instance
    has not been associated with a parent `~astropy.nddata.NDData` object.
    """

class NDUncertainty(metaclass=ABCMeta):
    """This is the metaclass for uncertainty classes used with `NDData`.

    Parameters
    ----------
    array : any type, optional
        The array or value (the parameter name is due to historical reasons) of
        the uncertainty. `numpy.ndarray`, `~astropy.units.Quantity` or
        `NDUncertainty` subclasses are recommended.
        If the `array` is `list`-like or `numpy.ndarray`-like it will be cast
        to a plain `numpy.ndarray`.
        Default is ``None``.

    unit : unit-like, optional
        Unit for the uncertainty ``array``. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    copy : `bool`, optional
        Indicates whether to save the `array` as a copy. ``True`` copies it
        before saving, while ``False`` tries to save every parameter as
        reference. Note however that it is not always possible to save the
        input as reference.
        Default is ``True``.

    Raises
    ------
    IncompatibleUncertaintiesException
        If given another `NDUncertainty`-like class as ``array`` if their
        ``uncertainty_type`` is different.
    """
    _unit: Incomplete
    def __init__(self, array: Incomplete | None = None, copy: bool = True, unit: Incomplete | None = None) -> None: ...
    @property
    @abstractmethod
    def uncertainty_type(self):
        """`str` : Short description of the type of uncertainty.

        Defined as abstract property so subclasses *have* to override this.
        """
    @property
    def supports_correlated(self):
        """`bool` : Supports uncertainty propagation with correlated uncertainties?

        .. versionadded:: 1.2
        """
    @property
    def array(self):
        """`numpy.ndarray` : the uncertainty's value."""
    _array: Incomplete
    @array.setter
    def array(self, value) -> None: ...
    @property
    def unit(self):
        """`~astropy.units.Unit` : The unit of the uncertainty, if any."""
    @unit.setter
    def unit(self, value) -> None:
        """
        The unit should be set to a value consistent with the parent NDData
        unit and the uncertainty type.
        """
    @property
    def quantity(self):
        """
        This uncertainty as an `~astropy.units.Quantity` object.
        """
    @property
    def parent_nddata(self):
        """`NDData` : reference to `NDData` instance with this uncertainty.

        In case the reference is not set uncertainty propagation will not be
        possible since propagation might need the uncertain data besides the
        uncertainty.
        """
    _parent_nddata: Incomplete
    @parent_nddata.setter
    def parent_nddata(self, value) -> None: ...
    @abstractmethod
    def _data_unit_to_uncertainty_unit(self, value):
        """
        Subclasses must override this property. It should take in a data unit
        and return the correct unit for the uncertainty given the uncertainty
        type.
        """
    def __repr__(self) -> str: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __getitem__(self, item):
        """Normal slicing on the array, keep the unit and return a reference."""
    def propagate(self, operation, other_nddata, result_data, correlation, axis: Incomplete | None = None):
        """Calculate the resulting uncertainty given an operation on the data.

        .. versionadded:: 1.2

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide` (or `numpy.divide`).

        other_nddata : `NDData` instance
            The second operand in the arithmetic operation.

        result_data : `~astropy.units.Quantity` or ndarray
            The result of the arithmetic operations on the data.

        correlation : `numpy.ndarray` or number
            The correlation (rho) is defined between the uncertainties in
            sigma_AB = sigma_A * sigma_B * rho. A value of ``0`` means
            uncorrelated operands.

        axis : int or tuple of ints, optional
            Axis over which to perform a collapsing operation.

        Returns
        -------
        resulting_uncertainty : `NDUncertainty` instance
            Another instance of the same `NDUncertainty` subclass containing
            the uncertainty of the result.

        Raises
        ------
        ValueError
            If the ``operation`` is not supported or if correlation is not zero
            but the subclass does not support correlated uncertainties.

        Notes
        -----
        First this method checks if a correlation is given and the subclass
        implements propagation with correlated uncertainties.
        Then the second uncertainty is converted (or an Exception is raised)
        to the same class in order to do the propagation.
        Then the appropriate propagation method is invoked and the result is
        returned.
        """
    def _convert_uncertainty(self, other_uncert):
        """Checks if the uncertainties are compatible for propagation.

        Checks if the other uncertainty is `NDUncertainty`-like and if so
        verify that the uncertainty_type is equal. If the latter is not the
        case try returning ``self.__class__(other_uncert)``.

        Parameters
        ----------
        other_uncert : `NDUncertainty` subclass
            The other uncertainty.

        Returns
        -------
        other_uncert : `NDUncertainty` subclass
            but converted to a compatible `NDUncertainty` subclass if
            possible and necessary.

        Raises
        ------
        IncompatibleUncertaintiesException:
            If the other uncertainty cannot be converted to a compatible
            `NDUncertainty` subclass.
        """
    @abstractmethod
    def _propagate_add(self, other_uncert, result_data, correlation): ...
    @abstractmethod
    def _propagate_subtract(self, other_uncert, result_data, correlation): ...
    @abstractmethod
    def _propagate_multiply(self, other_uncert, result_data, correlation): ...
    @abstractmethod
    def _propagate_divide(self, other_uncert, result_data, correlation): ...
    def represent_as(self, other_uncert):
        """Convert this uncertainty to a different uncertainty type.

        Parameters
        ----------
        other_uncert : `NDUncertainty` subclass
            The `NDUncertainty` subclass to convert to.

        Returns
        -------
        resulting_uncertainty : `NDUncertainty` instance
            An instance of ``other_uncert`` subclass containing the uncertainty
            converted to the new uncertainty type.

        Raises
        ------
        TypeError
            If either the initial or final subclasses do not support
            conversion, a `TypeError` is raised.
        """

class UnknownUncertainty(NDUncertainty):
    """This class implements any unknown uncertainty type.

    The main purpose of having an unknown uncertainty class is to prevent
    uncertainty propagation.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`
    """
    @property
    def supports_correlated(self):
        """`False` : Uncertainty propagation is *not* possible for this class."""
    @property
    def uncertainty_type(self):
        '''``"unknown"`` : `UnknownUncertainty` implements any unknown                            uncertainty type.
        '''
    def _data_unit_to_uncertainty_unit(self, value) -> None:
        """
        No way to convert if uncertainty is unknown.
        """
    def _convert_uncertainty(self, other_uncert) -> None:
        """Raise an Exception because unknown uncertainty types cannot
        implement propagation.
        """
    def _propagate_add(self, other_uncert, result_data, correlation) -> None:
        """Not possible for unknown uncertainty types."""
    def _propagate_subtract(self, other_uncert, result_data, correlation) -> None: ...
    def _propagate_multiply(self, other_uncert, result_data, correlation) -> None: ...
    def _propagate_divide(self, other_uncert, result_data, correlation) -> None: ...

class _VariancePropagationMixin:
    """
    Propagation of uncertainties for variances, also used to perform error
    propagation for variance-like uncertainties (standard deviation and inverse
    variance).
    """
    def _propagate_collapse(self, numpy_op, axis: Incomplete | None = None):
        """
        Error propagation for collapse operations on variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------
        numpy_op : function
            Numpy operation like `np.sum` or `np.max` to use in the collapse

        subtract : bool, optional
            If ``True``, propagate for subtraction, otherwise propagate for
            addition.

        axis : tuple, optional
            Axis on which to compute collapsing operations.
        """
    def _get_err_at_extremum(self, extremum, axis):
        """
        Return the value of the ``uncertainty`` array at the indices
        which satisfy the ``extremum`` function applied to the ``measurement`` array,
        where we expect ``extremum`` to be np.argmax or np.argmin, and
        we expect a two-dimensional output.

        Assumes the ``measurement`` and ``uncertainty`` array dimensions
        are ordered such that the zeroth dimension is the one to preserve.
        For example, if you start with array with shape (a, b, c), this
        function applies the ``extremum`` function to the last two dimensions,
        with shapes b and c.

        This operation is difficult to cast in a vectorized way. Here
        we implement it with a list comprehension, which is likely not the
        most performant solution.
        """
    def _propagate_add_sub(self, other_uncert, result_data, correlation, subtract: bool = False, to_variance=..., from_variance=...):
        """
        Error propagation for addition or subtraction of variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------
        other_uncert : `~astropy.nddata.NDUncertainty` instance
            The uncertainty, if any, of the other operand.

        result_data : `~astropy.nddata.NDData` instance
            The results of the operation on the data.

        correlation : float or array-like
            Correlation of the uncertainties.

        subtract : bool, optional
            If ``True``, propagate for subtraction, otherwise propagate for
            addition.

        to_variance : function, optional
            Function that will transform the input uncertainties to variance.
            The default assumes the uncertainty is the variance.

        from_variance : function, optional
            Function that will convert from variance to the input uncertainty.
            The default assumes the uncertainty is the variance.
        """
    def _propagate_multiply_divide(self, other_uncert, result_data, correlation, divide: bool = False, to_variance=..., from_variance=...):
        """
        Error propagation for multiplication or division of variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------
        other_uncert : `~astropy.nddata.NDUncertainty` instance
            The uncertainty, if any, of the other operand.

        result_data : `~astropy.nddata.NDData` instance
            The results of the operation on the data.

        correlation : float or array-like
            Correlation of the uncertainties.

        divide : bool, optional
            If ``True``, propagate for division, otherwise propagate for
            multiplication.

        to_variance : function, optional
            Function that will transform the input uncertainties to variance.
            The default assumes the uncertainty is the variance.

        from_variance : function, optional
            Function that will convert from variance to the input uncertainty.
            The default assumes the uncertainty is the variance.
        """

class StdDevUncertainty(_VariancePropagationMixin, NDUncertainty):
    """Standard deviation uncertainty assuming first order gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `StdDevUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will have the same unit as the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    `StdDevUncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, StdDevUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        StdDevUncertainty([0.1, 0.1, 0.1])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = StdDevUncertainty([0.2], unit='m', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        StdDevUncertainty([0.2])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 2
        >>> ndd.uncertainty
        StdDevUncertainty(2)

    .. note::
        The unit will not be displayed.
    """
    @property
    def supports_correlated(self):
        """`True` : `StdDevUncertainty` allows to propagate correlated                     uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
    @property
    def uncertainty_type(self):
        '''``"std"`` : `StdDevUncertainty` implements standard deviation.'''
    def _convert_uncertainty(self, other_uncert): ...
    def _propagate_add(self, other_uncert, result_data, correlation): ...
    def _propagate_subtract(self, other_uncert, result_data, correlation): ...
    def _propagate_multiply(self, other_uncert, result_data, correlation): ...
    def _propagate_divide(self, other_uncert, result_data, correlation): ...
    def _propagate_collapse(self, numpy_operation, axis): ...
    def _data_unit_to_uncertainty_unit(self, value): ...
    def _convert_to_variance(self): ...
    @classmethod
    def _convert_from_variance(cls, var_uncert): ...

class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
    """
    Variance uncertainty assuming first order Gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `VarianceUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will be the square of the unit of the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    Compare this example to that in `StdDevUncertainty`; the uncertainties
    in the examples below are equivalent to the uncertainties in
    `StdDevUncertainty`.

    `VarianceUncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, VarianceUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=VarianceUncertainty([0.01, 0.01, 0.01]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        VarianceUncertainty([0.01, 0.01, 0.01])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = VarianceUncertainty([0.04], unit='m^2', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        VarianceUncertainty([0.04])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 4
        >>> ndd.uncertainty
        VarianceUncertainty(4)

    .. note::
        The unit will not be displayed.
    """
    @property
    def uncertainty_type(self):
        '''``"var"`` : `VarianceUncertainty` implements variance.'''
    @property
    def supports_correlated(self):
        """`True` : `VarianceUncertainty` allows to propagate correlated                     uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
    def _propagate_add(self, other_uncert, result_data, correlation): ...
    def _propagate_subtract(self, other_uncert, result_data, correlation): ...
    def _propagate_multiply(self, other_uncert, result_data, correlation): ...
    def _propagate_divide(self, other_uncert, result_data, correlation): ...
    def _data_unit_to_uncertainty_unit(self, value): ...
    def _convert_to_variance(self): ...
    @classmethod
    def _convert_from_variance(cls, var_uncert): ...

class InverseVariance(_VariancePropagationMixin, NDUncertainty):
    """
    Inverse variance uncertainty assuming first order Gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `InverseVariance`. The class can handle if the uncertainty has a unit
    that differs from (but is convertible to) the parents `NDData` unit. The
    unit of the resulting uncertainty will the inverse square of the unit of
    the resulting data. Also support for correlation is possible but requires
    the correlation as input. It cannot handle correlation determination
    itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    Compare this example to that in `StdDevUncertainty`; the uncertainties
    in the examples below are equivalent to the uncertainties in
    `StdDevUncertainty`.

    `InverseVariance` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, InverseVariance
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=InverseVariance([100, 100, 100]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        InverseVariance([100, 100, 100])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = InverseVariance([25], unit='1/m^2', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        InverseVariance([25])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 0.25
        >>> ndd.uncertainty
        InverseVariance(0.25)

    .. note::
        The unit will not be displayed.
    """
    @property
    def uncertainty_type(self):
        '''``"ivar"`` : `InverseVariance` implements inverse variance.'''
    @property
    def supports_correlated(self):
        """`True` : `InverseVariance` allows to propagate correlated                     uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
    def _propagate_add(self, other_uncert, result_data, correlation): ...
    def _propagate_subtract(self, other_uncert, result_data, correlation): ...
    def _propagate_multiply(self, other_uncert, result_data, correlation): ...
    def _propagate_divide(self, other_uncert, result_data, correlation): ...
    def _data_unit_to_uncertainty_unit(self, value): ...
    def _convert_to_variance(self): ...
    @classmethod
    def _convert_from_variance(cls, var_uncert): ...
