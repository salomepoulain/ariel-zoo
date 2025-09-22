import abc
from _typeshed import Incomplete

__all__ = ['BasePeriodogram']

class BasePeriodogram(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, t, y, dy: Incomplete | None = None): ...
    @classmethod
    def from_timeseries(cls, timeseries, signal_column_name: Incomplete | None = None, uncertainty: Incomplete | None = None, **kwargs):
        """
        Initialize a periodogram from a time series object.

        If a binned time series is passed, the time at the center of the bins is
        used. Also note that this method automatically gets rid of NaN/undefined
        values when initializing the periodogram.

        Parameters
        ----------
        signal_column_name : str
            The name of the column containing the signal values to use.
        uncertainty : str or float or `~astropy.units.Quantity`, optional
            The name of the column containing the errors on the signal, or the
            value to use for the error, if a scalar.
        **kwargs
            Additional keyword arguments are passed to the initializer for this
            periodogram class.
        """
