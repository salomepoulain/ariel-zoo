from _typeshed import Incomplete
from astropy.coordinates import Angle as Angle
from astropy.units import UnitsError as UnitsError

DMS_RE: Incomplete
HMS_RE: Incomplete
DDEC_RE: Incomplete
DMIN_RE: Incomplete
DSEC_RE: Incomplete
SCAL_RE: Incomplete
CUSTOM_UNITS: Incomplete

def _fix_minus(labels: list[str], /) -> list[str]: ...

class BaseFormatterLocator:
    """
    A joint formatter/locator.
    """
    _unit: Incomplete
    _format_unit: Incomplete
    format: Incomplete
    def __init__(self, values: Incomplete | None = None, number: Incomplete | None = None, spacing: Incomplete | None = None, format: Incomplete | None = None, unit: Incomplete | None = None, format_unit: Incomplete | None = None) -> None: ...
    @property
    def values(self): ...
    _number: Incomplete
    _spacing: Incomplete
    _values: Incomplete
    @values.setter
    def values(self, values) -> None: ...
    @property
    def number(self): ...
    @number.setter
    def number(self, number) -> None: ...
    @property
    def spacing(self): ...
    @spacing.setter
    def spacing(self, spacing) -> None: ...
    def minor_locator(self, spacing, frequency, value_min, value_max): ...
    @property
    def format_unit(self): ...
    @format_unit.setter
    def format_unit(self, unit) -> None: ...
    @staticmethod
    def _locate_values(value_min, value_max, spacing): ...

class AngleFormatterLocator(BaseFormatterLocator):
    """
    A joint formatter/locator.

    Parameters
    ----------
    number : int, optional
        Number of ticks.
    """
    _decimal: Incomplete
    _sep: Incomplete
    show_decimal_unit: Incomplete
    _alwayssign: bool
    def __init__(self, values: Incomplete | None = None, number: Incomplete | None = None, spacing: Incomplete | None = None, format: Incomplete | None = None, unit: Incomplete | None = None, decimal: Incomplete | None = None, format_unit: Incomplete | None = None, show_decimal_unit: bool = True) -> None: ...
    @property
    def decimal(self): ...
    @decimal.setter
    def decimal(self, value) -> None: ...
    @property
    def spacing(self): ...
    _number: Incomplete
    _spacing: Incomplete
    _values: Incomplete
    @spacing.setter
    def spacing(self, spacing) -> None: ...
    @property
    def sep(self): ...
    @sep.setter
    def sep(self, separator) -> None: ...
    @property
    def format(self): ...
    _format: Incomplete
    _format_unit: Incomplete
    _precision: Incomplete
    _fields: int
    @format.setter
    def format(self, value) -> None: ...
    @property
    def base_spacing(self): ...
    def locator(self, value_min, value_max): ...
    def formatter(self, values, spacing, format: str = 'auto'): ...

class ScalarFormatterLocator(BaseFormatterLocator):
    """
    A joint formatter/locator.
    """
    def __init__(self, values: Incomplete | None = None, number: Incomplete | None = None, spacing: Incomplete | None = None, format: Incomplete | None = None, unit: Incomplete | None = None, format_unit: Incomplete | None = None) -> None: ...
    @property
    def spacing(self): ...
    _number: Incomplete
    _spacing: Incomplete
    _values: Incomplete
    @spacing.setter
    def spacing(self, spacing) -> None: ...
    @property
    def format(self): ...
    _format: Incomplete
    _precision: Incomplete
    @format.setter
    def format(self, value) -> None: ...
    @property
    def base_spacing(self): ...
    def locator(self, value_min, value_max): ...
    def formatter(self, values, spacing, format: str = 'auto'): ...
