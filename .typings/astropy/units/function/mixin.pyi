from _typeshed import Incomplete
from astropy.units.core import IrreducibleUnit as IrreducibleUnit, Unit as Unit

class FunctionMixin:
    """Mixin class that makes UnitBase subclasses callable.

    Provides a __call__ method that passes on arguments to a FunctionUnit.
    Instances of this class should define ``_function_unit_class`` pointing
    to the relevant class.

    See units.py and logarithmic.py for usage.
    """
    def __call__(self, unit: Incomplete | None = None): ...

class IrreducibleFunctionUnit(FunctionMixin, IrreducibleUnit): ...
class RegularFunctionUnit(FunctionMixin, Unit): ...
