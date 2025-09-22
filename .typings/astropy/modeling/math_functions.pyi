import abc
from astropy.modeling.core import Model

__all__ = ['SinUfunc', 'CosUfunc', 'TanUfunc', 'ArcsinUfunc', 'ArccosUfunc', 'ArctanUfunc', 'Arctan2Ufunc', 'HypotUfunc', 'SinhUfunc', 'CoshUfunc', 'TanhUfunc', 'ArcsinhUfunc', 'ArccoshUfunc', 'ArctanhUfunc', 'Deg2radUfunc', 'Rad2degUfunc', 'AddUfunc', 'SubtractUfunc', 'MultiplyUfunc', 'LogaddexpUfunc', 'Logaddexp2Ufunc', 'True_divideUfunc', 'Floor_divideUfunc', 'NegativeUfunc', 'PositiveUfunc', 'PowerUfunc', 'RemainderUfunc', 'FmodUfunc', 'DivmodUfunc', 'AbsoluteUfunc', 'FabsUfunc', 'RintUfunc', 'ExpUfunc', 'Exp2Ufunc', 'LogUfunc', 'Log2Ufunc', 'Log10Ufunc', 'Expm1Ufunc', 'Log1pUfunc', 'SqrtUfunc', 'SquareUfunc', 'CbrtUfunc', 'ReciprocalUfunc', 'DivideUfunc', 'ModUfunc']

class _NPUfuncModel(Model, metaclass=abc.ABCMeta):
    _is_dynamic: bool
    def __init__(self, **kwargs) -> None: ...

# Names in __all__ with no definition:
#   AbsoluteUfunc
#   AddUfunc
#   ArccosUfunc
#   ArccoshUfunc
#   ArcsinUfunc
#   ArcsinhUfunc
#   Arctan2Ufunc
#   ArctanUfunc
#   ArctanhUfunc
#   CbrtUfunc
#   CosUfunc
#   CoshUfunc
#   Deg2radUfunc
#   DivideUfunc
#   DivmodUfunc
#   Exp2Ufunc
#   ExpUfunc
#   Expm1Ufunc
#   FabsUfunc
#   Floor_divideUfunc
#   FmodUfunc
#   HypotUfunc
#   Log10Ufunc
#   Log1pUfunc
#   Log2Ufunc
#   LogUfunc
#   Logaddexp2Ufunc
#   LogaddexpUfunc
#   ModUfunc
#   MultiplyUfunc
#   NegativeUfunc
#   PositiveUfunc
#   PowerUfunc
#   Rad2degUfunc
#   ReciprocalUfunc
#   RemainderUfunc
#   RintUfunc
#   SinUfunc
#   SinhUfunc
#   SqrtUfunc
#   SquareUfunc
#   SubtractUfunc
#   TanUfunc
#   TanhUfunc
#   True_divideUfunc
