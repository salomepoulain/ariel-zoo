from . import UFUNC_HELPERS as UFUNC_HELPERS
from .helpers import get_converter as get_converter, helper_cbrt as helper_cbrt, helper_dimensionless_to_dimensionless as helper_dimensionless_to_dimensionless, helper_two_arg_dimensionless as helper_two_arg_dimensionless
from _typeshed import Incomplete
from astropy.units.core import dimensionless_unscaled as dimensionless_unscaled
from astropy.units.errors import UnitTypeError as UnitTypeError, UnitsError as UnitsError

dimensionless_to_dimensionless_sps_ufuncs: Incomplete
scipy_special_ufuncs = dimensionless_to_dimensionless_sps_ufuncs
degree_to_dimensionless_sps_ufuncs: Incomplete
two_arg_dimensionless_sps_ufuncs: Incomplete

def helper_degree_to_dimensionless(f, unit): ...
def helper_degree_minute_second_to_radian(f, unit1, unit2, unit3): ...
def get_scipy_special_helpers(): ...
