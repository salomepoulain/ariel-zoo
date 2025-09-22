from _typeshed import Incomplete

_ns: Incomplete

def _initialize_module() -> None: ...
def _enable():
    """
    Enable the VOUnit-required extra units so they appear in results of
    `~astropy.units.UnitBase.find_equivalent_units` and
    `~astropy.units.UnitBase.compose`, and are recognized in the ``Unit('...')``
    idiom.
    """
