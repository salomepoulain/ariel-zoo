from _typeshed import Incomplete

_ns: Incomplete

def _initialize_module() -> None: ...
def enable():
    """
    Enable deprecated units so they appear in results of
    `~astropy.units.UnitBase.find_equivalent_units` and
    `~astropy.units.UnitBase.compose`.

    This may be used with the ``with`` statement to enable deprecated
    units only temporarily.
    """
