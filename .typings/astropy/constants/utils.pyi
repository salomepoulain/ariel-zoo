from _typeshed import Incomplete
from collections.abc import Generator

__all__: Incomplete

def _get_c(codata, iaudata, module, not_in_module_only: bool = True) -> Generator[Incomplete]:
    """
    Generator to return a Constant object.

    Parameters
    ----------
    codata, iaudata : obj
        Modules containing CODATA and IAU constants of interest.

    module : obj
        Namespace module of interest.

    not_in_module_only : bool
        If ``True``, ignore constants that are already in the
        namespace of ``module``.

    Returns
    -------
    _c : Constant
        Constant object to process.

    """
def _set_c(codata, iaudata, module, not_in_module_only: bool = True, doclines: Incomplete | None = None, set_class: bool = False) -> None:
    """
    Set constants in a given module namespace.

    Parameters
    ----------
    codata, iaudata : obj
        Modules containing CODATA and IAU constants of interest.

    module : obj
        Namespace module to modify with the given ``codata`` and ``iaudata``.

    not_in_module_only : bool
        If ``True``, constants that are already in the namespace
        of ``module`` will not be modified.

    doclines : list or None
        If a list is given, this list will be modified in-place to include
        documentation of modified constants. This can be used to update
        docstring of ``module``.

    set_class : bool
        Namespace of ``module`` is populated with ``_c.__class__``
        instead of just ``_c`` from :func:`_get_c`.

    """
