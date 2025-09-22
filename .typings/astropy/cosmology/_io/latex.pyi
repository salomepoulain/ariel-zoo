from .table import to_table as to_table
from _typeshed import Incomplete
from astropy.cosmology.connect import readwrite_registry as readwrite_registry
from astropy.cosmology.core import Cosmology as Cosmology
from astropy.cosmology.parameter import Parameter as Parameter
from astropy.io.typing import PathLike as PathLike, WriteableFileLike as WriteableFileLike
from astropy.table import QTable as QTable, Table as Table
from typing import Any, TypeVar

_TableT = TypeVar('_TableT', Table)
_FORMAT_TABLE: Incomplete

def write_latex(cosmology: Cosmology, file: PathLike | WriteableFileLike[_TableT], *, overwrite: bool = False, cls: type[_TableT] = ..., latex_names: bool = True, **kwargs: Any) -> None:
    '''Serialize the |Cosmology| into a LaTeX.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.Cosmology` subclass instance
        The cosmology to serialize.
    file : path-like or file-like
        Location to save the serialized cosmology.

    overwrite : bool
        Whether to overwrite the file, if it exists.
    cls : type, optional keyword-only
        Astropy :class:`~astropy.table.Table` (sub)class to use when writing. Default is
        :class:`~astropy.table.QTable`.
    latex_names : bool, optional keyword-only
        Whether to use LaTeX names for the parameters. Default is `True`.
    **kwargs
        Passed to ``cls.write``

    Raises
    ------
    TypeError
        If kwarg (optional) \'cls\' is not a subclass of `astropy.table.Table`

    Examples
    --------
    We assume the following setup:

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> temp_dir = TemporaryDirectory()

    Writing a cosmology to a LaTeX file will produce a table with the cosmology\'s type,
    name, and parameters as columns.

        >>> from astropy.cosmology import Planck18
        >>> file = Path(temp_dir.name) / "file.tex"

        >>> Planck18.write(file, format="ascii.latex")
        >>> with open(file) as f: print(f.read())
        \\begin{table}
        \\begin{tabular}{cccccccc}
        cosmology & name & $H_0$ & $\\Omega_{m,0}$ & $T_{0}$ & $N_{eff}$ & $m_{nu}$ & $\\Omega_{b,0}$ \\\\\n        &  & $\\mathrm{km\\,Mpc^{-1}\\,s^{-1}}$ &  & $\\mathrm{K}$ &  & $\\mathrm{eV}$ &  \\\\\n        FlatLambdaCDM & Planck18 & 67.66 & 0.30966 & 2.7255 & 3.046 & 0.0 .. 0.06 & 0.04897 \\\\\n        \\end{tabular}
        \\end{table}
        <BLANKLINE>

    The cosmology\'s metadata is not included in the table.

    To save the cosmology in an existing file, use ``overwrite=True``; otherwise, an
    error will be raised.

        >>> Planck18.write(file, format="ascii.latex", overwrite=True)

    To use a different table class as the underlying writer, use the ``cls`` kwarg. For
    more information on the available table classes, see the documentation on Astropy\'s
    table classes and on ``Cosmology.to_format("astropy.table")``.

    By default the parameter names are converted to LaTeX format. To disable this, set
    ``latex_names=False``.

        >>> file = Path(temp_dir.name) / "file2.tex"
        >>> Planck18.write(file, format="ascii.latex", latex_names=False)
        >>> with open(file) as f: print(f.read())
        \\begin{table}
        \\begin{tabular}{cccccccc}
        cosmology & name & H0 & Om0 & Tcmb0 & Neff & m_nu & Ob0 \\\\\n        &  & $\\mathrm{km\\,Mpc^{-1}\\,s^{-1}}$ &  & $\\mathrm{K}$ &  & $\\mathrm{eV}$ &  \\\\\n        FlatLambdaCDM & Planck18 & 67.66 & 0.30966 & 2.7255 & 3.046 & 0.0 .. 0.06 & 0.04897 \\\\\n        \\end{tabular}
        \\end{table}
        <BLANKLINE>

    .. testcleanup::

        >>> temp_dir.cleanup()
    '''
def latex_identify(origin: object, filepath: str | None, *args: object, **kwargs: object) -> bool:
    """Identify if object uses the Table format.

    Returns
    -------
    bool
    """
