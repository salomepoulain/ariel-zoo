from .table import from_table as from_table, to_table as to_table
from _typeshed import Incomplete
from astropy.cosmology._typing import _CosmoT as _CosmoT
from astropy.cosmology.connect import readwrite_registry as readwrite_registry
from astropy.cosmology.core import Cosmology as Cosmology
from astropy.cosmology.parameter import Parameter as Parameter
from astropy.io.typing import PathLike as PathLike, ReadableFileLike as ReadableFileLike, WriteableFileLike as WriteableFileLike
from astropy.table import QTable as QTable, Table as Table
from typing import Any, TypeVar

_TableT = TypeVar('_TableT', 'Table')
_FORMAT_TABLE: Incomplete

def read_html_table(filename: PathLike | ReadableFileLike[Table], index: int | str | None = None, *, move_to_meta: bool = False, cosmology: str | type[_CosmoT] | None = None, latex_names: bool = True, **kwargs: Any) -> _CosmoT:
    '''Read a |Cosmology| from an HTML file.

    Parameters
    ----------
    filename : path-like or file-like
        From where to read the Cosmology.
    index : int or str or None, optional
        Needed to select the row in tables with multiple rows. ``index`` can be an
        integer for the row number or, if the table is indexed by a column, the value of
        that column. If the table is not indexed and ``index`` is a string, the "name"
        column is used as the indexing column.

    move_to_meta : bool, optional keyword-only
        Whether to move keyword arguments that are not in the Cosmology class\' signature
        to the Cosmology\'s metadata. This will only be applied if the Cosmology does NOT
        have a keyword-only argument (e.g. ``**kwargs``). Arguments moved to the
        metadata will be merged with existing metadata, preferring specified metadata in
        the case of a merge conflict (e.g. for ``Cosmology(meta={\'key\':10}, key=42)``,
        the ``Cosmology.meta`` will be ``{\'key\': 10}``).
    cosmology : str or |Cosmology| class or None, optional keyword-only
        The cosmology class (or string name thereof) to use when constructing the
        cosmology instance. The class also provides default parameter values, filling in
        any non-mandatory arguments missing in \'table\'.
    latex_names : bool, optional keyword-only
        Whether the |Table| (might) have latex column names for the parameters that need
        to be mapped to the correct parameter name -- e.g. $$H_{0}$$ to \'H0\'. This is
        `True` by default, but can be turned off (set to `False`) if there is a known
        name conflict (e.g. both an \'H0\' and \'$$H_{0}$$\' column) as this will raise an
        error. In this case, the correct name (\'H0\') is preferred.
    **kwargs : Any
        Passed to ``QTable.read``. ``format`` is set to \'ascii.html\', regardless of
        input.

    Returns
    -------
    |Cosmology| subclass instance

    Raises
    ------
    ValueError
        If the keyword argument \'format\' is given and is not "ascii.html".
    '''
def write_html_table(cosmology: Cosmology, file: PathLike | WriteableFileLike[_TableT], *, overwrite: bool = False, cls: type[_TableT] = ..., latex_names: bool = False, **kwargs: Any) -> None:
    '''Serialize the |Cosmology| into a HTML table.

    Parameters
    ----------
    cosmology : |Cosmology| subclass instance
        The cosmology to serialize.
    file : path-like or file-like
        Where to write the html table.

    overwrite : bool, optional keyword-only
        Whether to overwrite the file, if it exists.
    cls : |Table| class, optional keyword-only
        Astropy |Table| (sub)class to use when writing. Default is |QTable| class.
    latex_names : bool, optional keyword-only
        Whether to format the parameters (column) names to latex -- e.g. \'H0\' to
        $$H_{0}$$.
    **kwargs : Any
        Passed to ``cls.write``.

    Raises
    ------
    TypeError
        If the optional keyword-argument \'cls\' is not a subclass of |Table|.
    ValueError
        If the keyword argument \'format\' is given and is not "ascii.html".

    Examples
    --------
    We assume the following setup:

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> temp_dir = TemporaryDirectory()

    Writing a cosmology to a html file will produce a table with the cosmology\'s type,
    name, and parameters as columns.

        >>> from astropy.cosmology import Planck18
        >>> file = Path(temp_dir.name) / "file.html"
        >>> Planck18.write(file, overwrite=True)
        >>> with open(file) as f: print(f.read())
        <html>
        <head>
        <meta charset="utf-8"/>
        <meta content="text/html;charset=UTF-8" http-equiv="Content-type"/>
        </head>
        <body>
        <table>
        <thead>
        <tr>
            <th>cosmology</th> <th>name</th> <th>H0</th> <th>Om0</th> <th>Tcmb0</th>
            <th>Neff</th> <th colspan="3">m_nu</th> <th>Ob0</th>
        </tr>
        </thead>
        <tr>
            <td>FlatLambdaCDM</td> <td>Planck18</td> <td>67.66</td> <td>0.30966</td>
            <td>2.7255</td> <td>3.046</td> <td>0.0</td> <td>0.0</td> <td>0.06</td>
            <td>0.04897</td>
        </tr>
        </table>
        </body>
        </html>
        <BLANKLINE>
        <BLANKLINE>

    The cosmology\'s metadata is not included in the file.

    To save the cosmology in an existing file, use ``overwrite=True``; otherwise, an
    error will be raised.

        >>> Planck18.write(file, overwrite=True)

    To use a different table class as the underlying writer, use the ``cls`` kwarg. For
    more information on the available table classes, see the documentation on Astropy\'s
    table classes and on ``Cosmology.to_format("astropy.table")``.

    By default the parameter names are not converted to LaTeX / MathJax format. To
    enable this, set ``latex_names=True``.

        >>> file = Path(temp_dir.name) / "file2.html"
        >>> Planck18.write(file, latex_names=True)
        >>> with open(file) as f: print(f.read())
        <html>
        ...
        <thead>
            <tr>
            <th>cosmology</th>
            <th>name</th>
            <th>$$H_{0}$$</th>
            <th>$$\\Omega_{m,0}$$</th>
            <th>$$T_{0}$$</th>
            <th>$$N_{eff}$$</th>
            <th colspan="3">$$m_{nu}$$</th>
            <th>$$\\Omega_{b,0}$$</th>
            </tr>
        ...

    .. testcleanup::

        >>> temp_dir.cleanup()

    Notes
    -----
    A HTML file containing a Cosmology HTML table should have scripts enabling MathJax.

    .. code-block:: html

        <script type="text/javascript" id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
        </script>
    '''
def html_identify(origin: object, filepath: object, *args: object, **kwargs: object) -> bool:
    """Identify if an object uses the HTML Table format.

    Parameters
    ----------
    origin : object
        Not used.
    filepath : object
        From where to read the Cosmology.
    *args : object
        Not used.
    **kwargs : object
        Not used.

    Returns
    -------
    bool
        If the filepath is a string ending with '.html'.
    """
