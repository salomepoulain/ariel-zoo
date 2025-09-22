import astropy.config as _config
from .table import Table as Table
from _typeshed import Incomplete
from astropy import extern as extern
from astropy.utils.exceptions import AstropyDeprecationWarning as AstropyDeprecationWarning

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.table.jsviewer`.
    """
    jquery_url: Incomplete
    datatables_url: Incomplete
    css_urls: Incomplete

conf: Incomplete
_TABLE_DIR: Incomplete
EXTERN_JS_DIR: Incomplete
EXTERN_CSS_DIR: Incomplete
_SORTING_SCRIPT_PART_1: str
_SORTING_SCRIPT_PART_2: str
IPYNB_JS_SCRIPT: Incomplete
HTML_JS_SCRIPT: Incomplete
DEFAULT_CSS: str
DEFAULT_CSS_NB: str

class JSViewer:
    """Provides an interactive HTML export of a Table.

    This class provides an interface to the `DataTables
    <https://datatables.net/>`_ library, which allow to visualize interactively
    an HTML table. It is used by the `~astropy.table.Table.show_in_browser`
    method.

    Parameters
    ----------
    use_local_files : bool, optional
        Use local files or a CDN for JavaScript libraries. Default False.
    display_length : int, optional
        Number or rows to show. Default to 50.

    """
    _use_local_files: Incomplete
    display_length_menu: Incomplete
    display_length: Incomplete
    def __init__(self, use_local_files: bool = False, display_length: int = 50) -> None: ...
    @property
    def jquery_urls(self): ...
    @property
    def css_urls(self): ...
    def _jstable_file(self): ...
    def ipynb(self, table_id, css: Incomplete | None = None, sort_columns: str = '[]'): ...
    def html_js(self, table_id: str = 'table0', sort_columns: str = '[]'): ...

def write_table_jsviewer(table, filename, table_id: Incomplete | None = None, max_lines: int = 5000, table_class: str = 'display compact', jskwargs: Incomplete | None = None, css=..., htmldict: Incomplete | None = None, overwrite: bool = False) -> None: ...
