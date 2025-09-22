import astropy.config as _config
from . import connect as connect
from .bst import BST as BST
from .column import Column as Column, ColumnInfo as ColumnInfo, MaskedColumn as MaskedColumn, StringTruncateWarning as StringTruncateWarning
from .groups import ColumnGroups as ColumnGroups, TableGroups as TableGroups
from .jsviewer import JSViewer as JSViewer
from .operations import TableMergeError as TableMergeError, dstack as dstack, hstack as hstack, join as join, join_distance as join_distance, join_skycoord as join_skycoord, setdiff as setdiff, unique as unique, vstack as vstack
from .serialize import SerializedColumn as SerializedColumn, represent_mixins_as_columns as represent_mixins_as_columns
from .soco import SCEngine as SCEngine
from .sorted_array import SortedArray as SortedArray
from .table import NdarrayMixin as NdarrayMixin, PprintIncludeExclude as PprintIncludeExclude, QTable as QTable, Row as Row, Table as Table, TableAttribute as TableAttribute, TableColumns as TableColumns, TableFormatter as TableFormatter, TableReplaceWarning as TableReplaceWarning
from _typeshed import Incomplete
from astropy.io import registry as registry

__all__ = ['BST', 'Column', 'ColumnGroups', 'ColumnInfo', 'Conf', 'JSViewer', 'MaskedColumn', 'NdarrayMixin', 'QTable', 'Row', 'SCEngine', 'SerializedColumn', 'SortedArray', 'StringTruncateWarning', 'Table', 'TableAttribute', 'TableColumns', 'TableFormatter', 'TableGroups', 'TableMergeError', 'TableReplaceWarning', 'conf', 'connect', 'hstack', 'join', 'registry', 'represent_mixins_as_columns', 'setdiff', 'unique', 'vstack', 'dstack', 'conf', 'join_skycoord', 'join_distance', 'PprintIncludeExclude']

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.table`.
    """
    auto_colname: Incomplete
    default_notebook_table_class: Incomplete
    replace_warnings: Incomplete
    replace_inplace: Incomplete

conf: Incomplete
