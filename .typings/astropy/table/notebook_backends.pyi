from _typeshed import Incomplete

__all__ = ['classic', 'ipydatagrid']

def classic(table, tableid: Incomplete | None = None, css: Incomplete | None = None, display_length: int = 50, table_class: str = 'astropy-default', show_row_index: str = 'idx'):
    '''Render the table in HTML and show it in the Jupyter notebook.

    .. deprecated:: 6.1
       Use :func:`ipydatagrid` instead.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table to render.
    tableid : str or None
        An html ID tag for the table.  Default is ``table{id}-XXX``, where
        id is the unique integer id of the table object, id(table), and XXX
        is a random number to avoid conflicts when printing the same table
        multiple times.
    table_class : str or None
        A string with a list of HTML classes used to style the table.
        The special default string (\'astropy-default\') means that the string
        will be retrieved from the configuration item
        ``astropy.table.default_notebook_table_class``. Note that these
        table classes may make use of bootstrap, as this is loaded with the
        notebook.  See `this page <https://getbootstrap.com/css/#tables>`_
        for the list of classes.
    css : str
        A valid CSS string declaring the formatting for the table. Defaults
        to ``astropy.table.jsviewer.DEFAULT_CSS_NB``.
    display_length : int, optional
        Number or rows to show. Defaults to 50.
    show_row_index : str or False
        If this does not evaluate to False, a column with the given name
        will be added to the version of the table that gets displayed.
        This new column shows the index of the row in the table itself,
        even when the displayed table is re-sorted by another column. Note
        that if a column with this name already exists, this option will be
        ignored. Defaults to "idx".

    Returns
    -------
    html : object
        An ``IPython.display.HTML`` instance representing the given table.

    Notes
    -----
    Currently, unlike :meth:`~astropy.table.Table.show_in_browser`
    (with ``jsviewer=True``), this
    method needs to access online Javascript code repositories.  This is due
    to modern browsers\' limitations on accessing local files.  Hence, if you
    call this method while offline (and don\'t have a cached version of
    jquery and jquery.dataTables), you will not get the jsviewer features.

    '''
def ipydatagrid(table, **kwargs):
    """Render the table in HTML with ``ipydatagrid`` and show it in
    the Jupyter notebook.

    This function creates an ``ipydatagrid.DataGrid`` object by converting the input
    ``table`` to a ``pandas.DataFrame`` and passing ``**kwargs`` to the constructor.
    The available ``DataGrid`` options can be seen in a Jupyter notebook with
    ``help(ipydatagrid.DataGrid)``.

    .. note::
        This function requires optional dependencies ``pandas`` and ``ipydatagrid``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table to render.

    **kwargs : dict, optional
        Keyword arguments accepted by ``ipydatagrid.DataGrid``.

    Returns
    -------
    dg : object
        An ``ipydatagrid.DataGrid`` instance representing the given table.

    """
