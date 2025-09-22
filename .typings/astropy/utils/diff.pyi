__all__ = ['diff_values', 'report_diff_values', 'where_not_allclose']

def diff_values(a, b, rtol: float = 0.0, atol: float = 0.0):
    """
    Diff two scalar values. If both values are floats, they are compared to
    within the given absolute and relative tolerance.

    Parameters
    ----------
    a, b : int, float, str
        Scalar values to compare.

    rtol, atol : float
        Relative and absolute tolerances as accepted by
        :func:`numpy.allclose`.

    Returns
    -------
    is_different : bool
        `True` if they are different, else `False`.

    """
def report_diff_values(a, b, fileobj=..., indent_width: int = 0, rtol: float = 0.0, atol: float = 0.0):
    """
    Write a diff report between two values to the specified file-like object.

    Parameters
    ----------
    a, b
        Values to compare. Anything that can be turned into strings
        and compared using :py:mod:`difflib` should work.

    fileobj : object
        File-like object to write to.
        The default is ``sys.stdout``, which writes to terminal.

    indent_width : int
        Character column(s) to indent.

    rtol, atol : float
        Relative and absolute tolerances as accepted by
        :func:`numpy.allclose`.

    Returns
    -------
    identical : bool
        `True` if no diff, else `False`.

    """
def where_not_allclose(a, b, rtol: float = 1e-05, atol: float = 1e-08):
    """
    A version of :func:`numpy.allclose` that returns the indices
    where the two arrays differ, instead of just a boolean value.

    Parameters
    ----------
    a, b : array-like
        Input arrays to compare.

    rtol, atol : float
        Relative and absolute tolerances as accepted by
        :func:`numpy.allclose`.

    Returns
    -------
    idx : tuple of array
        Indices where the two arrays differ.

    """
