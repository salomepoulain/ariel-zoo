__all__ = ['common_dtype']

def common_dtype(arrs):
    """
    Use numpy to find the common dtype for a list of ndarrays.

    Only allow arrays within the following fundamental numpy data types:
    ``np.bool_``, ``np.object_``, ``np.number``, ``np.character``, ``np.void``

    Parameters
    ----------
    arrs : list of ndarray
        Arrays for which to find the common dtype

    Returns
    -------
    dtype_str : str
        String representation of dytpe (dtype ``str`` attribute)
    """
