__all__ = ['unescape_all']

def unescape_all(url):
    """Recursively unescape a given URL.

    .. note:: '&amp;&amp;' becomes a single '&'.

    Parameters
    ----------
    url : str or bytes
        URL to unescape.

    Returns
    -------
    clean_url : str or bytes
        Unescaped URL.

    """
