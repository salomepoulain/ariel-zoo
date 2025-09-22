from _typeshed import Incomplete

__all__ = ['make_validation_report']

def make_validation_report(urls: Incomplete | None = None, destdir: str = 'astropy.io.votable.validator.results', multiprocess: bool = True, stilts: Incomplete | None = None) -> None:
    """
    Validates a large collection of web-accessible VOTable files.

    Generates a report as a directory tree of HTML files.

    Parameters
    ----------
    urls : list of str, optional
        If provided, is a list of HTTP urls to download VOTable files
        from.  If not provided, a built-in set of ~22,000 urls
        compiled by HEASARC will be used.

    destdir : path-like, optional
        The directory to write the report to.  By default, this is a
        directory called ``'results'`` in the current directory. If the
        directory does not exist, it will be created.

    multiprocess : bool, optional
        If `True` (default), perform validations in parallel using all
        of the cores on this machine.

    stilts : path-like, optional
        To perform validation with ``votlint`` from the Java-based |STILTS|
        VOTable parser, in addition to `astropy.io.votable`, set this to the
        path of the ``'stilts.jar'`` file.  ``java`` on the system shell
        path will be used to run it.

    Notes
    -----
    Downloads of each given URL will be performed only once and cached
    locally in *destdir*.  To refresh the cache, remove *destdir*
    first.
    """
