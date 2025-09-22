from _typeshed import Incomplete
from astropy.extern.ply.lex import Lexer
from astropy.extern.ply.yacc import LRParser

__all__ = ['lex', 'ThreadSafeParser', 'yacc']

def lex(lextab: str, package: str, reflags: int = ...) -> Lexer:
    """Create a lexer from local variables.

    It automatically compiles the lexer in optimized mode, writing to
    ``lextab`` in the same directory as the calling file.

    This function is thread-safe. The returned lexer is *not* thread-safe, but
    if it is used exclusively with a single parser returned by :func:`yacc`
    then it will be safe.

    It is only intended to work with lexers defined within the calling
    function, rather than at class or module scope.

    Parameters
    ----------
    lextab : str
        Name for the file to write with the generated tables, if it does not
        already exist (without ``.py`` suffix).
    package : str
        Name of a test package which should be run with pytest to regenerate
        the output file. This is inserted into a comment in the generated
        file.
    reflags : int
        Passed to ``ply.lex``.
    """

class ThreadSafeParser:
    """Wrap a parser produced by ``ply.yacc.yacc``.

    It provides a :meth:`parse` method that is thread-safe.
    """
    parser: Incomplete
    _lock: Incomplete
    def __init__(self, parser: LRParser) -> None: ...
    def parse(self, *args, **kwargs):
        """Run the wrapped parser, with a lock to ensure serialization."""

def yacc(tabmodule: str, package: str) -> ThreadSafeParser:
    """Create a parser from local variables.

    It automatically compiles the parser in optimized mode, writing to
    ``tabmodule`` in the same directory as the calling file.

    This function is thread-safe, and the returned parser is also thread-safe,
    provided that it does not share a lexer with any other parser.

    It is only intended to work with parsers defined within the calling
    function, rather than at class or module scope.

    Parameters
    ----------
    tabmodule : str
        Name for the file to write with the generated tables, if it does not
        already exist (without ``.py`` suffix).
    package : str
        Name of a test package which should be run with pytest to regenerate
        the output file. This is inserted into a comment in the generated
        file.
    """
