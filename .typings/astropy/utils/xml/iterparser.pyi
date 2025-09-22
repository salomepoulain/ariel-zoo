from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['get_xml_iterator', 'get_xml_encoding', 'xml_readlines']

def get_xml_iterator(source, _debug_python_based_parser: bool = False) -> Generator[Incomplete]:
    """
    Returns an iterator over the elements of an XML file.

    The iterator doesn't ever build a tree, so it is much more memory
    and time efficient than the alternative in ``cElementTree``.

    Parameters
    ----------
    source : path-like, :term:`file-like (readable)`, or callable
        Handle that contains the data or function that reads it.
        If a function or callable object, it must directly read from a stream.
        Non-callable objects must define a ``read`` method.

    Returns
    -------
    parts : iterator

        The iterator returns 4-tuples (*start*, *tag*, *data*, *pos*):

            - *start*: when `True` is a start element event, otherwise
              an end element event.

            - *tag*: The name of the element

            - *data*: Depends on the value of *event*:

                - if *start* == `True`, data is a dictionary of
                  attributes

                - if *start* == `False`, data is a string containing
                  the text content of the element

            - *pos*: Tuple (*line*, *col*) indicating the source of the
              event.
    """
def get_xml_encoding(source):
    """
    Determine the encoding of an XML file by reading its header.

    Parameters
    ----------
    source : path-like, :term:`file-like (readable)`, or callable
        Handle that contains the data or function that reads it.
        If a function or callable object, it must directly read from a stream.
        Non-callable objects must define a ``read`` method.

    Returns
    -------
    encoding : str
    """
def xml_readlines(source):
    """
    Get the lines from a given XML file.  Correctly determines the
    encoding and always returns unicode.

    Parameters
    ----------
    source : path-like, :term:`file-like (readable)`, or callable
        Handle that contains the data or function that reads it.
        If a function or callable object, it must directly read from a stream.
        Non-callable objects must define a ``read`` method.

    Returns
    -------
    lines : list of unicode
    """
