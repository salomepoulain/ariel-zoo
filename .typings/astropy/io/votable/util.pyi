from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['convert_to_writable_filelike', 'stc_reference_frames', 'coerce_range_list_param']

def convert_to_writable_filelike(fd, compressed: bool = False) -> Generator[Incomplete]:
    """
    Returns a writable file-like object suitable for streaming output.

    Parameters
    ----------
    fd : str or file-like
        May be:

            - a file path string, in which case it is opened, and the file
              object is returned.

            - an object with a :meth:``write`` method, in which case that
              object is returned.

    compressed : bool, optional
        If `True`, create a gzip-compressed file.  (Default is `False`).

    Returns
    -------
    fd : :term:`file-like (writeable)`
    """

stc_reference_frames: Incomplete

def coerce_range_list_param(p, frames: Incomplete | None = None, numeric: bool = True):
    """
    Coerces and/or verifies the object *p* into a valid range-list-format parameter.

    As defined in `Section 8.7.2 of Simple
    Spectral Access Protocol
    <http://www.ivoa.net/documents/REC/DAL/SSA-20080201.html>`_.

    Parameters
    ----------
    p : str or sequence
        May be a string as passed verbatim to the service expecting a
        range-list, or a sequence.  If a sequence, each item must be
        either:

            - a numeric value

            - a named value, such as, for example, 'J' for named
              spectrum (if the *numeric* kwarg is False)

            - a 2-tuple indicating a range

            - the last item my be a string indicating the frame of
              reference

    frames : sequence of str, optional
        A sequence of acceptable frame of reference keywords.  If not
        provided, the default set in ``set_reference_frames`` will be
        used.

    numeric : bool, optional
        TODO

    Returns
    -------
    parts : tuple
        The result is a tuple:
            - a string suitable for passing to a service as a range-list
              argument

            - an integer counting the number of elements
    """
