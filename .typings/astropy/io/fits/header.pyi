import collections
from ._utils import parse_header as parse_header
from .card import Card as Card, KEYWORD_LENGTH as KEYWORD_LENGTH, UNDEFINED as UNDEFINED, _pad as _pad
from .file import _File as _File
from .util import decode_ascii as decode_ascii, encode_ascii as encode_ascii, fileobj_closed as fileobj_closed, fileobj_is_binary as fileobj_is_binary, path_like as path_like
from _typeshed import Incomplete
from astropy.utils import isiterable as isiterable
from astropy.utils.exceptions import AstropyUserWarning as AstropyUserWarning
from collections.abc import Generator

BLOCK_SIZE: int
HEADER_END_RE: Incomplete
VALID_HEADER_CHARS: Incomplete
END_CARD: Incomplete
_commentary_keywords: Incomplete
__doctest_skip__: Incomplete

class Header:
    """
    FITS header class.  This class exposes both a dict-like interface and a
    list-like interface to FITS headers.

    The header may be indexed by keyword and, like a dict, the associated value
    will be returned.  When the header contains cards with duplicate keywords,
    only the value of the first card with the given keyword will be returned.
    It is also possible to use a 2-tuple as the index in the form (keyword,
    n)--this returns the n-th value with that keyword, in the case where there
    are duplicate keywords.

    For example::

        >>> header['NAXIS']
        0
        >>> header[('FOO', 1)]  # Return the value of the second FOO keyword
        'foo'

    The header may also be indexed by card number::

        >>> header[0]  # Return the value of the first card in the header
        'T'

    Commentary keywords such as HISTORY and COMMENT are special cases: When
    indexing the Header object with either 'HISTORY' or 'COMMENT' a list of all
    the HISTORY/COMMENT values is returned::

        >>> header['HISTORY']
        This is the first history entry in this header.
        This is the second history entry in this header.
        ...

    See the Astropy documentation for more details on working with headers.

    Notes
    -----
    Although FITS keywords must be exclusively upper case, retrieving an item
    in a `Header` object is case insensitive.
    """
    def __init__(self, cards=[], copy: bool = False) -> None:
        """
        Construct a `Header` from an iterable and/or text file.

        Parameters
        ----------
        cards : list of `Card`, optional
            The cards to initialize the header with. Also allowed are other
            `Header` (or `dict`-like) objects.

            .. versionchanged:: 1.2
                Allowed ``cards`` to be a `dict`-like object.

        copy : bool, optional

            If ``True`` copies the ``cards`` if they were another `Header`
            instance.
            Default is ``False``.

            .. versionadded:: 1.3
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, keyword) -> bool: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other):
        """
        Two Headers are equal only if they have the exact same string
        representation.
        """
    def __add__(self, other): ...
    def __iadd__(self, other): ...
    def _ipython_key_completions_(self): ...
    @property
    def cards(self):
        """
        The underlying physical cards that make up this Header; it can be
        looked at, but it should not be modified directly.
        """
    @property
    def comments(self):
        """
        View the comments associated with each keyword, if any.

        For example, to see the comment on the NAXIS keyword:

            >>> header.comments['NAXIS']
            number of data axes

        Comments can also be updated through this interface:

            >>> header.comments['NAXIS'] = 'Number of data axes'

        """
    @property
    def _modified(self):
        """
        Whether or not the header has been modified; this is a property so that
        it can also check each card for modifications--cards may have been
        modified directly without the header containing it otherwise knowing.
        """
    @_modified.setter
    def _modified(self, val) -> None: ...
    @classmethod
    def fromstring(cls, data, sep: str = ''):
        '''
        Creates an HDU header from a byte string containing the entire header
        data.

        Parameters
        ----------
        data : str or bytes
           String or bytes containing the entire header.  In the case of bytes
           they will be decoded using latin-1 (only plain ASCII characters are
           allowed in FITS headers but latin-1 allows us to retain any invalid
           bytes that might appear in malformatted FITS files).

        sep : str, optional
            The string separating cards from each other, such as a newline.  By
            default there is no card separator (as is the case in a raw FITS
            file).  In general this is only used in cases where a header was
            printed as text (e.g. with newlines after each card) and you want
            to create a new `Header` from it by copy/pasting.

        Examples
        --------
        >>> from astropy.io.fits import Header
        >>> hdr = Header({\'SIMPLE\': True})
        >>> Header.fromstring(hdr.tostring()) == hdr
        True

        If you want to create a `Header` from printed text it\'s not necessary
        to have the exact binary structure as it would appear in a FITS file,
        with the full 80 byte card length.  Rather, each "card" can end in a
        newline and does not have to be padded out to a full card length as
        long as it "looks like" a FITS header:

        >>> hdr = Header.fromstring("""\\\n        ... SIMPLE  =                    T / conforms to FITS standard
        ... BITPIX  =                    8 / array data type
        ... NAXIS   =                    0 / number of array dimensions
        ... EXTEND  =                    T
        ... """, sep=\'\\n\')
        >>> hdr[\'SIMPLE\']
        True
        >>> hdr[\'BITPIX\']
        8
        >>> len(hdr)
        4

        Returns
        -------
        `Header`
            A new `Header` instance.
        '''
    @classmethod
    def fromfile(cls, fileobj, sep: str = '', endcard: bool = True, padding: bool = True):
        """
        Similar to :meth:`Header.fromstring`, but reads the header string from
        a given file-like object or filename.

        Parameters
        ----------
        fileobj : str, file-like
            A filename or an open file-like object from which a FITS header is
            to be read.  For open file handles the file pointer must be at the
            beginning of the header.

        sep : str, optional
            The string separating cards from each other, such as a newline.  By
            default there is no card separator (as is the case in a raw FITS
            file).

        endcard : bool, optional
            If True (the default) the header must end with an END card in order
            to be considered valid.  If an END card is not found an
            `OSError` is raised.

        padding : bool, optional
            If True (the default) the header will be required to be padded out
            to a multiple of 2880, the FITS header block size.  Otherwise any
            padding, or lack thereof, is ignored.

        Returns
        -------
        `Header`
            A new `Header` instance.
        """
    @classmethod
    def _fromcards(cls, cards): ...
    @classmethod
    def _from_blocks(cls, block_iter, is_binary, sep, endcard, padding):
        """
        The meat of `Header.fromfile`; in a separate method so that
        `Header.fromfile` itself is just responsible for wrapping file
        handling.  Also used by `_BaseHDU.fromstring`.

        ``block_iter`` should be a callable which, given a block size n
        (typically 2880 bytes as used by the FITS standard) returns an iterator
        of byte strings of that block size.

        ``is_binary`` specifies whether the returned blocks are bytes or text

        Returns both the entire header *string*, and the `Header` object
        returned by Header.fromstring on that string.
        """
    @classmethod
    def _find_end_card(cls, block, card_len):
        """
        Utility method to search a header block for the END card and handle
        invalid END cards.

        This method can also returned a modified copy of the input header block
        in case an invalid end card needs to be sanitized.
        """
    def tostring(self, sep: str = '', endcard: bool = True, padding: bool = True):
        """
        Returns a string representation of the header.

        By default this uses no separator between cards, adds the END card, and
        pads the string with spaces to the next multiple of 2880 bytes.  That
        is, it returns the header exactly as it would appear in a FITS file.

        Parameters
        ----------
        sep : str, optional
            The character or string with which to separate cards.  By default
            there is no separator, but one could use ``'\\\\n'``, for example, to
            separate each card with a new line

        endcard : bool, optional
            If True (default) adds the END card to the end of the header
            string

        padding : bool, optional
            If True (default) pads the string with spaces out to the next
            multiple of 2880 characters

        Returns
        -------
        str
            A string representing a FITS header.
        """
    def tofile(self, fileobj, sep: str = '', endcard: bool = True, padding: bool = True, overwrite: bool = False) -> None:
        """
        Writes the header to file or file-like object.

        By default this writes the header exactly as it would be written to a
        FITS file, with the END card included and padding to the next multiple
        of 2880 bytes.  However, aspects of this may be controlled.

        Parameters
        ----------
        fileobj : path-like or file-like, optional
            Either the pathname of a file, or an open file handle or file-like
            object.

        sep : str, optional
            The character or string with which to separate cards.  By default
            there is no separator, but one could use ``'\\\\n'``, for example, to
            separate each card with a new line

        endcard : bool, optional
            If `True` (default) adds the END card to the end of the header
            string

        padding : bool, optional
            If `True` (default) pads the string with spaces out to the next
            multiple of 2880 characters

        overwrite : bool, optional
            If ``True``, overwrite the output file if it exists. Raises an
            ``OSError`` if ``False`` and the output file exists. Default is
            ``False``.
        """
    @classmethod
    def fromtextfile(cls, fileobj, endcard: bool = False):
        """
        Read a header from a simple text file or file-like object.

        Equivalent to::

            >>> Header.fromfile(fileobj, sep='\\n', endcard=False,
            ...                 padding=False)

        See Also
        --------
        fromfile
        """
    def totextfile(self, fileobj, endcard: bool = False, overwrite: bool = False) -> None:
        """
        Write the header as text to a file or a file-like object.

        Equivalent to::

            >>> Header.tofile(fileobj, sep='\\n', endcard=False,
            ...               padding=False, overwrite=overwrite)

        See Also
        --------
        tofile
        """
    _cards: Incomplete
    _keyword_indices: Incomplete
    _rvkc_indices: Incomplete
    def clear(self) -> None:
        """
        Remove all cards from the header.
        """
    def copy(self, strip: bool = False):
        """
        Make a copy of the :class:`Header`.

        .. versionchanged:: 1.3
            `copy.copy` and `copy.deepcopy` on a `Header` will call this
            method.

        Parameters
        ----------
        strip : bool, optional
            If `True`, strip any headers that are specific to one of the
            standard HDU types, so that this header can be used in a different
            HDU.

        Returns
        -------
        `Header`
            A new :class:`Header` instance.
        """
    def __copy__(self): ...
    def __deepcopy__(self, *args, **kwargs): ...
    @classmethod
    def fromkeys(cls, iterable, value: Incomplete | None = None):
        """
        Similar to :meth:`dict.fromkeys`--creates a new `Header` from an
        iterable of keywords and an optional default value.

        This method is not likely to be particularly useful for creating real
        world FITS headers, but it is useful for testing.

        Parameters
        ----------
        iterable
            Any iterable that returns strings representing FITS keywords.

        value : optional
            A default value to assign to each keyword; must be a valid type for
            FITS keywords.

        Returns
        -------
        `Header`
            A new `Header` instance.
        """
    def get(self, key, default: Incomplete | None = None):
        """
        Similar to :meth:`dict.get`--returns the value associated with keyword
        in the header, or a default value if the keyword is not found.

        Parameters
        ----------
        key : str
            A keyword that may or may not be in the header.

        default : optional
            A default value to return if the keyword is not found in the
            header.

        Returns
        -------
        value: str, number, complex, bool, or ``astropy.io.fits.card.Undefined``
            The value associated with the given keyword, or the default value
            if the keyword is not in the header.
        """
    def set(self, keyword, value: Incomplete | None = None, comment: Incomplete | None = None, before: Incomplete | None = None, after: Incomplete | None = None) -> None:
        """
        Set the value and/or comment and/or position of a specified keyword.

        If the keyword does not already exist in the header, a new keyword is
        created in the specified position, or appended to the end of the header
        if no position is specified.

        This method is similar to :meth:`Header.update` prior to Astropy v0.1.

        .. note::
            It should be noted that ``header.set(keyword, value)`` and
            ``header.set(keyword, value, comment)`` are equivalent to
            ``header[keyword] = value`` and
            ``header[keyword] = (value, comment)`` respectively.

        Parameters
        ----------
        keyword : str
            A header keyword

        value : str, optional
            The value to set for the given keyword; if None the existing value
            is kept, but '' may be used to set a blank value

        comment : str, optional
            The comment to set for the given keyword; if None the existing
            comment is kept, but ``''`` may be used to set a blank comment

        before : str, int, optional
            Name of the keyword, or index of the `Card` before which this card
            should be located in the header.  The argument ``before`` takes
            precedence over ``after`` if both specified.

        after : str, int, optional
            Name of the keyword, or index of the `Card` after which this card
            should be located in the header.

        """
    def items(self) -> Generator[Incomplete]:
        """Like :meth:`dict.items`."""
    def keys(self) -> Generator[Incomplete]:
        """
        Like :meth:`dict.keys`--iterating directly over the `Header`
        instance has the same behavior.
        """
    def values(self) -> Generator[Incomplete]:
        """Like :meth:`dict.values`."""
    def pop(self, *args):
        """
        Works like :meth:`list.pop` if no arguments or an index argument are
        supplied; otherwise works like :meth:`dict.pop`.
        """
    def popitem(self):
        """Similar to :meth:`dict.popitem`."""
    def setdefault(self, key, default: Incomplete | None = None):
        """Similar to :meth:`dict.setdefault`."""
    def update(self, *args, **kwargs) -> None:
        """
        Update the Header with new keyword values, updating the values of
        existing keywords and appending new keywords otherwise; similar to
        `dict.update`.

        `update` accepts either a dict-like object or an iterable.  In the
        former case the keys must be header keywords and the values may be
        either scalar values or (value, comment) tuples.  In the case of an
        iterable the items must be (keyword, value) tuples or (keyword, value,
        comment) tuples.

        Arbitrary arguments are also accepted, in which case the update() is
        called again with the kwargs dict as its only argument.  That is,

        ::

            >>> header.update(NAXIS1=100, NAXIS2=100)

        is equivalent to::

            header.update({'NAXIS1': 100, 'NAXIS2': 100})

        """
    def append(self, card: Incomplete | None = None, useblanks: bool = True, bottom: bool = False, end: bool = False) -> None:
        """
        Appends a new keyword+value card to the end of the Header, similar
        to `list.append`.

        By default if the last cards in the Header have commentary keywords,
        this will append the new keyword before the commentary (unless the new
        keyword is also commentary).

        Also differs from `list.append` in that it can be called with no
        arguments: In this case a blank card is appended to the end of the
        Header.  In the case all the keyword arguments are ignored.

        Parameters
        ----------
        card : str, tuple
            A keyword or a (keyword, value, [comment]) tuple representing a
            single header card; the comment is optional in which case a
            2-tuple may be used

        useblanks : bool, optional
            If there are blank cards at the end of the Header, replace the
            first blank card so that the total number of cards in the Header
            does not increase.  Otherwise preserve the number of blank cards.

        bottom : bool, optional
            If True, instead of appending after the last non-commentary card,
            append after the last non-blank card.

        end : bool, optional
            If True, ignore the useblanks and bottom options, and append at the
            very end of the Header.

        """
    def extend(self, cards, strip: bool = True, unique: bool = False, update: bool = False, update_first: bool = False, useblanks: bool = True, bottom: bool = False, end: bool = False) -> None:
        """
        Appends multiple keyword+value cards to the end of the header, similar
        to `list.extend`.

        Parameters
        ----------
        cards : iterable
            An iterable of (keyword, value, [comment]) tuples; see
            `Header.append`.

        strip : bool, optional
            Remove any keywords that have meaning only to specific types of
            HDUs, so that only more general keywords are added from extension
            Header or Card list (default: `True`).

        unique : bool, optional
            If `True`, ensures that no duplicate keywords are appended;
            keywords already in this header are simply discarded.  The
            exception is commentary keywords (COMMENT, HISTORY, etc.): they are
            only treated as duplicates if their values match.

        update : bool, optional
            If `True`, update the current header with the values and comments
            from duplicate keywords in the input header.  This supersedes the
            ``unique`` argument.  Commentary keywords are treated the same as
            if ``unique=True``.

        update_first : bool, optional
            If the first keyword in the header is 'SIMPLE', and the first
            keyword in the input header is 'XTENSION', the 'SIMPLE' keyword is
            replaced by the 'XTENSION' keyword.  Likewise if the first keyword
            in the header is 'XTENSION' and the first keyword in the input
            header is 'SIMPLE', the 'XTENSION' keyword is replaced by the
            'SIMPLE' keyword.  This behavior is otherwise dumb as to whether or
            not the resulting header is a valid primary or extension header.
            This is mostly provided to support backwards compatibility with the
            old ``Header.fromTxtFile`` method, and only applies if
            ``update=True``.

        useblanks, bottom, end : bool, optional
            These arguments are passed to :meth:`Header.append` while appending
            new cards to the header.
        """
    def count(self, keyword):
        """
        Returns the count of the given keyword in the header, similar to
        `list.count` if the Header object is treated as a list of keywords.

        Parameters
        ----------
        keyword : str
            The keyword to count instances of in the header

        """
    def index(self, keyword, start: Incomplete | None = None, stop: Incomplete | None = None):
        """
        Returns the index if the first instance of the given keyword in the
        header, similar to `list.index` if the Header object is treated as a
        list of keywords.

        Parameters
        ----------
        keyword : str
            The keyword to look up in the list of all keywords in the header

        start : int, optional
            The lower bound for the index

        stop : int, optional
            The upper bound for the index

        """
    def insert(self, key, card, useblanks: bool = True, after: bool = False) -> None:
        '''
        Inserts a new keyword+value card into the Header at a given location,
        similar to `list.insert`.

        New keywords can also be inserted relative to existing keywords
        using, for example::

            >>> header = Header({"NAXIS1": 10})
            >>> header.insert(\'NAXIS1\', (\'NAXIS\', 2, \'Number of axes\'))

        to insert before an existing keyword, or::

            >>> header.insert(\'NAXIS1\', (\'NAXIS2\', 4096), after=True)

        to insert after an existing keyword.

        Parameters
        ----------
        key : int, str, or tuple
            The index into the list of header keywords before which the
            new keyword should be inserted, or the name of a keyword before
            which the new keyword should be inserted.  Can also accept a
            (keyword, index) tuple for inserting around duplicate keywords.

        card : str, tuple
            A keyword or a (keyword, value, [comment]) tuple; see
            `Header.append`

        useblanks : bool, optional
            If there are blank cards at the end of the Header, replace the
            first blank card so that the total number of cards in the Header
            does not increase.  Otherwise preserve the number of blank cards.

        after : bool, optional
            If set to `True`, insert *after* the specified index or keyword,
            rather than before it.  Defaults to `False`.
        '''
    def remove(self, keyword, ignore_missing: bool = False, remove_all: bool = False) -> None:
        """
        Removes the first instance of the given keyword from the header similar
        to `list.remove` if the Header object is treated as a list of keywords.

        Parameters
        ----------
        keyword : str
            The keyword of which to remove the first instance in the header.

        ignore_missing : bool, optional
            When True, ignores missing keywords.  Otherwise, if the keyword
            is not present in the header a KeyError is raised.

        remove_all : bool, optional
            When True, all instances of keyword will be removed.
            Otherwise only the first instance of the given keyword is removed.

        """
    def rename_keyword(self, oldkeyword, newkeyword, force: bool = False) -> None:
        """
        Rename a card's keyword in the header.

        Parameters
        ----------
        oldkeyword : str or int
            Old keyword or card index

        newkeyword : str
            New keyword

        force : bool, optional
            When `True`, if the new keyword already exists in the header, force
            the creation of a duplicate keyword. Otherwise a
            `ValueError` is raised.
        """
    def add_history(self, value, before: Incomplete | None = None, after: Incomplete | None = None) -> None:
        """
        Add a ``HISTORY`` card.

        Parameters
        ----------
        value : str
            History text to be added.

        before : str or int, optional
            Same as in `Header.update`

        after : str or int, optional
            Same as in `Header.update`
        """
    def add_comment(self, value, before: Incomplete | None = None, after: Incomplete | None = None) -> None:
        """
        Add a ``COMMENT`` card.

        Parameters
        ----------
        value : str
            Text to be added.

        before : str or int, optional
            Same as in `Header.update`

        after : str or int, optional
            Same as in `Header.update`
        """
    def add_blank(self, value: str = '', before: Incomplete | None = None, after: Incomplete | None = None) -> None:
        """
        Add a blank card.

        Parameters
        ----------
        value : str, optional
            Text to be added.

        before : str or int, optional
            Same as in `Header.update`

        after : str or int, optional
            Same as in `Header.update`
        """
    def strip(self) -> None:
        """
        Strip cards specific to a certain kind of header.

        Strip cards like ``SIMPLE``, ``BITPIX``, etc. so the rest of
        the header can be used to reconstruct another kind of header.
        """
    @property
    def data_size(self):
        """
        Return the size (in bytes) of the data portion following the `Header`.
        """
    @property
    def data_size_padded(self):
        """
        Return the size (in bytes) of the data portion following the `Header`
        including padding.
        """
    def _update(self, card) -> None:
        """
        The real update code.  If keyword already exists, its value and/or
        comment will be updated.  Otherwise a new card will be appended.

        This will not create a duplicate keyword except in the case of
        commentary cards.  The only other way to force creation of a duplicate
        is to use the insert(), append(), or extend() methods.
        """
    def _cardindex(self, key):
        """Returns an index into the ._cards list given a valid lookup key."""
    def _keyword_from_index(self, idx):
        """
        Given an integer index, return the (keyword, repeat) tuple that index
        refers to.  For most keywords the repeat will always be zero, but it
        may be greater than zero for keywords that are duplicated (especially
        commentary keywords).

        In a sense this is the inverse of self.index, except that it also
        supports duplicates.
        """
    def _relativeinsert(self, card, before: Incomplete | None = None, after: Incomplete | None = None, replace: bool = False):
        """
        Inserts a new card before or after an existing card; used to
        implement support for the legacy before/after keyword arguments to
        Header.update().

        If replace=True, move an existing card with the same keyword.
        """
    def _updateindices(self, idx, increment: bool = True) -> None:
        """
        For all cards with index above idx, increment or decrement its index
        value in the keyword_indices dict.
        """
    def _countblanks(self):
        """Returns the number of blank cards at the end of the Header."""
    def _useblanks(self, count) -> None: ...
    def _haswildcard(self, keyword):
        """Return `True` if the input keyword contains a wildcard pattern."""
    def _wildcardmatch(self, pattern):
        """
        Returns a list of indices of the cards matching the given wildcard
        pattern.

         * '*' matches 0 or more characters
         * '?' matches a single character
         * '...' matches 0 or more of any non-whitespace character
        """
    def _set_slice(self, key, value, target):
        """
        Used to implement Header.__setitem__ and CardAccessor.__setitem__.
        """
    def _splitcommentary(self, keyword, value):
        """
        Given a commentary keyword and value, returns a list of the one or more
        cards needed to represent the full value.  This is primarily used to
        create the multiple commentary cards needed to represent a long value
        that won't fit into a single commentary card.
        """
    def _add_commentary(self, key, value, before: Incomplete | None = None, after: Incomplete | None = None) -> None:
        """
        Add a commentary card.

        If ``before`` and ``after`` are `None`, add to the last occurrence
        of cards of the same name (except blank card).  If there is no
        card (or blank card), append at the end.
        """

class _DelayedHeader:
    """
    Descriptor used to create the Header object from the header string that
    was stored in HDU._header_str when parsing the file.
    """
    def __get__(self, obj, owner: Incomplete | None = None): ...
    def __set__(self, obj, val) -> None: ...
    def __delete__(self, obj) -> None: ...

class _BasicHeaderCards:
    """
    This class allows to access cards with the _BasicHeader.cards attribute.

    This is needed because during the HDU class detection, some HDUs uses
    the .cards interface.  Cards cannot be modified here as the _BasicHeader
    object will be deleted once the HDU object is created.

    """
    header: Incomplete
    def __init__(self, header) -> None: ...
    def __getitem__(self, key): ...

class _BasicHeader(collections.abc.Mapping):
    """This class provides a fast header parsing, without all the additional
    features of the Header class. Here only standard keywords are parsed, no
    support for CONTINUE, HIERARCH, COMMENT, HISTORY, or rvkc.

    The raw card images are stored and parsed only if needed. The idea is that
    to create the HDU objects, only a small subset of standard cards is needed.
    Once a card is parsed, which is deferred to the Card class, the Card object
    is kept in a cache. This is useful because a small subset of cards is used
    a lot in the HDU creation process (NAXIS, XTENSION, ...).

    """
    _raw_cards: Incomplete
    _keys: Incomplete
    _cards: Incomplete
    cards: Incomplete
    _modified: bool
    def __init__(self, cards) -> None: ...
    def __getitem__(self, key): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def index(self, keyword): ...
    @property
    def data_size(self):
        """
        Return the size (in bytes) of the data portion following the `Header`.
        """
    @property
    def data_size_padded(self):
        """
        Return the size (in bytes) of the data portion following the `Header`
        including padding.
        """
    @classmethod
    def fromfile(cls, fileobj):
        """The main method to parse a FITS header from a file. The parsing is
        done with the parse_header function implemented in Cython.
        """

class _CardAccessor:
    """
    This is a generic class for wrapping a Header in such a way that you can
    use the header's slice/filtering capabilities to return a subset of cards
    and do something with them.

    This is sort of the opposite notion of the old CardList class--whereas
    Header used to use CardList to get lists of cards, this uses Header to get
    lists of cards.
    """
    _header: Incomplete
    def __init__(self, header) -> None: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __getitem__(self, item): ...
    def _setslice(self, item, value):
        """
        Helper for implementing __setitem__ on _CardAccessor subclasses; slices
        should always be handled in this same way.
        """

class _HeaderComments(_CardAccessor):
    """
    A class used internally by the Header class for the Header.comments
    attribute access.

    This object can be used to display all the keyword comments in the Header,
    or look up the comments on specific keywords.  It allows all the same forms
    of keyword lookup as the Header class itself, but returns comments instead
    of values.
    """
    def __iter__(self): ...
    def __repr__(self) -> str:
        """Returns a simple list of all keywords and their comments."""
    def __getitem__(self, item):
        """
        Slices and filter strings return a new _HeaderComments containing the
        returned cards.  Otherwise the comment of a single card is returned.
        """
    def __setitem__(self, item, comment) -> None:
        """
        Set/update the comment on specified card or cards.

        Slice/filter updates work similarly to how Header.__setitem__ works.
        """

class _HeaderCommentaryCards(_CardAccessor):
    """
    This is used to return a list-like sequence over all the values in the
    header for a given commentary keyword, such as HISTORY.
    """
    _keyword: Incomplete
    _count: Incomplete
    _indices: Incomplete
    def __init__(self, header, keyword: str = '') -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __repr__(self) -> str: ...
    def __getitem__(self, idx): ...
    def __setitem__(self, item, value) -> None:
        """
        Set the value of a specified commentary card or cards.

        Slice/filter updates work similarly to how Header.__setitem__ works.
        """

def _block_size(sep):
    """
    Determine the size of a FITS header block if a non-blank separator is used
    between cards.
    """
def _pad_length(stringlen):
    """Bytes needed to pad the input stringlen to the next FITS block."""
def _check_padding(header_str, block_size, is_eof, check_block_size: bool = True) -> None: ...
def _hdr_data_size(header):
    """Calculate the data size (in bytes) following the given `Header`."""
