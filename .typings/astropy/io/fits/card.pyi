from .verify import _Verify
from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['Card', 'Undefined']

class Undefined:
    """Undefined value."""
    def __init__(self) -> None: ...

class Card(_Verify):
    length = CARD_LENGTH
    _keywd_FSC_RE: Incomplete
    _keywd_hierarch_RE: Incomplete
    _digits_FSC: str
    _digits_NFSC: str
    _numr_FSC: Incomplete
    _numr_NFSC: Incomplete
    _number_FSC_RE: Incomplete
    _number_NFSC_RE: Incomplete
    _strg: str
    _comm_field: str
    _strg_comment_RE: Incomplete
    _ascii_text_re: Incomplete
    _value_FSC_RE: Incomplete
    _value_NFSC_RE: Incomplete
    _rvkc_identifier: str
    _rvkc_field: Incomplete
    _rvkc_field_specifier_s: Incomplete
    _rvkc_field_specifier_val: Incomplete
    _rvkc_keyword_val: Incomplete
    _rvkc_keyword_val_comm: Incomplete
    _rvkc_field_specifier_val_RE: Incomplete
    _rvkc_keyword_name_RE: Incomplete
    _rvkc_keyword_val_comm_RE: Incomplete
    _commentary_keywords: Incomplete
    _special_keywords: Incomplete
    _value_indicator = VALUE_INDICATOR
    _keyword: Incomplete
    _value: Incomplete
    _comment: Incomplete
    _valuestring: Incomplete
    _image: Incomplete
    _verified: bool
    _hierarch: bool
    _invalid: bool
    _field_specifier: Incomplete
    _rawkeyword: Incomplete
    _rawvalue: Incomplete
    _modified: bool
    _valuemodified: bool
    def __init__(self, keyword: Incomplete | None = None, value: Incomplete | None = None, comment: Incomplete | None = None, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index): ...
    @property
    def keyword(self):
        """Returns the keyword name parsed from the card image."""
    @keyword.setter
    def keyword(self, keyword) -> None:
        """Set the key attribute; once set it cannot be modified."""
    @property
    def value(self):
        """The value associated with the keyword stored in this card."""
    @value.setter
    def value(self, value) -> None: ...
    @value.deleter
    def value(self) -> None: ...
    @property
    def rawkeyword(self):
        """On record-valued keyword cards this is the name of the standard <= 8
        character FITS keyword that this RVKC is stored in.  Otherwise it is
        the card's normal keyword.
        """
    @property
    def rawvalue(self):
        """On record-valued keyword cards this is the raw string value in
        the ``<field-specifier>: <value>`` format stored in the card in order
        to represent a RVKC.  Otherwise it is the card's normal value.
        """
    @property
    def comment(self):
        """Get the comment attribute from the card image if not already set."""
    @comment.setter
    def comment(self, comment) -> None: ...
    @comment.deleter
    def comment(self) -> None: ...
    @property
    def field_specifier(self):
        """
        The field-specifier of record-valued keyword cards; always `None` on
        normal cards.
        """
    @field_specifier.setter
    def field_specifier(self, field_specifier) -> None: ...
    @field_specifier.deleter
    def field_specifier(self) -> None: ...
    @property
    def image(self):
        '''
        The card "image", that is, the 80 byte character string that represents
        this card in an actual FITS header.
        '''
    @property
    def is_blank(self):
        """
        `True` if the card is completely blank--that is, it has no keyword,
        value, or comment.  It appears in the header as 80 spaces.

        Returns `False` otherwise.
        """
    @classmethod
    def fromstring(cls, image):
        """
        Construct a `Card` object from a (raw) string. It will pad the string
        if it is not the length of a card image (80 columns).  If the card
        image is longer than 80 columns, assume it contains ``CONTINUE``
        card(s).
        """
    @classmethod
    def normalize_keyword(cls, keyword):
        """
        `classmethod` to convert a keyword value that may contain a
        field-specifier to uppercase.  The effect is to raise the key to
        uppercase and leave the field specifier in its original case.

        Parameters
        ----------
        keyword : or str
            A keyword value or a ``keyword.field-specifier`` value
        """
    def _check_if_rvkc(self, *args):
        """
        Determine whether or not the card is a record-valued keyword card.

        If one argument is given, that argument is treated as a full card image
        and parsed as such.  If two arguments are given, the first is treated
        as the card keyword (including the field-specifier if the card is
        intended as a RVKC), and the second as the card value OR the first value
        can be the base keyword, and the second value the 'field-specifier:
        value' string.

        If the check passes the ._keyword, ._value, and .field_specifier
        keywords are set.

        Examples
        --------
        ::

            self._check_if_rvkc('DP1', 'AXIS.1: 2')
            self._check_if_rvkc('DP1.AXIS.1', 2)
            self._check_if_rvkc('DP1     = AXIS.1: 2')
        """
    def _check_if_rvkc_image(self, *args):
        """
        Implements `Card._check_if_rvkc` for the case of an unparsed card
        image.  If given one argument this is the full intact image.  If given
        two arguments the card has already been split between keyword and
        value+comment at the standard value indicator '= '.
        """
    def _init_rvkc(self, keyword, field_specifier, field, value) -> None:
        """
        Sort of addendum to Card.__init__ to set the appropriate internal
        attributes if the card was determined to be a RVKC.
        """
    def _parse_keyword(self): ...
    def _parse_value(self):
        """Extract the keyword value from the card image."""
    def _parse_comment(self):
        """Extract the keyword value from the card image."""
    def _split(self):
        """
        Split the card image between the keyword and the rest of the card.
        """
    def _fix_keyword(self) -> None: ...
    def _fix_value(self) -> None:
        """Fix the card image for fixable non-standard compliance."""
    def _format_keyword(self): ...
    def _format_value(self): ...
    def _format_comment(self): ...
    def _format_image(self): ...
    def _format_long_image(self):
        """
        Break up long string value/comment into ``CONTINUE`` cards.
        This is a primitive implementation: it will put the value
        string in one block and the comment string in another.  Also,
        it does not break at the blank space between words.  So it may
        not look pretty.
        """
    def _format_long_commentary_image(self):
        """
        If a commentary card's value is too long to fit on a single card, this
        will render the card as multiple consecutive commentary card of the
        same type.
        """
    def _verify(self, option: str = 'warn'): ...
    def _itersubcards(self) -> Generator[Incomplete]:
        """
        If the card image is greater than 80 characters, it should consist of a
        normal card followed by one or more CONTINUE card.  This method returns
        the subcards that make up this logical card.

        This can also support the case where a HISTORY or COMMENT card has a
        long value that is stored internally as multiple concatenated card
        images.
        """
