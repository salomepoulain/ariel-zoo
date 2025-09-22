from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['Link', 'Info', 'Values', 'Field', 'Param', 'CooSys', 'TimeSys', 'FieldRef', 'ParamRef', 'Group', 'TableElement', 'Resource', 'VOTableFile', 'Element', 'MivotBlock']

class _IDProperty:
    @property
    def ID(self):
        """
        The XML ID_ of the element.  May be `None` or a string
        conforming to XML ID_ syntax.
        """
    _ID: Incomplete
    @ID.setter
    def ID(self, ID) -> None: ...
    @ID.deleter
    def ID(self) -> None: ...

class _NameProperty:
    @property
    def name(self):
        """An optional name for the element."""
    _name: Incomplete
    @name.setter
    def name(self, name) -> None: ...
    @name.deleter
    def name(self) -> None: ...

class _XtypeProperty:
    @property
    def xtype(self):
        """Extended data type information."""
    _xtype: Incomplete
    @xtype.setter
    def xtype(self, xtype) -> None: ...
    @xtype.deleter
    def xtype(self) -> None: ...

class _UtypeProperty:
    _utype_in_v1_2: bool
    @property
    def utype(self):
        """The usage-specific or `unique type`_ of the element."""
    _utype: Incomplete
    @utype.setter
    def utype(self, utype) -> None: ...
    @utype.deleter
    def utype(self) -> None: ...

class _UcdProperty:
    _ucd_in_v1_2: bool
    @property
    def ucd(self):
        """The `unified content descriptor`_ for the element."""
    _ucd: Incomplete
    @ucd.setter
    def ucd(self, ucd) -> None: ...
    @ucd.deleter
    def ucd(self) -> None: ...

class _DescriptionProperty:
    @property
    def description(self):
        """
        An optional string describing the element.  Corresponds to the
        DESCRIPTION_ element.
        """
    _description: Incomplete
    @description.setter
    def description(self, description) -> None: ...
    @description.deleter
    def description(self) -> None: ...

class Element:
    """
    A base class for all classes that represent XML elements in the
    VOTABLE file.
    """
    _element_name: str
    _attr_list: Incomplete
    def _add_unknown_tag(self, iterator, tag, data, config, pos) -> None: ...
    def _ignore_add(self, iterator, tag, data, config, pos) -> None: ...
    def _add_definitions(self, iterator, tag, data, config, pos) -> None: ...
    def parse(self, iterator, config) -> None:
        """
        For internal use. Parse the XML content of the children of the
        element.

        Parameters
        ----------
        iterator : xml iterable
            An iterator over XML elements as returned by
            `~astropy.utils.xml.iterparser.get_xml_iterator`.

        config : dict
            The configuration dictionary that affects how certain
            elements are read.

        Returns
        -------
        self : `~astropy.io.votable.tree.Element`
            Returns self as a convenience.
        """
    def to_xml(self, w, **kwargs) -> None:
        """
        For internal use. Output the element to XML.

        Parameters
        ----------
        w : astropy.utils.xml.writer.XMLWriter object
            An XML writer to write to.
        **kwargs : dict
            Any configuration parameters to control the output.
        """

class SimpleElement(Element):
    """
    A base class for simple elements, such as FIELD, PARAM and INFO
    that don't require any special parsing or outputting machinery.
    """
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def parse(self, iterator, config): ...
    def to_xml(self, w, **kwargs) -> None: ...

class SimpleElementWithContent(SimpleElement):
    """
    A base class for simple elements, such as FIELD, PARAM and INFO
    that don't require any special parsing or outputting machinery.
    """
    _content: Incomplete
    def __init__(self) -> None: ...
    def parse(self, iterator, config): ...
    def to_xml(self, w, **kwargs) -> None: ...
    @property
    def content(self):
        """The content of the element."""
    @content.setter
    def content(self, content) -> None: ...
    @content.deleter
    def content(self) -> None: ...

class Link(SimpleElement, _IDProperty):
    """
    LINK_ elements: used to reference external documents and servers through a URI.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _attr_list: Incomplete
    _element_name: str
    _config: Incomplete
    _pos: Incomplete
    ID: Incomplete
    title: Incomplete
    value: Incomplete
    action: Incomplete
    def __init__(self, ID: Incomplete | None = None, title: Incomplete | None = None, value: Incomplete | None = None, href: Incomplete | None = None, action: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **kwargs) -> None: ...
    @property
    def content_role(self):
        """Defines the MIME role of the referenced object.

        Must be one of:

          None, 'query', 'hints', 'doc', 'location' or 'type'
        """
    _content_role: Incomplete
    @content_role.setter
    def content_role(self, content_role) -> None: ...
    @content_role.deleter
    def content_role(self) -> None: ...
    @property
    def content_type(self):
        """Defines the MIME content type of the referenced object."""
    _content_type: Incomplete
    @content_type.setter
    def content_type(self, content_type) -> None: ...
    @content_type.deleter
    def content_type(self) -> None: ...
    @property
    def href(self):
        """
        A URI to an arbitrary protocol.  The vo package only supports
        http and anonymous ftp.
        """
    _href: Incomplete
    @href.setter
    def href(self, href) -> None: ...
    @href.deleter
    def href(self) -> None: ...
    def to_table_column(self, column) -> None: ...
    @classmethod
    def from_table_column(cls, d): ...

class Info(SimpleElementWithContent, _IDProperty, _XtypeProperty, _UtypeProperty):
    """
    INFO_ elements: arbitrary key-value pairs for extensions to the standard.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _element_name: str
    _attr_list_11: Incomplete
    _attr_list_12: Incomplete
    _utype_in_v1_2: bool
    _config: Incomplete
    _pos: Incomplete
    ID: Incomplete
    xtype: Incomplete
    ucd: Incomplete
    utype: Incomplete
    _attr_list: Incomplete
    def __init__(self, ID: Incomplete | None = None, name: Incomplete | None = None, value: Incomplete | None = None, id: Incomplete | None = None, xtype: Incomplete | None = None, ref: Incomplete | None = None, unit: Incomplete | None = None, ucd: Incomplete | None = None, utype: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None: ...
    @property
    def name(self):
        """[*required*] The key of the key-value pair."""
    _name: Incomplete
    @name.setter
    def name(self, name) -> None: ...
    @property
    def value(self):
        """
        [*required*] The value of the key-value pair.  (Always stored
        as a string or unicode string).
        """
    _value: Incomplete
    @value.setter
    def value(self, value) -> None: ...
    @property
    def content(self):
        """The content inside the INFO element."""
    _content: Incomplete
    @content.setter
    def content(self, content) -> None: ...
    @content.deleter
    def content(self) -> None: ...
    @property
    def ref(self):
        """
        Refer to another INFO_ element by ID_, defined previously in
        the document.
        """
    _ref: Incomplete
    @ref.setter
    def ref(self, ref) -> None: ...
    @ref.deleter
    def ref(self) -> None: ...
    @property
    def unit(self):
        """A string specifying the units_ for the INFO_."""
    _unit: Incomplete
    @unit.setter
    def unit(self, unit) -> None: ...
    @unit.deleter
    def unit(self) -> None: ...
    def to_xml(self, w, **kwargs) -> None: ...

class Values(Element, _IDProperty):
    """
    VALUES_ element: used within FIELD_ and PARAM_ elements to define the domain of values.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _config: Incomplete
    _pos: Incomplete
    _votable: Incomplete
    _field: Incomplete
    ID: Incomplete
    _ref: Incomplete
    _options: Incomplete
    def __init__(self, votable, field, ID: Incomplete | None = None, null: Incomplete | None = None, ref: Incomplete | None = None, type: str = 'legal', id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extras) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def null(self):
        """
        For integral datatypes, *null* is used to define the value
        used for missing values.
        """
    _null: Incomplete
    @null.setter
    def null(self, null) -> None: ...
    @null.deleter
    def null(self) -> None: ...
    @property
    def type(self):
        """Defines the applicability of the domain defined by this VALUES_ element [*required*].

        Must be one of the following strings:

          - 'legal': The domain of this column applies in general to
            this datatype. (default)

          - 'actual': The domain of this column applies only to the
            data enclosed in the parent table.
        """
    _type: Incomplete
    @type.setter
    def type(self, type) -> None: ...
    @property
    def ref(self):
        """
        Refer to another VALUES_ element by ID_, defined previously in
        the document, for MIN/MAX/OPTION information.
        """
    @ref.setter
    def ref(self, ref) -> None: ...
    @ref.deleter
    def ref(self) -> None: ...
    @property
    def min(self):
        """
        The minimum value of the domain.  See :attr:`min_inclusive`.
        """
    _min: Incomplete
    @min.setter
    def min(self, min) -> None: ...
    @min.deleter
    def min(self) -> None: ...
    @property
    def min_inclusive(self):
        """When `True`, the domain includes the minimum value."""
    _min_inclusive: bool
    @min_inclusive.setter
    def min_inclusive(self, inclusive) -> None: ...
    @min_inclusive.deleter
    def min_inclusive(self) -> None: ...
    @property
    def max(self):
        """
        The maximum value of the domain.  See :attr:`max_inclusive`.
        """
    _max: Incomplete
    @max.setter
    def max(self, max) -> None: ...
    @max.deleter
    def max(self) -> None: ...
    @property
    def max_inclusive(self):
        """When `True`, the domain includes the maximum value."""
    _max_inclusive: bool
    @max_inclusive.setter
    def max_inclusive(self, inclusive) -> None: ...
    @max_inclusive.deleter
    def max_inclusive(self) -> None: ...
    @property
    def options(self):
        """
        A list of string key-value tuples defining other OPTION
        elements for the domain.  All options are ignored -- they are
        stored for round-tripping purposes only.
        """
    def parse(self, iterator, config): ...
    def _parse_minmax(self, val): ...
    def is_defaults(self):
        """
        Are the settings on this ``VALUE`` element all the same as the
        XML defaults?.
        """
    def to_xml(self, w, **kwargs): ...
    def to_table_column(self, column) -> None: ...
    def from_table_column(self, column) -> None: ...

class Field(SimpleElement, _IDProperty, _NameProperty, _XtypeProperty, _UtypeProperty, _UcdProperty):
    """
    FIELD_ element: describes the datatype of a particular column of data.

    The keyword arguments correspond to setting members of the same
    name, documented below.

    If *ID* is provided, it is used for the column name in the
    resulting recarray of the table.  If no *ID* is provided, *name*
    is used instead.  If neither is provided, an exception will be
    raised.
    """
    _attr_list_11: Incomplete
    _attr_list_12: Incomplete
    _element_name: str
    _config: Incomplete
    _pos: Incomplete
    _attr_list: Incomplete
    description: Incomplete
    _votable: Incomplete
    ID: Incomplete
    name: Incomplete
    ucd: Incomplete
    utype: Incomplete
    _links: Incomplete
    title: Incomplete
    xtype: Incomplete
    def __init__(self, votable, ID: Incomplete | None = None, name: Incomplete | None = None, datatype: Incomplete | None = None, arraysize: Incomplete | None = None, ucd: Incomplete | None = None, unit: Incomplete | None = None, width: Incomplete | None = None, precision: Incomplete | None = None, utype: Incomplete | None = None, ref: Incomplete | None = None, type: Incomplete | None = None, id: Incomplete | None = None, xtype: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None: ...
    @classmethod
    def uniqify_names(cls, fields) -> None:
        """
        Make sure that all names and titles in a list of fields are
        unique, by appending numbers if necessary.
        """
    converter: Incomplete
    def _setup(self, config, pos) -> None: ...
    @property
    def datatype(self):
        """The datatype of the column [*required*].

        Valid values (as defined by the spec) are:

          'boolean', 'bit', 'unsignedByte', 'short', 'int', 'long',
          'char', 'unicodeChar', 'float', 'double', 'floatComplex', or
          'doubleComplex'

        Many VOTABLE files in the wild use 'string' instead of 'char',
        so that is also a valid option, though 'string' will always be
        converted to 'char' when writing the file back out.
        """
    _datatype: Incomplete
    @datatype.setter
    def datatype(self, datatype) -> None: ...
    @property
    def precision(self):
        """
        Along with :attr:`width`, defines the `numerical accuracy`_
        associated with the data.  These values are used to limit the
        precision when writing floating point values back to the XML
        file.  Otherwise, it is purely informational -- the Numpy
        recarray containing the data itself does not use this
        information.
        """
    _precision: Incomplete
    @precision.setter
    def precision(self, precision) -> None: ...
    @precision.deleter
    def precision(self) -> None: ...
    @property
    def width(self):
        """
        Along with :attr:`precision`, defines the `numerical
        accuracy`_ associated with the data.  These values are used to
        limit the precision when writing floating point values back to
        the XML file.  Otherwise, it is purely informational -- the
        Numpy recarray containing the data itself does not use this
        information.
        """
    _width: Incomplete
    @width.setter
    def width(self, width) -> None: ...
    @width.deleter
    def width(self) -> None: ...
    @property
    def ref(self):
        """
        On FIELD_ elements, ref is used only for informational
        purposes, for example to refer to a COOSYS_ or TIMESYS_ element.
        """
    _ref: Incomplete
    @ref.setter
    def ref(self, ref) -> None: ...
    @ref.deleter
    def ref(self) -> None: ...
    @property
    def unit(self):
        """A string specifying the units_ for the FIELD_."""
    _unit: Incomplete
    @unit.setter
    def unit(self, unit) -> None: ...
    @unit.deleter
    def unit(self) -> None: ...
    @property
    def arraysize(self):
        """
        Specifies the size of the multidimensional array if this
        FIELD_ contains more than a single value.

        See `multidimensional arrays`_.
        """
    _arraysize: Incomplete
    @arraysize.setter
    def arraysize(self, arraysize) -> None: ...
    @arraysize.deleter
    def arraysize(self) -> None: ...
    @property
    def type(self):
        """
        The type attribute on FIELD_ elements is reserved for future
        extensions.
        """
    _type: Incomplete
    @type.setter
    def type(self, type) -> None: ...
    @type.deleter
    def type(self) -> None: ...
    @property
    def values(self):
        """
        A :class:`Values` instance (or `None`) defining the domain
        of the column.
        """
    _values: Incomplete
    @values.setter
    def values(self, values) -> None: ...
    @values.deleter
    def values(self) -> None: ...
    @property
    def links(self):
        """
        A list of :class:`Link` instances used to reference more
        details about the meaning of the FIELD_.  This is purely
        informational and is not used by the `astropy.io.votable`
        package.
        """
    def parse(self, iterator, config): ...
    def to_xml(self, w, **kwargs) -> None: ...
    def to_table_column(self, column) -> None:
        """
        Sets the attributes of a given `astropy.table.Column` instance
        to match the information in this `Field`.
        """
    @classmethod
    def from_table_column(cls, votable, column):
        """
        Restores a `Field` instance from a given
        `astropy.table.Column` instance.
        """

class Param(Field):
    """
    PARAM_ element: constant-valued columns in the data.

    :class:`Param` objects are a subclass of :class:`Field`, and have
    all of its methods and members.  Additionally, it defines :attr:`value`.
    """
    _attr_list_11: Incomplete
    _attr_list_12: Incomplete
    _element_name: str
    _value: Incomplete
    def __init__(self, votable, ID: Incomplete | None = None, name: Incomplete | None = None, value: Incomplete | None = None, datatype: Incomplete | None = None, arraysize: Incomplete | None = None, ucd: Incomplete | None = None, unit: Incomplete | None = None, width: Incomplete | None = None, precision: Incomplete | None = None, utype: Incomplete | None = None, type: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None: ...
    @property
    def value(self):
        """
        [*required*] The constant value of the parameter.  Its type is
        determined by the :attr:`~Field.datatype` member.
        """
    @value.setter
    def value(self, value) -> None: ...
    def _setup(self, config, pos) -> None: ...
    def to_xml(self, w, **kwargs) -> None: ...

class CooSys(SimpleElement):
    """
    COOSYS_ element: defines a coordinate system.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _attr_list: Incomplete
    _element_name: str
    _config: Incomplete
    _pos: Incomplete
    refposition: Incomplete
    def __init__(self, ID: Incomplete | None = None, equinox: Incomplete | None = None, epoch: Incomplete | None = None, system: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, refposition: Incomplete | None = None, **extra) -> None: ...
    @property
    def ID(self):
        """
        [*required*] The XML ID of the COOSYS_ element, used for
        cross-referencing.  May be `None` or a string conforming to
        XML ID_ syntax.
        """
    _ID: Incomplete
    @ID.setter
    def ID(self, ID) -> None: ...
    @property
    def system(self):
        """Specifies the type of coordinate system.

        Valid choices are:

          'eq_FK4', 'eq_FK5', 'ICRS', 'ecl_FK4', 'ecl_FK5', 'galactic',
          'supergalactic', 'xy', 'barycentric', or 'geo_app'
        """
    _system: Incomplete
    @system.setter
    def system(self, system) -> None: ...
    @system.deleter
    def system(self) -> None: ...
    @property
    def equinox(self):
        '''
        A parameter required to fix the equatorial or ecliptic systems
        (as e.g. "J2000" as the default "eq_FK5" or "B1950" as the
        default "eq_FK4").
        '''
    _equinox: Incomplete
    @equinox.setter
    def equinox(self, equinox) -> None: ...
    @equinox.deleter
    def equinox(self) -> None: ...
    @property
    def epoch(self):
        """
        Specifies the epoch of the positions.  It must be a string
        specifying an astronomical year.
        """
    _epoch: Incomplete
    @epoch.setter
    def epoch(self, epoch) -> None: ...
    @epoch.deleter
    def epoch(self) -> None: ...

class TimeSys(SimpleElement):
    """
    TIMESYS_ element: defines a time system.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _attr_list: Incomplete
    _element_name: str
    _config: Incomplete
    _pos: Incomplete
    def __init__(self, ID: Incomplete | None = None, timeorigin: Incomplete | None = None, timescale: Incomplete | None = None, refposition: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None: ...
    @property
    def ID(self):
        """
        [*required*] The XML ID of the TIMESYS_ element, used for
        cross-referencing.  Must be a string conforming to
        XML ID_ syntax.
        """
    _ID: Incomplete
    @ID.setter
    def ID(self, ID) -> None: ...
    @property
    def timeorigin(self):
        '''
        Specifies the time origin of the time coordinate,
        given as a Julian Date for the time scale and
        reference point defined. It is usually given as a
        floating point literal; for convenience, the magic
        strings "MJD-origin" (standing for 2400000.5) and
        "JD-origin" (standing for 0) are also allowed.

        The timeorigin attribute MUST be given unless the
        time’s representation contains a year of a calendar
        era, in which case it MUST NOT be present. In VOTables,
        these representations currently are Gregorian calendar
        years with xtype="timestamp", or years in the Julian
        or Besselian calendar when a column has yr, a, or Ba as
        its unit and no time origin is given.
        '''
    _timeorigin: Incomplete
    @timeorigin.setter
    def timeorigin(self, timeorigin) -> None: ...
    @timeorigin.deleter
    def timeorigin(self) -> None: ...
    @property
    def timescale(self):
        """
        [*required*] String specifying the time scale used. Values
        should be taken from the IVOA timescale vocabulary (documented
        at http://www.ivoa.net/rdf/timescale).
        """
    _timescale: Incomplete
    @timescale.setter
    def timescale(self, timescale) -> None: ...
    @timescale.deleter
    def timescale(self) -> None: ...
    @property
    def refposition(self):
        """
        [*required*] String specifying the reference position. Values
        should be taken from the IVOA refposition vocabulary (documented
        at http://www.ivoa.net/rdf/refposition).
        """
    _refposition: Incomplete
    @refposition.setter
    def refposition(self, refposition) -> None: ...
    @refposition.deleter
    def refposition(self) -> None: ...

class FieldRef(SimpleElement, _UtypeProperty, _UcdProperty):
    """
    FIELDref_ element: used inside of GROUP_ elements to refer to remote FIELD_ elements.
    """
    _attr_list_11: Incomplete
    _attr_list_12: Incomplete
    _element_name: str
    _utype_in_v1_2: bool
    _ucd_in_v1_2: bool
    _config: Incomplete
    _pos: Incomplete
    _table: Incomplete
    ucd: Incomplete
    utype: Incomplete
    _attr_list: Incomplete
    def __init__(self, table, ref, ucd: Incomplete | None = None, utype: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None:
        """
        *table* is the :class:`TableElement` object that this :class:`FieldRef`
        is a member of.

        *ref* is the ID to reference a :class:`Field` object defined
        elsewhere.
        """
    @property
    def ref(self):
        """The ID_ of the FIELD_ that this FIELDref_ references."""
    _ref: Incomplete
    @ref.setter
    def ref(self, ref) -> None: ...
    @ref.deleter
    def ref(self) -> None: ...
    def get_ref(self):
        """
        Lookup the :class:`Field` instance that this :class:`FieldRef`
        references.
        """

class ParamRef(SimpleElement, _UtypeProperty, _UcdProperty):
    """
    PARAMref_ element: used inside of GROUP_ elements to refer to remote PARAM_ elements.

    The keyword arguments correspond to setting members of the same
    name, documented below.

    It contains the following publicly-accessible members:

      *ref*: An XML ID referring to a <PARAM> element.
    """
    _attr_list_11: Incomplete
    _attr_list_12: Incomplete
    _element_name: str
    _utype_in_v1_2: bool
    _ucd_in_v1_2: bool
    _config: Incomplete
    _pos: Incomplete
    _table: Incomplete
    ucd: Incomplete
    utype: Incomplete
    _attr_list: Incomplete
    def __init__(self, table, ref, ucd: Incomplete | None = None, utype: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    @property
    def ref(self):
        """The ID_ of the PARAM_ that this PARAMref_ references."""
    _ref: Incomplete
    @ref.setter
    def ref(self, ref) -> None: ...
    @ref.deleter
    def ref(self) -> None: ...
    def get_ref(self):
        """
        Lookup the :class:`Param` instance that this :class:``PARAMref``
        references.
        """

class Group(Element, _IDProperty, _NameProperty, _UtypeProperty, _UcdProperty, _DescriptionProperty):
    """
    GROUP_ element: groups FIELD_ and PARAM_ elements.

    This information is currently ignored by the vo package---that is
    the columns in the recarray are always flat---but the grouping
    information is stored so that it can be written out again to the
    XML file.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _config: Incomplete
    _pos: Incomplete
    _table: Incomplete
    ID: Incomplete
    name: Incomplete
    ucd: Incomplete
    utype: Incomplete
    description: Incomplete
    _entries: Incomplete
    def __init__(self, table, ID: Incomplete | None = None, name: Incomplete | None = None, ref: Incomplete | None = None, ucd: Incomplete | None = None, utype: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def ref(self):
        """
        Currently ignored, as it's not clear from the spec how this is
        meant to work.
        """
    _ref: Incomplete
    @ref.setter
    def ref(self, ref) -> None: ...
    @ref.deleter
    def ref(self) -> None: ...
    @property
    def entries(self):
        """
        [read-only] A list of members of the GROUP_.  This list may
        only contain objects of type :class:`Param`, :class:`Group`,
        :class:`ParamRef` and :class:`FieldRef`.
        """
    def _add_fieldref(self, iterator, tag, data, config, pos) -> None: ...
    def _add_paramref(self, iterator, tag, data, config, pos) -> None: ...
    def _add_param(self, iterator, tag, data, config, pos) -> None: ...
    def _add_group(self, iterator, tag, data, config, pos) -> None: ...
    def parse(self, iterator, config): ...
    def to_xml(self, w, **kwargs) -> None: ...
    def iter_fields_and_params(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all :class:`Param` elements in this
        :class:`Group`.
        """
    def iter_groups(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all sub-:class:`Group` instances in
        this :class:`Group`.
        """

class TableElement(Element, _IDProperty, _NameProperty, _UcdProperty, _DescriptionProperty):
    '''
    TABLE_ element: optionally contains data.

    It contains the following publicly-accessible and mutable
    attribute:

        *array*: A Numpy masked array of the data itself, where each
        row is a row of votable data, and columns are named and typed
        based on the <FIELD> elements of the table.  The mask is
        parallel to the data array, except for variable-length fields.
        For those fields, the numpy array\'s column type is "object"
        (``"O"``), and another masked array is stored there.

    If the TableElement contains no data, (for example, its enclosing
    :class:`Resource` has :attr:`~Resource.type` == \'meta\') *array*
    will have zero-length.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    '''
    _config: Incomplete
    _pos: Incomplete
    _empty: bool
    _votable: Incomplete
    ID: Incomplete
    name: Incomplete
    _ref: Incomplete
    ucd: Incomplete
    utype: Incomplete
    _nrows: Incomplete
    description: Incomplete
    _fields: Incomplete
    _all_fields: Incomplete
    _params: Incomplete
    _groups: Incomplete
    _links: Incomplete
    _infos: Incomplete
    array: Incomplete
    def __init__(self, votable, ID: Incomplete | None = None, name: Incomplete | None = None, ref: Incomplete | None = None, ucd: Incomplete | None = None, utype: Incomplete | None = None, nrows: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **extra) -> None: ...
    def __repr__(self) -> str: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    @property
    def ref(self): ...
    @ref.setter
    def ref(self, ref) -> None:
        """
        Refer to another TABLE, previously defined, by the *ref* ID_
        for all metadata (FIELD_, PARAM_ etc.) information.
        """
    @ref.deleter
    def ref(self) -> None: ...
    @property
    def format(self):
        """The serialization format of the table [*required*].

        Must be one of:

          'tabledata' (TABLEDATA_), 'binary' (BINARY_), 'binary2' (BINARY2_)
          'fits' (FITS_).

        Note that the 'fits' format, since it requires an external
        file, can not be written out.  Any file read in with 'fits'
        format will be read out, by default, in 'tabledata' format.

        See :ref:`astropy:votable-serialization`.
        """
    _format: Incomplete
    @format.setter
    def format(self, format) -> None: ...
    @property
    def nrows(self):
        """
        [*immutable*] The number of rows in the table, as specified in
        the XML file.
        """
    @property
    def fields(self):
        """
        A list of :class:`Field` objects describing the types of each
        of the data columns.
        """
    @property
    def all_fields(self):
        """
        A list of :class:`Field` objects describing the types of each
        of the data columns. Contrary to ``fields``, this property should
        list every field that's available on disk, including deselected columns.
        """
    @property
    def params(self):
        """
        A list of parameters (constant-valued columns) for the
        table.  Must contain only :class:`Param` objects.
        """
    @property
    def groups(self):
        """
        A list of :class:`Group` objects describing how the columns
        and parameters are grouped.  Currently this information is
        only kept around for round-tripping and informational
        purposes.
        """
    @property
    def links(self):
        """
        A list of :class:`Link` objects (pointers to other documents
        or servers through a URI) for the table.
        """
    @property
    def infos(self):
        """
        A list of :class:`Info` objects for the table.  Allows for
        post-operational diagnostics.
        """
    def is_empty(self):
        """
        Returns True if this table doesn't contain any real data
        because it was skipped over by the parser (through use of the
        ``table_number`` kwarg).
        """
    def create_arrays(self, nrows: int = 0, config: Incomplete | None = None, *, colnumbers: Incomplete | None = None) -> None:
        """
        Create a new array to hold the data based on the current set
        of fields, and store them in the *array* and member variable.
        Any data in the existing array will be lost.

        *nrows*, if provided, is the number of rows to allocate.
        *colnumbers*, if provided, is the list of column indices to select.
        By default, all columns are selected.
        """
    def _resize_strategy(self, size):
        """
        Return a new (larger) size based on size, used for
        reallocating an array when it fills up.  This is in its own
        function so the resizing strategy can be easily replaced.
        """
    def add_field(self, field: Field) -> None: ...
    def _register_field(self, iterator, tag, data, config, pos) -> None: ...
    def _add_param(self, iterator, tag, data, config, pos) -> None: ...
    def _add_group(self, iterator, tag, data, config, pos) -> None: ...
    def _add_link(self, iterator, tag, data, config, pos) -> None: ...
    def _add_info(self, iterator, tag, data, config, pos) -> None: ...
    def parse(self, iterator, config): ...
    def _parse_tabledata(self, iterator, colnumbers, config): ...
    def _get_binary_data_stream(self, iterator, config): ...
    def _parse_binary(self, mode, iterator, colnumbers, config, pos): ...
    def _parse_fits(self, iterator, extnum, config): ...
    def _parse_parquet(self, iterator, config):
        """
        Functionality to parse parquet files that are embedded
        in VOTables.
        """
    def to_xml(self, w, **kwargs) -> None: ...
    def _write_tabledata(self, w, **kwargs) -> None: ...
    def _write_binary(self, mode, w, **kwargs) -> None: ...
    def to_table(self, use_names_over_ids: bool = False):
        """
        Convert this VO Table to an `astropy.table.Table` instance.

        Parameters
        ----------
        use_names_over_ids : bool, optional
           When `True` use the ``name`` attributes of columns as the
           names of columns in the `astropy.table.Table` instance.
           Since names are not guaranteed to be unique, this may cause
           some columns to be renamed by appending numbers to the end.
           Otherwise (default), use the ID attributes as the column
           names.

        .. warning::
           Variable-length array fields may not be restored
           identically when round-tripping through the
           `astropy.table.Table` instance.
        """
    @classmethod
    def from_table(cls, votable, table):
        """
        Create a `TableElement` instance from a given `astropy.table.Table`
        instance.
        """
    def iter_fields_and_params(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all FIELD and PARAM elements in the
        TABLE.
        """
    get_field_by_id: Incomplete
    get_field_by_id_or_name: Incomplete
    get_fields_by_utype: Incomplete
    def iter_groups(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all GROUP elements in the TABLE.
        """
    get_group_by_id: Incomplete
    get_groups_by_utype: Incomplete
    def iter_info(self) -> Generator[Incomplete, Incomplete]: ...

class MivotBlock(Element):
    '''
    MIVOT Block holder:
    Processing VO model views on data is out of the scope of Astropy.
    This is why the only VOmodel-related feature implemented here the
    extraction or the writing of a mapping block from/to a VOTable
    There is no syntax validation other than the allowed tag names.
    The mapping block is handled as a correctly indented XML string
    which is meant to be parsed by the calling API (e.g., PyVO).

    The constructor takes "content" as a parameter, it is the string
    serialization of the MIVOT block.
    If it is None, the instance is meant to be set by the Resource parser.
    Orherwise, the parameter value is parsed to make sure it matches
    the MIVOT XML structure.

    '''
    _content: Incomplete
    _indent_level: int
    _on_error: bool
    def __init__(self, content: Incomplete | None = None) -> None: ...
    def __str__(self) -> str: ...
    def _add_statement(self, start, tag, data, config, pos) -> None:
        """
        Convert the tag as a string and append it to the mapping
        block string with the correct indentation level.
        The signature is the same as for all _add_* methods of the parser.
        """
    def _unknown_mapping_tag(self, start, tag, data, config, pos) -> None:
        """
        In case of unexpected tag, the parsing stops and the mapping block
        is set with a REPORT tag telling what went wrong.
        The signature si that same as for all _add_* methods of the parser.
        """
    @property
    def content(self):
        """
        The XML mapping block serialized as string.
        If there is not mapping block, an empty block is returned in order to
        prevent client code to deal with None blocks.
        """
    def parse(self, votable, iterator, config):
        """
        Regular parser similar to others VOTable components.
        """
    def to_xml(self, w) -> None:
        """
        Tells the writer to insert the MIVOT block in its output stream.
        """
    def check_content_format(self):
        """
        Check if the content is on xml format by building a VOTable,
        putting a MIVOT block in the first resource and trying to parse the VOTable.
        """

class Table(TableElement): ...

class Resource(Element, _IDProperty, _NameProperty, _UtypeProperty, _DescriptionProperty):
    """
    RESOURCE_ element: Groups TABLE_ and RESOURCE_ elements.

    The keyword arguments correspond to setting members of the same
    name, documented below.
    """
    _config: Incomplete
    _pos: Incomplete
    name: Incomplete
    ID: Incomplete
    utype: Incomplete
    _extra_attributes: Incomplete
    description: Incomplete
    _coordinate_systems: Incomplete
    _time_systems: Incomplete
    _groups: Incomplete
    _params: Incomplete
    _infos: Incomplete
    _links: Incomplete
    _tables: Incomplete
    _resources: Incomplete
    _mivot_block: Incomplete
    def __init__(self, name: Incomplete | None = None, ID: Incomplete | None = None, utype: Incomplete | None = None, type: str = 'results', id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def type(self):
        """The type of the resource [*required*].

        Must be either:

          - 'results': This resource contains actual result values
            (default)

          - 'meta': This resource contains only datatype descriptions
            (FIELD_ elements), but no actual data.
        """
    _type: Incomplete
    @type.setter
    def type(self, type) -> None: ...
    @property
    def mivot_block(self):
        """
        Returns the MIVOT block instance.
        If the host resource is of type results, it is taken from the first
        child resource with a MIVOT block, if any.
        Otherwise, it is taken from the host resource.
        """
    @mivot_block.setter
    def mivot_block(self, mivot_block) -> None: ...
    @property
    def extra_attributes(self):
        """Dictionary of extra attributes of the RESOURCE_ element.

        This is dictionary of string keys to string values containing any
        extra attributes of the RESOURCE_ element that are not defined
        in the specification. The specification explicitly allows
        for extra attributes here, but nowhere else.
        """
    @property
    def coordinate_systems(self):
        """
        A list of coordinate system definitions (COOSYS_ elements) for
        the RESOURCE_.  Must contain only `CooSys` objects.
        """
    @property
    def time_systems(self):
        """
        A list of time system definitions (TIMESYS_ elements) for
        the RESOURCE_.  Must contain only `TimeSys` objects.
        """
    @property
    def infos(self):
        """
        A list of informational parameters (key-value pairs) for the
        resource.  Must only contain `Info` objects.
        """
    @property
    def groups(self):
        """
        A list of groups.
        """
    @property
    def params(self):
        """
        A list of parameters (constant-valued columns) for the
        resource.  Must contain only `Param` objects.
        """
    @property
    def links(self):
        """
        A list of links (pointers to other documents or servers
        through a URI) for the resource.  Must contain only `Link`
        objects.
        """
    @property
    def tables(self):
        """
        A list of tables in the resource.  Must contain only
        `TableElement` objects.
        """
    @property
    def resources(self):
        """
        A list of nested resources inside this resource.  Must contain
        only `Resource` objects.
        """
    def _add_table(self, iterator, tag, data, config, pos) -> None: ...
    def _add_info(self, iterator, tag, data, config, pos) -> None: ...
    def _add_group(self, iterator, tag, data, config, pos) -> None: ...
    def _add_param(self, iterator, tag, data, config, pos) -> None: ...
    def _add_coosys(self, iterator, tag, data, config, pos) -> None: ...
    def _add_timesys(self, iterator, tag, data, config, pos) -> None: ...
    def _add_resource(self, iterator, tag, data, config, pos) -> None: ...
    def _add_link(self, iterator, tag, data, config, pos) -> None: ...
    _votable: Incomplete
    def parse(self, votable, iterator, config): ...
    def to_xml(self, w, **kwargs) -> None: ...
    def iter_tables(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterates over all tables in the resource and
        nested resources.
        """
    def iter_fields_and_params(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterates over all FIELD_ and PARAM_ elements in
        the resource, its tables and nested resources.
        """
    def iter_coosys(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterates over all the COOSYS_ elements in the
        resource and nested resources.
        """
    def iter_timesys(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterates over all the TIMESYS_ elements in the
        resource and nested resources.
        """
    def iter_info(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterates over all the INFO_ elements in the
        resource and nested resources.
        """

class VOTableFile(Element, _IDProperty, _DescriptionProperty):
    """
    VOTABLE_ element: represents an entire file.

    The keyword arguments correspond to setting members of the same
    name, documented below.

    *version* is settable at construction time only, since conformance
    tests for building the rest of the structure depend on it.
    """
    _config: Incomplete
    _pos: Incomplete
    ID: Incomplete
    description: Incomplete
    _coordinate_systems: Incomplete
    _time_systems: Incomplete
    _params: Incomplete
    _infos: Incomplete
    _resources: Incomplete
    _groups: Incomplete
    _version: Incomplete
    def __init__(self, ID: Incomplete | None = None, id: Incomplete | None = None, config: Incomplete | None = None, pos: Incomplete | None = None, version: str = '1.4') -> None: ...
    def __repr__(self) -> str: ...
    @property
    def version(self):
        """
        The version of the VOTable specification that the file uses.
        """
    @version.setter
    def version(self, version) -> None: ...
    @property
    def coordinate_systems(self):
        """
        A list of coordinate system descriptions for the file.  Must
        contain only `CooSys` objects.
        """
    @property
    def time_systems(self):
        """
        A list of time system descriptions for the file.  Must
        contain only `TimeSys` objects.
        """
    @property
    def params(self):
        """
        A list of parameters (constant-valued columns) that apply to
        the entire file.  Must contain only `Param` objects.
        """
    @property
    def infos(self):
        """
        A list of informational parameters (key-value pairs) for the
        entire file.  Must only contain `Info` objects.
        """
    @property
    def resources(self):
        """
        A list of resources, in the order they appear in the file.
        Must only contain `Resource` objects.
        """
    @property
    def groups(self):
        """
        A list of groups, in the order they appear in the file.  Only
        supported as a child of the VOTABLE element in VOTable 1.2 or
        later.
        """
    def _add_param(self, iterator, tag, data, config, pos) -> None: ...
    def _add_resource(self, iterator, tag, data, config, pos) -> None: ...
    def _add_coosys(self, iterator, tag, data, config, pos) -> None: ...
    def _add_timesys(self, iterator, tag, data, config, pos) -> None: ...
    def _add_info(self, iterator, tag, data, config, pos) -> None: ...
    def _add_group(self, iterator, tag, data, config, pos) -> None: ...
    def _get_version_checks(self): ...
    _version_namespace_map: Incomplete
    def parse(self, iterator, config): ...
    def to_xml(self, fd, compressed: bool = False, tabledata_format: Incomplete | None = None, _debug_python_based_parser: bool = False, _astropy_version: Incomplete | None = None) -> None:
        """
        Write to an XML file.

        Parameters
        ----------
        fd : str or file-like
            Where to write the file. If a file-like object, must be writable.

        compressed : bool, optional
            When `True`, write to a gzip-compressed file.  (Default:
            `False`)

        tabledata_format : str, optional
            Override the format of the table(s) data to write.  Must
            be one of ``tabledata`` (text representation), ``binary`` or
            ``binary2``.  By default, use the format that was specified
            in each `TableElement` object as it was created or read in.  See
            :ref:`astropy:votable-serialization`.
        """
    def iter_tables(self) -> Generator[Incomplete, Incomplete]:
        '''
        Iterates over all tables in the VOTable file in a "flat" way,
        ignoring the nesting of resources etc.
        '''
    def get_first_table(self):
        """
        Often, you know there is only one table in the file, and
        that's all you need.  This method returns that first table.
        """
    get_table_by_id: Incomplete
    get_tables_by_utype: Incomplete
    def get_table_by_index(self, idx):
        """
        Get a table by its ordinal position in the file.
        """
    def iter_fields_and_params(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all FIELD_ and PARAM_ elements in the
        VOTABLE_ file.
        """
    get_field_by_id: Incomplete
    get_fields_by_utype: Incomplete
    get_field_by_id_or_name: Incomplete
    def iter_values(self) -> Generator[Incomplete]:
        """
        Recursively iterate over all VALUES_ elements in the VOTABLE_
        file.
        """
    get_values_by_id: Incomplete
    def iter_groups(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all GROUP_ elements in the VOTABLE_
        file.
        """
    get_group_by_id: Incomplete
    get_groups_by_utype: Incomplete
    def iter_coosys(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all COOSYS_ elements in the VOTABLE_
        file.
        """
    get_coosys_by_id: Incomplete
    def iter_timesys(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all TIMESYS_ elements in the VOTABLE_
        file.
        """
    get_timesys_by_id: Incomplete
    def iter_info(self) -> Generator[Incomplete, Incomplete]:
        """
        Recursively iterate over all INFO_ elements in the VOTABLE_
        file.
        """
    get_info_by_id: Incomplete
    get_infos_by_name: Incomplete
    def set_all_tables_format(self, format) -> None:
        """
        Set the output storage format of all tables in the file.
        """
    @classmethod
    def from_table(cls, table, table_id: Incomplete | None = None):
        """
        Create a `VOTableFile` instance from a given
        `astropy.table.Table` instance.

        Parameters
        ----------
        table_id : str, optional
            Set the given ID attribute on the returned TableElement instance.
        """
