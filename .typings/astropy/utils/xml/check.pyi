def check_id(ID):
    """
    Returns `True` if *ID* is a valid XML ID.
    """
def fix_id(ID):
    """
    Given an arbitrary string, create one that can be used as an xml
    id.  This is rather simplistic at the moment, since it just
    replaces non-valid characters with underscores.
    """

_token_regex: str

def check_token(token):
    """
    Returns `True` if *token* is a valid XML token, as defined by XML
    Schema Part 2.
    """
def check_mime_content_type(content_type):
    """
    Returns `True` if *content_type* is a valid MIME content type
    (syntactically at least), as defined by RFC 2045.
    """
def check_anyuri(uri):
    """
    Returns `True` if *uri* is a valid URI as defined in RFC 2396.
    """
