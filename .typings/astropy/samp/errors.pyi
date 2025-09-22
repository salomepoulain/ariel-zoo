import xmlrpc.client as xmlrpc
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['SAMPWarning', 'SAMPHubError', 'SAMPClientError', 'SAMPProxyError']

class SAMPWarning(AstropyUserWarning):
    """
    SAMP-specific Astropy warning class.
    """
class SAMPHubError(Exception):
    """
    SAMP Hub exception.
    """
class SAMPClientError(Exception):
    """
    SAMP Client exceptions.
    """
class SAMPProxyError(xmlrpc.Fault):
    """
    SAMP Proxy Hub exception.
    """
