import socketserver
from .constants import SAMP_ICON as SAMP_ICON
from .errors import SAMPWarning as SAMPWarning
from _typeshed import Incomplete
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

__all__: Incomplete

class SAMPSimpleXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    """
    XMLRPC handler of Standard Profile requests.
    """
    def do_GET(self) -> None: ...
    def do_POST(self) -> None:
        """
        Handles the HTTP POST request.

        Attempts to interpret all HTTP POST requests as XML-RPC calls,
        which are forwarded to the server's ``_dispatch`` method for
        handling.
        """

class ThreadingXMLRPCServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    """
    Asynchronous multithreaded XMLRPC server.
    """
    log: Incomplete
    def __init__(self, addr, log: Incomplete | None = None, requestHandler=..., logRequests: bool = True, allow_none: bool = True, encoding: Incomplete | None = None) -> None: ...
    def handle_error(self, request, client_address) -> None: ...
