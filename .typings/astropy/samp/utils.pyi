from _typeshed import Incomplete

__all__ = ['SAMPMsgReplierWrapper']

class _ServerProxyPoolMethod:
    __proxies: Incomplete
    __name: Incomplete
    def __init__(self, proxies, name) -> None: ...
    def __getattr__(self, name): ...
    def __call__(self, *args, **kwrds): ...

class ServerProxyPool:
    """
    A thread-safe pool of `xmlrpc.ServerProxy` objects.
    """
    _proxies: Incomplete
    def __init__(self, size, proxy_class, *args, **keywords) -> None: ...
    def __getattr__(self, name): ...
    def shutdown(self) -> None:
        """Shut down the proxy pool by closing all active connections."""

class SAMPMsgReplierWrapper:
    """
    Function decorator that allows to automatically grab errors and returned
    maps (if any) from a function bound to a SAMP call (or notify).

    Parameters
    ----------
    cli : :class:`~astropy.samp.SAMPIntegratedClient` or :class:`~astropy.samp.SAMPClient`
        SAMP client instance. Decorator initialization, accepting the instance
        of the client that receives the call or notification.
    """
    cli: Incomplete
    def __init__(self, cli) -> None: ...
    def __call__(self, f): ...

class _HubAsClient:
    _handler: Incomplete
    def __init__(self, handler) -> None: ...
    def __getattr__(self, name): ...

class _HubAsClientMethod:
    __send: Incomplete
    __name: Incomplete
    def __init__(self, send, name) -> None: ...
    def __getattr__(self, name): ...
    def __call__(self, *args): ...
