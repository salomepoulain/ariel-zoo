from _typeshed import Incomplete

__all__ = ['SAMPHubProxy']

class SAMPHubProxy:
    """
    Proxy class to simplify the client interaction with a SAMP hub (via the
    standard profile).
    """
    proxy: Incomplete
    _connected: bool
    def __init__(self) -> None: ...
    @property
    def is_connected(self):
        """
        Whether the hub proxy is currently connected to a hub.
        """
    lockfile: Incomplete
    def connect(self, hub: Incomplete | None = None, hub_params: Incomplete | None = None, pool_size: int = 20) -> None:
        """
        Connect to the current SAMP Hub.

        Parameters
        ----------
        hub : `~astropy.samp.SAMPHubServer`, optional
            The hub to connect to.

        hub_params : dict, optional
            Optional dictionary containing the lock-file content of the Hub
            with which to connect. This dictionary has the form
            ``{<token-name>: <token-string>, ...}``.

        pool_size : int, optional
            The number of socket connections opened to communicate with the
            Hub.
        """
    def disconnect(self) -> None:
        """
        Disconnect from the current SAMP Hub.
        """
    @property
    def _samp_hub(self):
        """
        Property to abstract away the path to the hub, which allows this class
        to be used for other profiles.
        """
    def ping(self):
        """
        Proxy to ``ping`` SAMP Hub method (Standard Profile only).
        """
    def set_xmlrpc_callback(self, private_key, xmlrpc_addr):
        """
        Proxy to ``setXmlrpcCallback`` SAMP Hub method (Standard Profile only).
        """
    def register(self, secret):
        """
        Proxy to ``register`` SAMP Hub method.
        """
    def unregister(self, private_key):
        """
        Proxy to ``unregister`` SAMP Hub method.
        """
    def declare_metadata(self, private_key, metadata):
        """
        Proxy to ``declareMetadata`` SAMP Hub method.
        """
    def get_metadata(self, private_key, client_id):
        """
        Proxy to ``getMetadata`` SAMP Hub method.
        """
    def declare_subscriptions(self, private_key, subscriptions):
        """
        Proxy to ``declareSubscriptions`` SAMP Hub method.
        """
    def get_subscriptions(self, private_key, client_id):
        """
        Proxy to ``getSubscriptions`` SAMP Hub method.
        """
    def get_registered_clients(self, private_key):
        """
        Proxy to ``getRegisteredClients`` SAMP Hub method.
        """
    def get_subscribed_clients(self, private_key, mtype):
        """
        Proxy to ``getSubscribedClients`` SAMP Hub method.
        """
    def notify(self, private_key, recipient_id, message):
        """
        Proxy to ``notify`` SAMP Hub method.
        """
    def notify_all(self, private_key, message):
        """
        Proxy to ``notifyAll`` SAMP Hub method.
        """
    def call(self, private_key, recipient_id, msg_tag, message):
        """
        Proxy to ``call`` SAMP Hub method.
        """
    def call_all(self, private_key, msg_tag, message):
        """
        Proxy to ``callAll`` SAMP Hub method.
        """
    def call_and_wait(self, private_key, recipient_id, message, timeout):
        """
        Proxy to ``callAndWait`` SAMP Hub method.
        """
    def reply(self, private_key, msg_id, response):
        """
        Proxy to ``reply`` SAMP Hub method.
        """
