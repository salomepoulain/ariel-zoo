from _typeshed import Incomplete

__all__ = ['SAMPIntegratedClient']

class SAMPIntegratedClient:
    """
    A Simple SAMP client.

    This class is meant to simplify the client usage providing a proxy class
    that merges the :class:`~astropy.samp.SAMPClient` and
    :class:`~astropy.samp.SAMPHubProxy` functionalities in a
    simplified API.

    Parameters
    ----------
    name : str, optional
        Client name (corresponding to ``samp.name`` metadata keyword).

    description : str, optional
        Client description (corresponding to ``samp.description.text`` metadata
        keyword).

    metadata : dict, optional
        Client application metadata in the standard SAMP format.

    addr : str, optional
        Listening address (or IP). This defaults to 127.0.0.1 if the internet
        is not reachable, otherwise it defaults to the host name.

    port : int, optional
        Listening XML-RPC server socket port. If left set to 0 (the default),
        the operating system will select a free port.

    callable : bool, optional
        Whether the client can receive calls and notifications. If set to
        `False`, then the client can send notifications and calls, but can not
        receive any.
    """
    hub: Incomplete
    client_arguments: Incomplete
    client: Incomplete
    def __init__(self, name: Incomplete | None = None, description: Incomplete | None = None, metadata: Incomplete | None = None, addr: Incomplete | None = None, port: int = 0, callable: bool = True) -> None: ...
    @property
    def is_connected(self):
        """
        Testing method to verify the client connection with a running Hub.

        Returns
        -------
        is_connected : bool
            True if the client is connected to a Hub, False otherwise.
        """
    def connect(self, hub: Incomplete | None = None, hub_params: Incomplete | None = None, pool_size: int = 20) -> None:
        """
        Connect with the current or specified SAMP Hub, start and register the
        client.

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
        Unregister the client from the current SAMP Hub, stop the client and
        disconnect from the Hub.
        """
    def ping(self):
        """
        Proxy to ``ping`` SAMP Hub method (Standard Profile only).
        """
    def declare_metadata(self, metadata):
        """
        Proxy to ``declareMetadata`` SAMP Hub method.
        """
    def get_metadata(self, client_id):
        """
        Proxy to ``getMetadata`` SAMP Hub method.
        """
    def get_subscriptions(self, client_id):
        """
        Proxy to ``getSubscriptions`` SAMP Hub method.
        """
    def get_registered_clients(self):
        """
        Proxy to ``getRegisteredClients`` SAMP Hub method.

        This returns all the registered clients, excluding the current client.
        """
    def get_subscribed_clients(self, mtype):
        """
        Proxy to ``getSubscribedClients`` SAMP Hub method.
        """
    def _format_easy_msg(self, mtype, params): ...
    def notify(self, recipient_id, message):
        """
        Proxy to ``notify`` SAMP Hub method.
        """
    def enotify(self, recipient_id, mtype, **params):
        '''
        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.notify`.

        This is a proxy to ``notify`` method that allows to send the
        notification message in a simplified way.

        Note that reserved ``extra_kws`` keyword is a dictionary with the
        special meaning of being used to add extra keywords, in addition to
        the standard ``samp.mtype`` and ``samp.params``, to the message sent.

        Parameters
        ----------
        recipient_id : str
            Recipient ID

        mtype : str
            the MType to be notified

        params : dict or set of str
            Variable keyword set which contains the list of parameters for the
            specified MType.

        Examples
        --------
        >>> from astropy.samp import SAMPIntegratedClient
        >>> cli = SAMPIntegratedClient()
        >>> ...
        >>> cli.enotify("samp.msg.progress", msgid = "xyz", txt = "initialization",
        ...             percent = "10", extra_kws = {"my.extra.info": "just an example"})
        '''
    def notify_all(self, message):
        """
        Proxy to ``notifyAll`` SAMP Hub method.
        """
    def enotify_all(self, mtype, **params):
        '''
        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.notify_all`.

        This is a proxy to ``notifyAll`` method that allows to send the
        notification message in a simplified way.

        Note that reserved ``extra_kws`` keyword is a dictionary with the
        special meaning of being used to add extra keywords, in addition to
        the standard ``samp.mtype`` and ``samp.params``, to the message sent.

        Parameters
        ----------
        mtype : str
            MType to be notified.

        params : dict or set of str
            Variable keyword set which contains the list of parameters for
            the specified MType.

        Examples
        --------
        >>> from astropy.samp import SAMPIntegratedClient
        >>> cli = SAMPIntegratedClient()
        >>> ...
        >>> cli.enotify_all("samp.msg.progress", txt = "initialization",
        ...                 percent = "10",
        ...                 extra_kws = {"my.extra.info": "just an example"})
        '''
    def call(self, recipient_id, msg_tag, message):
        """
        Proxy to ``call`` SAMP Hub method.
        """
    def ecall(self, recipient_id, msg_tag, mtype, **params):
        '''
        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.call`.

        This is a proxy to ``call`` method that allows to send a call message
        in a simplified way.

        Note that reserved ``extra_kws`` keyword is a dictionary with the
        special meaning of being used to add extra keywords, in addition to
        the standard ``samp.mtype`` and ``samp.params``, to the message sent.

        Parameters
        ----------
        recipient_id : str
            Recipient ID

        msg_tag : str
            Message tag to use

        mtype : str
            MType to be sent

        params : dict of set of str
            Variable keyword set which contains the list of parameters for
            the specified MType.

        Examples
        --------
        >>> from astropy.samp import SAMPIntegratedClient
        >>> cli = SAMPIntegratedClient()
        >>> ...
        >>> msgid = cli.ecall("abc", "xyz", "samp.msg.progress",
        ...                   txt = "initialization", percent = "10",
        ...                   extra_kws = {"my.extra.info": "just an example"})
        '''
    def call_all(self, msg_tag, message):
        """
        Proxy to ``callAll`` SAMP Hub method.
        """
    def ecall_all(self, msg_tag, mtype, **params) -> None:
        '''
        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.call_all`.

        This is a proxy to ``callAll`` method that allows to send the call
        message in a simplified way.

        Note that reserved ``extra_kws`` keyword is a dictionary with the
        special meaning of being used to add extra keywords, in addition to
        the standard ``samp.mtype`` and ``samp.params``, to the message sent.

        Parameters
        ----------
        msg_tag : str
            Message tag to use

        mtype : str
            MType to be sent

        params : dict of set of str
            Variable keyword set which contains the list of parameters for
            the specified MType.

        Examples
        --------
        >>> from astropy.samp import SAMPIntegratedClient
        >>> cli = SAMPIntegratedClient()
        >>> ...
        >>> msgid = cli.ecall_all("xyz", "samp.msg.progress",
        ...                       txt = "initialization", percent = "10",
        ...                       extra_kws = {"my.extra.info": "just an example"})
        '''
    def call_and_wait(self, recipient_id, message, timeout):
        """
        Proxy to ``callAndWait`` SAMP Hub method.
        """
    def ecall_and_wait(self, recipient_id, mtype, timeout, **params):
        '''
        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.call_and_wait`.

        This is a proxy to ``callAndWait`` method that allows to send the call
        message in a simplified way.

        Note that reserved ``extra_kws`` keyword is a dictionary with the
        special meaning of being used to add extra keywords, in addition to
        the standard ``samp.mtype`` and ``samp.params``, to the message sent.

        Parameters
        ----------
        recipient_id : str
            Recipient ID

        mtype : str
            MType to be sent

        timeout : str
            Call timeout in seconds

        params : dict of set of str
            Variable keyword set which contains the list of parameters for
            the specified MType.

        Examples
        --------
        >>> from astropy.samp import SAMPIntegratedClient
        >>> cli = SAMPIntegratedClient()
        >>> ...
        >>> cli.ecall_and_wait("xyz", "samp.msg.progress", "5",
        ...                    txt = "initialization", percent = "10",
        ...                    extra_kws = {"my.extra.info": "just an example"})
        '''
    def reply(self, msg_id, response):
        """
        Proxy to ``reply`` SAMP Hub method.
        """
    def _format_easy_response(self, status, result, error): ...
    def ereply(self, msg_id, status, result: Incomplete | None = None, error: Incomplete | None = None):
        '''
        Easy to use version of :meth:`~astropy.samp.integrated_client.SAMPIntegratedClient.reply`.

        This is a proxy to ``reply`` method that allows to send a reply
        message in a simplified way.

        Parameters
        ----------
        msg_id : str
            Message ID to which reply.

        status : str
            Content of the ``samp.status`` response keyword.

        result : dict
            Content of the ``samp.result`` response keyword.

        error : dict
            Content of the ``samp.error`` response keyword.

        Examples
        --------
        >>> from astropy.samp import SAMPIntegratedClient, SAMP_STATUS_ERROR
        >>> cli = SAMPIntegratedClient()
        >>> ...
        >>> cli.ereply("abd", SAMP_STATUS_ERROR, result={},
        ...            error={"samp.errortxt": "Test error message"})
        '''
    def receive_notification(self, private_key, sender_id, message): ...
    def receive_call(self, private_key, sender_id, msg_id, message): ...
    def receive_response(self, private_key, responder_id, msg_tag, response): ...
    def bind_receive_message(self, mtype, function, declare: bool = True, metadata: Incomplete | None = None) -> None: ...
    def bind_receive_notification(self, mtype, function, declare: bool = True, metadata: Incomplete | None = None) -> None: ...
    def bind_receive_call(self, mtype, function, declare: bool = True, metadata: Incomplete | None = None) -> None: ...
    def bind_receive_response(self, msg_tag, function) -> None: ...
    def unbind_receive_notification(self, mtype, declare: bool = True) -> None: ...
    def unbind_receive_call(self, mtype, declare: bool = True) -> None: ...
    def unbind_receive_response(self, msg_tag) -> None: ...
    def declare_subscriptions(self, subscriptions: Incomplete | None = None) -> None: ...
    def get_private_key(self): ...
    def get_public_id(self): ...
