from _typeshed import Incomplete

__all__ = ['SAMPHubServer', 'WebProfileDialog']

class SAMPHubServer:
    """
    SAMP Hub Server.

    Parameters
    ----------
    secret : str, optional
        The secret code to use for the SAMP lockfile. If none is is specified,
        the :func:`uuid.uuid1` function is used to generate one.

    addr : str, optional
        Listening address (or IP). This defaults to 127.0.0.1 if the internet
        is not reachable, otherwise it defaults to the host name.

    port : int, optional
        Listening XML-RPC server socket port. If left set to 0 (the default),
        the operating system will select a free port.

    lockfile : str, optional
        Custom lockfile name.

    timeout : int, optional
        Hub inactivity timeout. If ``timeout > 0`` then the Hub automatically
        stops after an inactivity period longer than ``timeout`` seconds. By
        default ``timeout`` is set to 0 (Hub never expires).

    client_timeout : int, optional
        Client inactivity timeout. If ``client_timeout > 0`` then the Hub
        automatically unregisters the clients which result inactive for a
        period longer than ``client_timeout`` seconds. By default
        ``client_timeout`` is set to 0 (clients never expire).

    mode : str, optional
        Defines the Hub running mode. If ``mode`` is ``'single'`` then the Hub
        runs using the standard ``.samp`` lock-file, having a single instance
        for user desktop session. Otherwise, if ``mode`` is ``'multiple'``,
        then the Hub runs using a non-standard lock-file, placed in
        ``.samp-1`` directory, of the form ``samp-hub-<UUID>``, where
        ``<UUID>`` is a unique UUID assigned to the hub.

    label : str, optional
        A string used to label the Hub with a human readable name. This string
        is written in the lock-file assigned to the ``hub.label`` token.

    web_profile : bool, optional
        Enables or disables the Web Profile support.

    web_profile_dialog : class, optional
        Allows a class instance to be specified using ``web_profile_dialog``
        to replace the terminal-based message with e.g. a GUI pop-up. Two
        `queue.Queue` instances will be added to the instance as attributes
        ``queue_request`` and ``queue_result``. When a request is received via
        the ``queue_request`` queue, the pop-up should be displayed, and a
        value of `True` or `False` should be added to ``queue_result``
        depending on whether the user accepted or refused the connection.

    web_port : int, optional
        The port to use for web SAMP. This should not be changed except for
        testing purposes, since web SAMP should always use port 21012.

    pool_size : int, optional
        The number of socket connections opened to communicate with the
        clients.
    """
    _id: Incomplete
    _is_running: bool
    _customlockfilename: Incomplete
    _lockfile: Incomplete
    _addr: Incomplete
    _port: Incomplete
    _mode: Incomplete
    _label: Incomplete
    _timeout: Incomplete
    _client_timeout: Incomplete
    _pool_size: Incomplete
    _web_profile: Incomplete
    _web_profile_dialog: Incomplete
    _web_port: Incomplete
    _web_profile_server: Incomplete
    _web_profile_callbacks: Incomplete
    _web_profile_requests_queue: Incomplete
    _web_profile_requests_result: Incomplete
    _web_profile_requests_semaphore: Incomplete
    _host_name: str
    _thread_lock: Incomplete
    _thread_run: Incomplete
    _thread_hub_timeout: Incomplete
    _thread_client_timeout: Incomplete
    _launched_threads: Incomplete
    _last_activity_time: Incomplete
    _client_activity_time: Incomplete
    _hub_msg_id_counter: int
    _hub_secret_code_customized: Incomplete
    _hub_secret: Incomplete
    _hub_public_id: str
    _private_keys: Incomplete
    _metadata: Incomplete
    _mtype2ids: Incomplete
    _id2mtypes: Incomplete
    _xmlrpc_endpoints: Incomplete
    _sync_msg_ids_heap: Incomplete
    _client_id_counter: int
    def __init__(self, secret: Incomplete | None = None, addr: Incomplete | None = None, port: int = 0, lockfile: Incomplete | None = None, timeout: int = 0, client_timeout: int = 0, mode: str = 'single', label: str = '', web_profile: bool = True, web_profile_dialog: Incomplete | None = None, web_port: int = 21012, pool_size: int = 20) -> None: ...
    @property
    def id(self):
        """
        The unique hub ID.
        """
    def _register_standard_api(self, server) -> None: ...
    def _register_web_profile_api(self, server) -> None: ...
    _server: Incomplete
    _url: Incomplete
    def _start_standard_server(self) -> None: ...
    def _start_web_profile_server(self) -> None: ...
    def _launch_thread(self, group: Incomplete | None = None, target: Incomplete | None = None, name: Incomplete | None = None, args: Incomplete | None = None) -> None: ...
    def _join_launched_threads(self, timeout: Incomplete | None = None) -> None: ...
    def _timeout_test_hub(self) -> None: ...
    def _timeout_test_client(self) -> None: ...
    def _hub_as_client_request_handler(self, method, args): ...
    _hub_private_key: Incomplete
    def _setup_hub_as_client(self) -> None: ...
    def start(self, wait: bool = False) -> None:
        """
        Start the current SAMP Hub instance and create the lock file. Hub
        start-up can be blocking or non blocking depending on the ``wait``
        parameter.

        Parameters
        ----------
        wait : bool
            If `True` then the Hub process is joined with the caller, blocking
            the code flow. Usually `True` option is used to run a stand-alone
            Hub in an executable script. If `False` (default), then the Hub
            process runs in a separated thread. `False` is usually used in a
            Python shell.
        """
    @property
    def params(self):
        """
        The hub parameters (which are written to the logfile).
        """
    def _start_threads(self) -> None: ...
    def _create_secret_code(self): ...
    def stop(self) -> None:
        """
        Stop the current SAMP Hub instance and delete the lock file.
        """
    def _join_all_threads(self, timeout: Incomplete | None = None) -> None: ...
    @property
    def is_running(self):
        """Return an information concerning the Hub running status.

        Returns
        -------
        running : bool
            Is the hub running?
        """
    def _serve_forever(self) -> None: ...
    def _notify_shutdown(self) -> None: ...
    def _notify_register(self, private_key) -> None: ...
    def _notify_unregister(self, private_key) -> None: ...
    def _notify_metadata(self, private_key) -> None: ...
    def _notify_subscriptions(self, private_key) -> None: ...
    def _notify_disconnection(self, private_key) -> None: ...
    def _ping(self): ...
    def _query_by_metadata(self, key, value): ...
    def _set_xmlrpc_callback(self, private_key, xmlrpc_addr): ...
    def _perform_standard_register(self): ...
    def _register(self, secret): ...
    def _get_new_ids(self): ...
    def _unregister(self, private_key): ...
    def _declare_metadata(self, private_key, metadata): ...
    def _get_metadata(self, private_key, client_id): ...
    def _declare_subscriptions(self, private_key, mtypes): ...
    def _get_subscriptions(self, private_key, client_id): ...
    def _get_registered_clients(self, private_key): ...
    def _get_subscribed_clients(self, private_key, mtype): ...
    @staticmethod
    def get_mtype_subtypes(mtype):
        '''
        Return a list containing all the possible wildcarded subtypes of MType.

        Parameters
        ----------
        mtype : str
            MType to be parsed.

        Returns
        -------
        types : list
            List of subtypes

        Examples
        --------
        >>> from astropy.samp import SAMPHubServer
        >>> SAMPHubServer.get_mtype_subtypes("samp.app.ping")
        [\'samp.app.ping\', \'samp.app.*\', \'samp.*\', \'*\']
        '''
    def _is_subscribed(self, private_key, mtype): ...
    def _notify(self, private_key, recipient_id, message): ...
    def _notify_(self, sender_private_key, recipient_public_id, message) -> None: ...
    def _notify_all(self, private_key, message): ...
    def _notify_all_(self, sender_private_key, message): ...
    def _call(self, private_key, recipient_id, msg_tag, message): ...
    def _call_(self, sender_private_key, sender_public_id, recipient_public_id, msg_id, message) -> None: ...
    def _call_all(self, private_key, msg_tag, message): ...
    def _call_all_(self, sender_private_key, sender_public_id, msg_tag, message): ...
    def _call_and_wait(self, private_key, recipient_id, message, timeout): ...
    def _reply(self, private_key, msg_id, response):
        """
        The main method that gets called for replying. This starts up an
        asynchronous reply thread and returns.
        """
    def _reply_(self, responder_private_key, msg_id, response) -> None: ...
    def _retry_method(self, recipient_private_key, recipient_public_id, samp_method_name, arg_params) -> None:
        """
        This method is used to retry a SAMP call several times.

        Parameters
        ----------
        recipient_private_key
            The private key of the receiver of the call
        recipient_public_key
            The public key of the receiver of the call
        samp_method_name : str
            The name of the SAMP method to call
        arg_params : tuple
            Any additional arguments to be passed to the SAMP method
        """
    def _public_id_to_private_key(self, public_id): ...
    def _get_new_hub_msg_id(self, sender_public_id, sender_msg_id): ...
    def _update_last_activity_time(self, private_key: Incomplete | None = None) -> None: ...
    def _receive_notification(self, private_key, sender_id, message): ...
    def _receive_call(self, private_key, sender_id, msg_id, message): ...
    def _receive_response(self, private_key, responder_id, msg_tag, response): ...
    def _web_profile_register(self, identity_info, client_address=('unknown', 0), origin: str = 'unknown'): ...
    def _web_profile_allowReverseCallbacks(self, private_key, allow): ...
    def _web_profile_pullCallbacks(self, private_key, timeout_secs): ...

class WebProfileDialog:
    """
    A base class to make writing Web Profile GUI consent dialogs
    easier.

    The concrete class must:

        1) Poll ``handle_queue`` periodically, using the timer services
           of the GUI's event loop.  This function will call
           ``self.show_dialog`` when a request requires authorization.
           ``self.show_dialog`` will be given the arguments:

              - ``samp_name``: The name of the application making the request.

              - ``details``: A dictionary of details about the client
                making the request.

              - ``client``: A hostname, port pair containing the client
                address.

              - ``origin``: A string containing the origin of the
                request.

        2) Call ``consent`` or ``reject`` based on the user's response to
           the dialog.
    """
    def handle_queue(self) -> None: ...
    def consent(self) -> None: ...
    def reject(self) -> None: ...
