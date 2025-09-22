import logging
from . import config as _config
from _typeshed import Incomplete
from collections.abc import Generator
from logging import CRITICAL as CRITICAL, DEBUG as DEBUG, ERROR as ERROR, FATAL as FATAL, INFO as INFO, NOTSET as NOTSET, WARNING as WARNING

__all__ = ['Conf', 'conf', 'log', 'AstropyLogger', 'LoggingError', 'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL']

log: Incomplete

class LoggingError(Exception):
    """
    This exception is for various errors that occur in the astropy logger,
    typically when activating or deactivating logger-related features.
    """
class _AstLogIPYExc(Exception):
    """
    An exception that is used only as a placeholder to indicate to the
    IPython exception-catching mechanism that the astropy
    exception-capturing is activated. It should not actually be used as
    an exception anywhere.
    """

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.logger`.
    """
    log_level: Incomplete
    log_warnings: Incomplete
    log_exceptions: Incomplete
    log_to_file: Incomplete
    log_file_path: Incomplete
    log_file_level: Incomplete
    log_file_format: Incomplete
    log_file_encoding: Incomplete

conf: Incomplete

class AstropyLogger(Logger):
    """
    This class is used to set up the Astropy logging.

    The main functionality added by this class over the built-in
    logging.Logger class is the ability to keep track of the origin of the
    messages, the ability to enable logging of warnings.warn calls and
    exceptions, and the addition of colorized output and context managers to
    easily capture messages to a file or list.
    """
    def makeRecord(self, name, level, pathname, lineno, msg, args, exc_info, func: Incomplete | None = None, extra: Incomplete | None = None, sinfo: Incomplete | None = None): ...
    _showwarning_orig: Incomplete
    def _showwarning(self, *args, **kwargs): ...
    def warnings_logging_enabled(self): ...
    def enable_warnings_logging(self) -> None:
        """
        Enable logging of warnings.warn() calls.

        Once called, any subsequent calls to ``warnings.warn()`` are
        redirected to this logger and emitted with level ``WARN``. Note that
        this replaces the output from ``warnings.warn``.

        This can be disabled with ``disable_warnings_logging``.
        """
    def disable_warnings_logging(self) -> None:
        """
        Disable logging of warnings.warn() calls.

        Once called, any subsequent calls to ``warnings.warn()`` are no longer
        redirected to this logger.

        This can be re-enabled with ``enable_warnings_logging``.
        """
    _excepthook_orig: Incomplete
    def _excepthook(self, etype, value, traceback) -> None: ...
    def exception_logging_enabled(self):
        """
        Determine if the exception-logging mechanism is enabled.

        Returns
        -------
        exclog : bool
            True if exception logging is on, False if not.
        """
    def enable_exception_logging(self) -> None:
        """
        Enable logging of exceptions.

        Once called, any uncaught exceptions will be emitted with level
        ``ERROR`` by this logger, before being raised.

        This can be disabled with ``disable_exception_logging``.
        """
    def disable_exception_logging(self) -> None:
        """
        Disable logging of exceptions.

        Once called, any uncaught exceptions will no longer be emitted by this
        logger.

        This can be re-enabled with ``enable_exception_logging``.
        """
    def enable_color(self) -> None:
        """
        Enable colorized output.
        """
    def disable_color(self) -> None:
        """
        Disable colorized output.
        """
    def log_to_file(self, filename, filter_level: Incomplete | None = None, filter_origin: Incomplete | None = None) -> Generator[None]:
        """
        Context manager to temporarily log messages to a file.

        Parameters
        ----------
        filename : str
            The file to log messages to.
        filter_level : str
            If set, any log messages less important than ``filter_level`` will
            not be output to the file. Note that this is in addition to the
            top-level filtering for the logger, so if the logger has level
            'INFO', then setting ``filter_level`` to ``INFO`` or ``DEBUG``
            will have no effect, since these messages are already filtered
            out.
        filter_origin : str
            If set, only log messages with an origin starting with
            ``filter_origin`` will be output to the file.

        Notes
        -----
        By default, the logger already outputs log messages to a file set in
        the Astropy configuration file. Using this context manager does not
        stop log messages from being output to that file, nor does it stop log
        messages from being printed to standard output.

        Examples
        --------
        The context manager is used as::

            with logger.log_to_file('myfile.log'):
                # your code here
        """
    def log_to_list(self, filter_level: Incomplete | None = None, filter_origin: Incomplete | None = None) -> Generator[Incomplete]:
        """
        Context manager to temporarily log messages to a list.

        Parameters
        ----------
        filename : str
            The file to log messages to.
        filter_level : str
            If set, any log messages less important than ``filter_level`` will
            not be output to the file. Note that this is in addition to the
            top-level filtering for the logger, so if the logger has level
            'INFO', then setting ``filter_level`` to ``INFO`` or ``DEBUG``
            will have no effect, since these messages are already filtered
            out.
        filter_origin : str
            If set, only log messages with an origin starting with
            ``filter_origin`` will be output to the file.

        Notes
        -----
        Using this context manager does not stop log messages from being
        output to standard output.

        Examples
        --------
        The context manager is used as::

            with logger.log_to_list() as log_list:
                # your code here
        """
    def _set_defaults(self) -> None:
        """
        Reset logger to its initial state.
        """

class StreamHandler(logging.StreamHandler):
    """
    A specialized StreamHandler that logs INFO and DEBUG messages to
    stdout, and all other messages to stderr.  Also provides coloring
    of the output, if enabled in the parent logger.
    """
    def emit(self, record) -> None:
        """
        The formatter for stderr.
        """

class FilterOrigin:
    """A filter for the record origin."""
    origin: Incomplete
    def __init__(self, origin) -> None: ...
    def filter(self, record): ...

class ListHandler(logging.Handler):
    """A handler that can be used to capture the records in a list."""
    log_list: Incomplete
    def __init__(self, filter_level: Incomplete | None = None, filter_origin: Incomplete | None = None) -> None: ...
    def emit(self, record) -> None: ...
