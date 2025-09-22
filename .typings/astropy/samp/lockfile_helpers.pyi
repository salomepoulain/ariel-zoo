from .errors import SAMPHubError as SAMPHubError, SAMPWarning as SAMPWarning
from _typeshed import Incomplete
from astropy import log as log
from astropy.utils.data import get_readable_fileobj as get_readable_fileobj

def read_lockfile(lockfilename):
    """
    Read in the lockfile given by ``lockfilename`` into a dictionary.
    """
def write_lockfile(lockfilename, lockfiledict) -> None: ...
def create_lock_file(lockfilename: Incomplete | None = None, mode: Incomplete | None = None, hub_id: Incomplete | None = None, hub_params: Incomplete | None = None): ...
def get_main_running_hub():
    """
    Get either the hub given by the environment variable SAMP_HUB, or the one
    given by the lockfile .samp in the user home directory.
    """
def get_running_hubs():
    """
    Return a dictionary containing the lock-file contents of all the currently
    running hubs (single and/or multiple mode).

    The dictionary format is:

    ``{<lock-file>: {<token-name>: <token-string>, ...}, ...}``

    where ``{<lock-file>}`` is the lock-file name, ``{<token-name>}`` and
    ``{<token-string>}`` are the lock-file tokens (name and content).

    Returns
    -------
    running_hubs : dict
        Lock-file contents of all the currently running hubs.
    """
def check_running_hub(lockfilename):
    """
    Test whether a hub identified by ``lockfilename`` is running or not.

    Parameters
    ----------
    lockfilename : str
        Lock-file name (path + file name) of the Hub to be tested.

    Returns
    -------
    is_running : bool
        Whether the hub is running
    hub_params : dict
        If the hub is running this contains the parameters from the lockfile
    """
def remove_garbage_lock_files() -> None: ...
