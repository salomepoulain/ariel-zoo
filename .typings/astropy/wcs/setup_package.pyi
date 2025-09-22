from _typeshed import Incomplete

WCSROOT: Incomplete
WCSVERSION: str

def b(s): ...
def string_escape(s): ...
def determine_64_bit_int():
    '''
    The only configuration parameter needed at compile-time is how to
    specify a 64-bit signed integer.  Python\'s ctypes module can get us
    that information.
    If we can\'t be absolutely certain, we default to "long long int",
    which is correct on most platforms (x86, x86_64).  If we find
    platforms where this heuristic doesn\'t work, we may need to
    hardcode for them.
    '''
def write_wcsconfig_h(paths) -> None:
    """
    Writes out the wcsconfig.h header with local configuration.
    """
def generate_c_docstrings() -> None: ...
def get_wcslib_cfg(cfg, wcslib_files, include_paths) -> None: ...
def get_extensions(): ...
