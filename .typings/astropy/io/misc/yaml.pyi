import yaml
from _typeshed import Incomplete

__all__ = ['AstropyLoader', 'AstropyDumper', 'load', 'load_all', 'dump']

class AstropyLoader(yaml.SafeLoader):
    """
    Custom SafeLoader that constructs astropy core objects as well
    as Python tuple and unicode objects.

    This class is not directly instantiated by user code, but instead is
    used to maintain the available constructor functions that are
    called when parsing a YAML stream.  See the `PyYaml documentation
    <https://pyyaml.org/wiki/PyYAMLDocumentation>`_ for details of the
    class signature.
    """
    def _construct_python_tuple(self, node): ...
    def _construct_python_unicode(self, node): ...

class AstropyDumper(yaml.SafeDumper):
    """
    Custom SafeDumper that represents astropy core objects as well
    as Python tuple and unicode objects.

    This class is not directly instantiated by user code, but instead is
    used to maintain the available representer functions that are
    called when generating a YAML stream from an object.  See the
    `PyYaml documentation <https://pyyaml.org/wiki/PyYAMLDocumentation>`_
    for details of the class signature.
    """
    def _represent_tuple(self, data): ...
    def represent_float(self, data): ...

def load(stream):
    """Parse the first YAML document in a stream using the AstropyLoader and
    produce the corresponding Python object.

    Parameters
    ----------
    stream : str or file-like
        YAML input

    Returns
    -------
    obj : object
        Object corresponding to YAML document
    """
def load_all(stream):
    """Parse the all YAML documents in a stream using the AstropyLoader class and
    produce the corresponding Python object.

    Parameters
    ----------
    stream : str or file-like
        YAML input

    Returns
    -------
    obj : object
        Object corresponding to YAML document

    """
def dump(data, stream: Incomplete | None = None, **kwargs):
    """Serialize a Python object into a YAML stream using the AstropyDumper class.
    If stream is None, return the produced string instead.

    Parameters
    ----------
    data : object
        Object to serialize to YAML
    stream : file-like, optional
        YAML output (if not supplied a string is returned)
    **kwargs
        Other keyword arguments that get passed to yaml.dump()

    Returns
    -------
    out : str or None
        If no ``stream`` is supplied then YAML output is returned as str

    """
