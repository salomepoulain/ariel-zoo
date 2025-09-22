from _typeshed import Incomplete

__all__ = ['DISTRIBUTION_SAFE_FUNCTIONS', 'DISPATCHED_FUNCTIONS', 'UNSUPPORTED_FUNCTIONS', 'broadcast_arrays', 'concatenate']

DISTRIBUTION_SAFE_FUNCTIONS: Incomplete
DISPATCHED_FUNCTIONS: Incomplete
UNSUPPORTED_FUNCTIONS: Incomplete

def broadcast_arrays(*args, subok: bool = False):
    """Broadcast arrays to a common shape.

    Like `numpy.broadcast_arrays`, applied to both distributions and other data.
    Note that ``subok`` is taken to mean whether or not subclasses of
    the distribution are allowed, i.e., for ``subok=False``,
    `~astropy.uncertainty.NdarrayDistribution` instances will be returned.
    """
def concatenate(arrays, axis: int = 0, out: Incomplete | None = None, dtype: Incomplete | None = None, casting: str = 'same_kind'):
    """Concatenate arrays.

    Like `numpy.concatenate`, but any array that is not already a |Distribution|
    is turned into one with identical samples.
    """
