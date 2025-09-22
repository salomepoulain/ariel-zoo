from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['MERGE_STRATEGIES', 'MergeStrategy', 'MergePlus', 'MergeNpConcatenate', 'enable_merge_strategies', 'merge']

MERGE_STRATEGIES: Incomplete

class MergeStrategy:
    """
    Base class for defining a strategy for merging metadata from two
    sources, left and right, into a single output.

    The primary functionality for the class is the ``merge(cls, left, right)``
    class method.  This takes ``left`` and ``right`` side arguments and
    returns a single merged output.

    The first class attribute is ``types``.  This is defined as a list of
    (left_types, right_types) tuples that indicate for which input types the
    merge strategy applies.  In determining whether to apply this merge
    strategy to a pair of (left, right) objects, a test is done:
    ``isinstance(left, left_types) and isinstance(right, right_types)``.  For
    example::

      types = [(np.ndarray, np.ndarray),  # Two ndarrays
               (np.ndarray, (list, tuple)),  # ndarray and (list or tuple)
               ((list, tuple), np.ndarray)]  # (list or tuple) and ndarray

    As a convenience, ``types`` can be defined as a single two-tuple instead of
    a list of two-tuples, e.g. ``types = (np.ndarray, np.ndarray)``.

    The other class attribute is ``enabled``, which defaults to ``False`` in
    the base class.  By defining a subclass of ``MergeStrategy`` the new merge
    strategy is automatically registered to be available for use in
    merging. However, by default the new merge strategy is *not enabled*.  This
    prevents inadvertently changing the behavior of unrelated code that is
    performing metadata merge operations.

    In most cases (particularly in library code that others might use) it is
    recommended to leave custom strategies disabled and use the
    `~astropy.utils.metadata.enable_merge_strategies` context manager to locally
    enable the desired strategies.  However, if one is confident that the
    new strategy will not produce unexpected behavior, then one can globally
    enable it by setting the ``enabled`` class attribute to ``True``.

    Examples
    --------
    Here we define a custom merge strategy that takes an int or float on
    the left and right sides and returns a list with the two values.

      >>> from astropy.utils.metadata import MergeStrategy
      >>> class MergeNumbersAsList(MergeStrategy):
      ...     types = ((int, float), (int, float))  # (left_types, right_types)
      ...
      ...     @classmethod
      ...     def merge(cls, left, right):
      ...         return [left, right]

    """
    enabled: bool
    def __init_subclass__(cls): ...

class MergePlus(MergeStrategy):
    """
    Merge ``left`` and ``right`` objects using the plus operator.  This
    merge strategy is globally enabled by default.
    """
    types: Incomplete
    enabled: bool
    @classmethod
    def merge(cls, left, right): ...

class MergeNpConcatenate(MergeStrategy):
    """
    Merge ``left`` and ``right`` objects using np.concatenate.  This
    merge strategy is globally enabled by default.

    This will upcast a list or tuple to np.ndarray and the output is
    always ndarray.
    """
    types: Incomplete
    enabled: bool
    @classmethod
    def merge(cls, left, right): ...

def enable_merge_strategies(*merge_strategies) -> Generator[None]:
    """
    Context manager to temporarily enable one or more custom metadata merge
    strategies.

    Examples
    --------
    Here we define a custom merge strategy that takes an int or float on
    the left and right sides and returns a list with the two values.

      >>> from astropy.utils.metadata import MergeStrategy
      >>> class MergeNumbersAsList(MergeStrategy):
      ...     types = ((int, float),  # left side types
      ...              (int, float))  # right side types
      ...     @classmethod
      ...     def merge(cls, left, right):
      ...         return [left, right]

    By defining this class the merge strategy is automatically registered to be
    available for use in merging. However, by default new merge strategies are
    *not enabled*.  This prevents inadvertently changing the behavior of
    unrelated code that is performing metadata merge operations.

    In order to use the new merge strategy, use this context manager as in the
    following example::

      >>> from astropy.table import Table, vstack
      >>> from astropy.utils.metadata import enable_merge_strategies
      >>> t1 = Table([[1]], names=['a'])
      >>> t2 = Table([[2]], names=['a'])
      >>> t1.meta = {'m': 1}
      >>> t2.meta = {'m': 2}
      >>> with enable_merge_strategies(MergeNumbersAsList):
      ...    t12 = vstack([t1, t2])
      >>> t12.meta['m']
      [1, 2]

    One can supply further merge strategies as additional arguments to the
    context manager.

    As a convenience, the enabling operation is actually done by checking
    whether the registered strategies are subclasses of the context manager
    arguments.  This means one can define a related set of merge strategies and
    then enable them all at once by enabling the base class.  As a trivial
    example, *all* registered merge strategies can be enabled with::

      >>> with enable_merge_strategies(MergeStrategy):
      ...    t12 = vstack([t1, t2])

    Parameters
    ----------
    *merge_strategies : :class:`~astropy.utils.metadata.MergeStrategy` class
        Merge strategies that will be enabled.
    """
def merge(left, right, merge_func: Incomplete | None = None, metadata_conflicts: str = 'warn', warn_str_func=..., error_str_func=...):
    """
    Merge the ``left`` and ``right`` metadata objects.

    This is a simplistic and limited implementation at this point.
    """
