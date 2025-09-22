from _typeshed import Incomplete

__all__ = ['BaseTransform', 'CompositeTransform']

class BaseTransform:
    """
    A transformation object.

    This is used to construct transformations such as scaling, stretching, and
    so on.
    """
    def __add__(self, other): ...

class CompositeTransform(BaseTransform):
    """
    A combination of two transforms.

    Parameters
    ----------
    transform_1 : :class:`astropy.visualization.BaseTransform`
        The first transform to apply.
    transform_2 : :class:`astropy.visualization.BaseTransform`
        The second transform to apply.
    """
    transform_1: Incomplete
    transform_2: Incomplete
    def __init__(self, transform_1, transform_2) -> None: ...
    def __call__(self, values, clip: bool = True): ...
    @property
    def inverse(self): ...
