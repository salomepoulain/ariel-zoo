from .core import CompoundModel as CompoundModel
from _typeshed import Incomplete

class Convolution(CompoundModel):
    """
    Wrapper class for a convolution model.

    Parameters
    ----------
    operator: tuple
        The SPECIAL_OPERATORS entry for the convolution being used.
    model : Model
        The model for the convolution.
    kernel: Model
        The kernel model for the convolution.
    bounding_box : tuple
        A bounding box to define the limits of the integration
        approximation for the convolution.
    resolution : float
        The resolution for the approximation of the convolution.
    cache : bool, optional
        Allow convolution computation to be cached for reuse. This is
        enabled by default.

    Notes
    -----
    This is wrapper is necessary to handle the limitations of the
    pseudospectral convolution binary operator implemented in
    astropy.convolution under `~astropy.convolution.convolve_fft`. In this
    `~astropy.convolution.convolve_fft` it is assumed that the inputs ``array``
    and ``kernel`` span a sufficient portion of the support of the functions of
    the convolution. Consequently, the ``Compound`` created by the
    `~astropy.convolution.convolve_models` function makes the assumption that
    one should pass an input array that sufficiently spans this space. This means
    that slightly different input arrays to this model will result in different
    outputs, even on points of intersection between these arrays.

    This issue is solved by requiring a ``bounding_box`` together with a
    resolution so that one can pre-calculate the entire domain and then
    (by default) cache the convolution values. The function then just
    interpolates the results from this cache.
    """
    bounding_box: Incomplete
    _resolution: Incomplete
    _cache_convolution: Incomplete
    _kwargs: Incomplete
    _convolution: Incomplete
    def __init__(self, operator, model, kernel, bounding_box, resolution, cache: bool = True) -> None: ...
    def clear_cache(self) -> None:
        """
        Clears the cached convolution.
        """
    def _get_convolution(self, **kwargs): ...
    @staticmethod
    def _convolution_inputs(*args): ...
    @staticmethod
    def _convolution_outputs(outputs, output_shape): ...
    def __call__(self, *args, **kw): ...
