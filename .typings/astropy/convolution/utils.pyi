from _typeshed import Incomplete

__all__ = ['discretize_model', 'KernelError', 'KernelSizeError', 'KernelArithmeticError']

class KernelError(Exception):
    """
    Base error class for kernel errors.
    """
class KernelSizeError(KernelError):
    """
    Called when size of kernels is even.
    """
class KernelArithmeticError(KernelError):
    """Called when doing invalid arithmetic with a kernel."""

def discretize_model(model, x_range, y_range: Incomplete | None = None, mode: str = 'center', factor: int = 10):
    """
    Evaluate an analytical model function on a pixel grid.

    Parameters
    ----------
    model : `~astropy.modeling.Model` or callable.
        Analytical model function to be discretized. A callable that is
        not a `~astropy.modeling.Model` instance is converted to a model
        using `~astropy.modeling.custom_model`.
    x_range : 2-tuple
        Lower and upper bounds of x pixel values at which the model is
        evaluated. The upper bound is non-inclusive. A ``x_range`` of
        ``(0, 3)`` means the model will be evaluated at x pixels 0, 1,
        and 2. The difference between the upper and lower bound must be
        a whole number so that the output array size is well defined.
    y_range : 2-tuple or `None`, optional
        Lower and upper bounds of y pixel values at which the model is
        evaluated. The upper bound is non-inclusive. A ``y_range`` of
        ``(0, 3)`` means the model will be evaluated at y pixels of 0,
        1, and 2. The difference between the upper and lower bound must
        be a whole number so that the output array size is well defined.
        ``y_range`` is necessary only for 2D models.
    mode : {'center', 'linear_interp', 'oversample', 'integrate'}, optional
        One of the following modes:
            * ``'center'`` (default)
                Discretize model by taking the value at the center of
                the pixel bins.
            * ``'linear_interp'``
                Discretize model by linearly interpolating between the
                values at the edges (1D) or corners (2D) of the pixel
                bins. For 2D models, the interpolation is bilinear.
            * ``'oversample'``
                Discretize model by taking the average of model values
                in the pixel bins on an oversampled grid. Use the
                ``factor`` keyword to set the integer oversampling
                factor.
            * ``'integrate'``
                Discretize model by integrating the model over the pixel
                bins using `scipy.integrate.quad`. This mode conserves
                the model integral on a subpixel scale, but is very
                slow.
    factor : int, optional
        The integer oversampling factor used when ``mode='oversample'``.
        Ignored otherwise.

    Returns
    -------
    array : `numpy.ndarray`
        The discretized model array.

    Examples
    --------
    In this example, we define a
    `~astropy.modeling.functional_models.Gaussian1D` model that has been
    normalized so that it sums to 1.0. We then discretize this model
    using the ``'center'``, ``'linear_interp'``, and ``'oversample'``
    (with ``factor=10``) modes.

    .. plot::
        :show-source-link:

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.convolution.utils import discretize_model
        from astropy.modeling.models import Gaussian1D

        gauss_1D = Gaussian1D(1 / (0.5 * np.sqrt(2 * np.pi)), 0, 0.5)
        x_range = (-2, 3)
        x = np.arange(*x_range)
        y_center = discretize_model(gauss_1D, x_range, mode='center')
        y_edge = discretize_model(gauss_1D, x_range, mode='linear_interp')
        y_oversample = discretize_model(gauss_1D, x_range, mode='oversample')

        fig, ax = plt.subplots(figsize=(8, 6))
        label = f'center (sum={y_center.sum():.3f})'
        ax.plot(x, y_center, '.-', label=label)
        label = f'linear_interp (sum={y_edge.sum():.3f})'
        ax.plot(x, y_edge, '.-', label=label)
        label = f'oversample (sum={y_oversample.sum():.3f})'
        ax.plot(x, y_oversample, '.-', label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('Value')
        plt.legend()
    """
