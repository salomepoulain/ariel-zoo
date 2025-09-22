from _typeshed import Incomplete

__all__ = ['design_matrix', 'construct_regularization', 'periodic_fit']

def design_matrix(t, bands, frequency, dy: Incomplete | None = None, nterms_base: int = 1, nterms_band: int = 1): ...
def construct_regularization(bands, nterms_base: int = 1, nterms_band: int = 1, reg_base: Incomplete | None = None, reg_band: float = 1e-06): ...
def periodic_fit(t, y, dy, bands, frequency, t_fit, bands_fit, center_data: bool = True, nterms_base: int = 1, nterms_band: int = 1, reg_base: Incomplete | None = None, reg_band: float = 1e-06, regularize_by_trace: bool = True):
    """Compute the Lomb-Scargle model fit at a given frequency

    Parameters
    ----------
    t, y, dy : float or array-like
        The times, observations, and uncertainties to fit
    bands : str, or array-like
        The bands of each observation
    frequency : float
        The frequency at which to compute the model
    t_fit : float or array-like
        The times at which the fit should be computed
    center_data : bool (default=True)
        If True, center the input data before applying the fit
    nterms : int (default=1)
        The number of Fourier terms to include in the fit

    Returns
    -------
    y_fit : ndarray
        The model fit evaluated at each value of t_fit
    """
