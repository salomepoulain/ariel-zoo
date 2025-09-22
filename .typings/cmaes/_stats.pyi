import numpy as np

@np.vectorize
def norm_cdf(x: float, loc: float = 0.0, scale: float = 1.0) -> float: ...
@np.vectorize
def chi2_ppf(q: float) -> float:
    """
    only deal with the special case df=1, loc=0, scale=1
    solve chi2.cdf(x; df=1) = erf(sqrt(x/2)) = q with bisection method
    """
