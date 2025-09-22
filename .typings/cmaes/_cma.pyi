import numpy as np
from _typeshed import Incomplete
from typing import Any

_EPS: float
_MEAN_MAX: float
_SIGMA_MAX: float

class CMA:
    '''CMA-ES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import CMA

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = CMA(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).

        lr_adapt:
            Flag for learning rate adaptation (optional; default=False)
    '''
    _n_dim: Incomplete
    _popsize: Incomplete
    _mu: Incomplete
    _mu_eff: Incomplete
    _cc: Incomplete
    _c1: Incomplete
    _cmu: Incomplete
    _c_sigma: Incomplete
    _d_sigma: Incomplete
    _cm: Incomplete
    _chi_n: Incomplete
    _weights: Incomplete
    _p_sigma: Incomplete
    _pc: Incomplete
    _mean: Incomplete
    _C: Incomplete
    _sigma: Incomplete
    _D: np.ndarray | None
    _B: np.ndarray | None
    _bounds: Incomplete
    _n_max_resampling: Incomplete
    _g: int
    _rng: Incomplete
    _lr_adapt: Incomplete
    _alpha: float
    _beta_mean: float
    _beta_Sigma: float
    _gamma: float
    _Emean: Incomplete
    _ESigma: Incomplete
    _Vmean: float
    _VSigma: float
    _eta_mean: float
    _eta_Sigma: float
    _tolx: Incomplete
    _tolxup: float
    _tolfun: float
    _tolconditioncov: float
    _funhist_term: Incomplete
    _funhist_values: Incomplete
    def __init__(self, mean: np.ndarray, sigma: float, bounds: np.ndarray | None = None, n_max_resampling: int = 100, seed: int | None = None, population_size: int | None = None, cov: np.ndarray | None = None, lr_adapt: bool = False) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    @property
    def dim(self) -> int:
        """A number of dimensions"""
    @property
    def population_size(self) -> int:
        """A population size"""
    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
    @property
    def mean(self) -> np.ndarray:
        """Mean Vector"""
    def reseed_rng(self, seed: int) -> None: ...
    def set_bounds(self, bounds: np.ndarray | None) -> None:
        """Update boundary constraints"""
    def ask(self) -> np.ndarray:
        """Sample a parameter"""
    def _eigen_decomposition(self) -> tuple[np.ndarray, np.ndarray]: ...
    def _sample_solution(self) -> np.ndarray: ...
    def _is_feasible(self, param: np.ndarray) -> bool: ...
    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray: ...
    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""
    def _lr_adaptation(self, old_mean: np.ndarray, old_sigma: float, old_Sigma: np.ndarray, old_invsqrtC: np.ndarray) -> None: ...
    def should_stop(self) -> bool: ...

def _is_valid_bounds(bounds: np.ndarray | None, mean: np.ndarray) -> bool: ...
def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray: ...
def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray: ...
