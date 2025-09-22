import numpy as np
from _typeshed import Incomplete
from typing import Any

_EPS: float
_MEAN_MAX: float
_SIGMA_MAX: float

class CatCMA:
    '''CatCMA stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

            import numpy as np
            from cmaes import CatCMA

            def sphere_com(x, c):
                return sum(x*x) + len(c) - sum(c[:,0])

            optimizer = CatCMA(mean=3 * np.ones(3), sigma=2.0, cat_num=np.array([3, 3, 3]))

            for generation in range(50):
                solutions = []
                for _ in range(optimizer.population_size):
                    # Ask a parameter
                    x, c = optimizer.ask()
                    value = sphere_com(x, c)
                    solutions.append(((x, c), value))
                    print(f"#{generation} {value}")

                # Tell evaluation values.
                optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multivariate gaussian distribution.

        sigma:
            Initial standard deviation of covariance matrix.

        cat_num:
            Numbers of categories.

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

        cat_param:
            A parameter of categorical distribution (optional).

        margin:
            A margin (lower bound) of categorical distribution (optional).

        min_eigenvalue:
            Lower bound of eigenvalue of multivariate Gaussian distribution (optional).
    '''
    _n_co: Incomplete
    _n_ca: Incomplete
    _n: Incomplete
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
    _K: Incomplete
    _Kmax: Incomplete
    _q: Incomplete
    _q_min: Incomplete
    _min_eigenvalue: Incomplete
    _param_sum: Incomplete
    _alpha: float
    _delta_init: float
    _Delta: float
    _Delta_max: Incomplete
    _gamma: float
    _s: Incomplete
    _delta: Incomplete
    _eps: Incomplete
    _bounds: Incomplete
    _n_max_resampling: Incomplete
    _g: int
    _rng: Incomplete
    _tolxup: float
    _tolfun: float
    _tolconditioncov: float
    _funhist_term: Incomplete
    _funhist_values: Incomplete
    def __init__(self, mean: np.ndarray, sigma: float, cat_num: np.ndarray, bounds: np.ndarray | None = None, n_max_resampling: int = 100, seed: int | None = None, population_size: int | None = None, cov: np.ndarray | None = None, cat_param: np.ndarray | None = None, margin: np.ndarray | None = None, min_eigenvalue: float | None = None) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    @property
    def cont_dim(self) -> int:
        """A number of dimensions of continuous variable"""
    @property
    def cat_dim(self) -> int:
        """A number of dimensions of categorical variable"""
    @property
    def dim(self) -> int:
        """A number of dimensions"""
    @property
    def cat_num(self) -> np.ndarray:
        """Numbers of categories"""
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
    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a parameter"""
    def _eigen_decomposition(self) -> tuple[np.ndarray, np.ndarray]: ...
    def _sample_solution(self) -> tuple[np.ndarray, np.ndarray]: ...
    def _is_feasible(self, param: np.ndarray) -> bool: ...
    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray: ...
    def tell(self, solutions: list[tuple[tuple[np.ndarray, np.ndarray], float]]) -> None:
        """Tell evaluation values"""
    def should_stop(self) -> bool: ...

def _is_valid_bounds(bounds: np.ndarray | None, mean: np.ndarray) -> bool: ...
def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray: ...
def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray: ...
