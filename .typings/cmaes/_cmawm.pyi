import numpy as np
from _typeshed import Incomplete
from cmaes import CMA as CMA
from cmaes._cma import _is_valid_bounds as _is_valid_bounds
from cmaes._stats import chi2_ppf as chi2_ppf, norm_cdf as norm_cdf

chi2_ppf: Incomplete
norm_cdf: Incomplete

class CMAwM:
    """CMA-ES with Margin class with ask-and-tell interface.
    The code is adapted from https://github.com/EvoConJP/CMA-ES_with_Margin.

    Example:

        .. code::

            import numpy as np
            from cmaes import CMAwM

            def ellipsoid_onemax(x, n_zdim):
                n = len(x)
                n_rdim = n - n_zdim
                ellipsoid = sum([(1000 ** (i / (n_rdim - 1)) * x[i]) ** 2 for i in range(n_rdim)])
                onemax = n_zdim - (0. < x[(n - n_zdim):]).sum()
                return ellipsoid + 10 * onemax

            binary_dim, continuous_dim = 10, 10
            dim = binary_dim + continuous_dim
            bounds = np.concatenate(
                [
                    np.tile([0, 1], (binary_dim, 1)),
                    np.tile([-np.inf, np.inf], (continuous_dim, 1)),
                ]
            )
            steps = np.concatenate([np.ones(binary_dim), np.zeros(continuous_dim)])
            optimizer = CMAwM(mean=np.zeros(dim), sigma=2.0, bounds=bounds, steps=steps)

            evals = 0
            while True:
                solutions = []
                for _ in range(optimizer.population_size):
                    x_for_eval, x_for_tell = optimizer.ask()
                    value = ellipsoid_onemax(x_for_eval, binary_dim)
                    evals += 1
                    solutions.append((x_for_tell, value))
                optimizer.tell(solutions)

                if optimizer.should_stop():
                    break

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter.

        steps:
            Each value represents a step of discretization for each dimension.
            Zero (or negative value) means a continuous space.

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

        margin:
            A margin parameter (optional).
    """
    _cma: Incomplete
    _n_max_resampling: Incomplete
    _discrete_idx: Incomplete
    _continuous_idx: Incomplete
    _continuous_space: Incomplete
    _n_zdim: Incomplete
    margin: Incomplete
    z_space: Incomplete
    z_lim: Incomplete
    z_lim_low: Incomplete
    z_lim_up: Incomplete
    m_z_lim_low: Incomplete
    m_z_lim_up: Incomplete
    _A: Incomplete
    def __init__(self, mean: np.ndarray, sigma: float, bounds: np.ndarray, steps: np.ndarray, n_max_resampling: int = 100, seed: int | None = None, population_size: int | None = None, cov: np.ndarray | None = None, margin: float | None = None) -> None: ...
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
    @property
    def _rng(self) -> np.random.RandomState: ...
    def reseed_rng(self, seed: int) -> None: ...
    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a parameter and return (i) encoded x and (ii) raw x.
        The encoded x is used for the evaluation.
        The raw x is used for updating the distribution."""
    def _is_continuous_feasible(self, continuous_param: np.ndarray) -> bool: ...
    def _repair_continuous_params(self, continuous_param: np.ndarray) -> np.ndarray: ...
    def _encode_discrete_params(self, discrete_param: np.ndarray) -> np.ndarray:
        """Encode the values into discrete domain."""
    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""
    def should_stop(self) -> bool: ...
