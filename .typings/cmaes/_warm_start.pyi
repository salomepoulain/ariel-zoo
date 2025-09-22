import numpy as np

def get_warm_start_mgd(source_solutions: list[tuple[np.ndarray, float]], gamma: float = 0.1, alpha: float = 0.1) -> tuple[np.ndarray, float, np.ndarray]:
    """Estimates a promising distribution of the source task, then
    returns a multivariate gaussian distribution (the mean vector
    and the covariance matrix) used for initialization of the CMA-ES.

    Args:
        source_solutions:
            List of solutions (parameter, value) on a source task.

        gamma:
            top-(gamma x 100)% solutions are selected from a set of solutions
            on a source task. (default: 0.1).

        alpha:
            prior parameter for the initial covariance matrix (default: 0.1).

    Returns:
        The tuple of mean vector, sigma, and covariance matrix.
    """
