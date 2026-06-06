"""Adstock and saturation transforms for media variables."""

import numpy as np
import pytensor.tensor as pt


def adstock_geometric(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Geometric (infinite) adstock decay.

    alpha=0 means no carry-over; alpha=0.9 means heavy carry-over.
    """
    n = len(x)
    y = np.zeros(n)
    y[0] = x[0]
    for t in range(1, n):
        y[t] = x[t] + alpha * y[t - 1]
    return y


def adstock_geometric_pt(x, alpha):
    """PyTensor version for use inside PyMC models."""
    def step(x_t, y_prev, alpha):
        return x_t + alpha * y_prev

    result, _ = pt.scan(
        fn=step,
        sequences=[x],
        outputs_info=[pt.zeros(())],
        non_sequences=[alpha],
    )
    return result


def saturation_hill(x: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    Hill (S-curve) saturation.

    k  = half-saturation point (spend at which response = 0.5 * max)
    n  = slope / steepness (n>1 → sigmoidal, n=1 → hyperbolic)
    """
    return x**n / (k**n + x**n)


def saturation_hill_pt(x, k, n):
    """PyTensor version for use inside PyMC models."""
    return x**n / (k**n + x**n)


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] for stable MCMC sampling."""
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)
