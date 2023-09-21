from __future__ import annotations


import numpy as np
from numba import njit
from tqdm import tqdm


def perona_malik_diffusion(traces: np.ndarray,
                           iters: int = 25,
                           kappa: float = 0.15,
                           gamma: float = 0.25,
                           sigma: float = 0,
                           in_place: bool = False) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion. This is a non-linear smoothing technique that avoids the
    temporal distortion introduced onto traces by standard gaussian smoothing.

    The parameter `kappa` controls the level of smoothing ("diffusion") as a function of the derivative of the trace
    (or "gradient" in the case of 2D images where this algorithm is often used). This function is known as the
    diffusion coefficient. When the derivative for some portion of the trace is low, the algorithm will encourage
    smoothing to reduce noise. If the derivative is large like during a burst of activity, the algorithm will discourage
    smoothing to maintain its structure. Here, the argument `kappa` is multiplied by the dynamic range to generate the
    true kappa.

    The diffusion coefficient implemented here is e^(-(derivative/kappa)^2).

    Perona-Malik diffusion is an iterative process. The parameter `gamma` controls the rate of diffusion, while
    parameter `iters` sets the number of iterations to perform.

    This implementation is currently situated to handle 1-D vectors because it gives us some performance benefits.

    :param traces: matrix of M neurons by N samples

    :param iters: number of iterations

    :param kappa: used to calculate the true kappa, where true kappa = kappa * dynamic range. range 0-1

    :param gamma: rate of diffusion for each iter. range 0-1

    :param in_place: whether to calculate in-place

    :returns: smoothed traces
    """

    assert (0 < iters), "iters must be at least 1"
    assert (0 <= kappa <= 1), "kappa must satisfy 0 <= kappa <= 1"
    assert (0 <= gamma <= 1), "gamma must satisfy 0 <= gamma <= 1"

    if in_place:
        smoothed_traces = traces
    else:
        smoothed_traces = traces.copy()

    if traces.ndim == 1:
        return _perona_malik_diffusion(traces, iters, kappa, gamma, sigma, in_place)
    else:
        for neuron in tqdm(range(traces.shape[0]), total=traces.shape[0], desc="Smoothing Traces"):
            smoothed_traces[neuron, :] = _perona_malik_diffusion(traces[neuron, :], iters, kappa, gamma, sigma, in_place)

    return smoothed_traces


def _perona_malik_diffusion(trace: np.ndarray,
                            iters: int = 25,
                            kappa: float = 0.15,
                            gamma: float = 0.25,
                            sigma: float = 0,
                            in_place: bool = False) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion. This is a non-linear extension of gaussian smoothing
    technique that avoids the temporal distortion introduced onto traces by standard gaussian smoothing.

    The parameter `kappa` controls the level of smoothing ("diffusion") as a function of the derivative of the trace
    (or "gradient" in the case of 2D images where this algorithm is often used). This function is known as the
    diffusion coefficient. When the derivative for some portion of the trace is low, the algorithm will encourage
    smoothing to reduce noise. If the derivative is large like during a burst of activity, the algorithm will discourage
    smoothing to maintain its structure. Here, the argument `kappa` is multiplied by the dynamic range to generate the
    true kappa.

    The diffusion coefficient implemented here is e^(-(derivative/kappa)^2). If the diffusion coefficient was a
    scalar instead of a function of the derivative the algorithm would be equivalent to standard gaussian smoothing.

    Perona-Malik diffusion is an iterative process. The parameter `gamma` controls the rate of diffusion, while
    parameter `iters` sets the number of iterations to perform.

    This implementation is currently situated to handle 1-D vectors because it gives us some performance benefits.

    :param trace: trace for a single neuron (i.e., vector)

    :param iters: number of iterations

    :param kappa: used to calculate the true kappa, where true kappa = kappa * dynamic range. range 0-1

    :param gamma: rate of diffusion for each iter. range 0-1

    :param in_place: whether to calculate in-place

    :returns: smoothed traces
    """

    if in_place:
        smoothed_trace = trace
    else:
        smoothed_trace = trace.copy()

    kappa = kappa * (np.max(smoothed_trace) - np.min(smoothed_trace))

    # preallocate
    derivative = np.zeros_like(smoothed_trace)

    for _ in range(iters):
        derivative = np.zeros_like(smoothed_trace)
        derivative[:-1] = np.diff(smoothed_trace)
        if sigma > 0:
            derivative = gaussian_filter1d(derivative, sigma)
        diffusion = derivative * _diffusion_coefficient(derivative, kappa)
        diffusion[1:] -= diffusion[:-1]
        smoothed_trace += (gamma * diffusion)

    return smoothed_trace


@njit
def _diffusion_coefficient(derivative: np.ndarray, kappa: float) -> np.ndarray:
    """
    Diffusion coefficient

    :param derivative: derivative

    :param kappa: kappa

    :returns: value
    """
    return np.exp(-((derivative / kappa)**2.0))
