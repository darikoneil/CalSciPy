from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns


plt.style.use("CalSciPy.main")


def jitterize(original, jitter, trials):
    return np.random.normal(loc=original, scale=jitter, size=trials)


# width_jitter
# amplitude_jitter
# centre_jitter


def generate_sample_neuron(time, a, c, w, aj, cj, wj, trials):
    times = [time for _ in range(trials)]
    mus = jitterize(c, cj, trials)
    sigmas = jitterize(w, wj, 10)
    return [gaussian_pdf(time_, mu_, sigma_) for time_, mu_, sigma_ in zip(times, mus, sigmas)]


def gaussian_pdf(x, mu, sigma):
    z = sigma * np.sqrt(2*np.pi)
    z = 1/z
    numerator = (x - mu)**2
    denominator = sigma**2
    denominator *= -2
    return z * np.exp(numerator / denominator)


if __name__ == "__main__":

    x = np.arange(-15, 40, 0.1)
    ys = generate_sample_neuron(x, 1, 0, np.sqrt(0.2), 0, 2, 0.1, 5)

