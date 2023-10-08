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


def generate_sample_neuron(time, mu, sigma, trials):
    trials = [gaussian_pdf(time, mu, sigma) for _ in range(trials)]
    return trials


def gaussian_pdf(x, mu, sigma):
    z = sigma * np.sqrt(2*np.pi)
    z = 1/z
    numerator = (x - mu)**2
    denominator = sigma**2
    denominator *= -2
    return z * np.exp(numerator / denominator)


if __name__ == "__main__":

    sample = np.arange(-5, 5, 0.1)
    m = 0
    s = np.sqrt(0.2)
    y = gaussian_pdf(sample, m, s)
    fig, ax = plt.subplots(1, 1)
    ax.plot(sample, y)
