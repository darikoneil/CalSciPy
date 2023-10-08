from __future__ import annotations
from typing import Optional, Sequence, NamedTuple
from numbers import Number
import numpy as np


class _ResponseJitter(NamedTuple):
    amplitude: Number = 0
    peak: Number = 0
    width: Number = 0

    def __str__(self):
        return f"{self._asdict()}"

    def __repr__(self):
        return f"Jitter={self._asdict()}"


class SimulatedResponse:
    def __init__(self,
                 peak: Number,
                 width: Number,
                 amplitude: Number = 1.0,
                 reliability: Number = 1.0,
                 jitter: Optional[Number, Sequence[Number, Number, Number]] = None,
                 noise: Optional[Number] = None):

        # preallocate properties
        self._amplitude = None
        self._jitter = None
        self._peak = None
        self._reliability = None
        self._width = None

        # set properties
        self.amplitude = amplitude
        self.jitter = jitter
        self.noise = noise
        self.peak = peak
        self.reliability = reliability
        self.width = width

    @property
    def amplitude(self):
        return jitterize(self._amplitude, self.jitter.amplitude, 1)

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def base_tuning(self):
        return {
            "amplitude": self._amplitude,
            "peak": self._peak,
            "width": self._width
        }

    @property
    def jitter(self):
        return self._jitter

    @jitter.setter
    def jitter(self, value):
        if isinstance(value, _ResponseJitter):
            self._jitter = value
        elif isinstance(value, Number):
            self._jitter = _ResponseJitter(value, value, value)
        elif isinstance(value, Sequence):
            self._jitter = _ResponseJitter(*value)
        elif value is None:
            self._jitter = _ResponseJitter()
        else:
            raise TypeError("Unable to parse jitter argument")

    @property
    def peak(self):
        return jitterize(self._peak, self.jitter.peak, 1)

    @peak.setter
    def peak(self, value):
        self._peak = value

    @property
    def reliability(self):
        return self._reliability

    @reliability.setter
    def reliability(self, value):
        if 0 <= value <= 1:
            self._reliability = value
        else:
            raise AssertionError("Reliability value must be between 0.0 to 1.0")

    @property
    def width(self):
        return jitterize(self._width, self.jitter.width, 1)

    @width.setter
    def width(self, value):

        self._width = value / 4

    def respond(self, stimulus: np.ndarray):

        amplitude = self.amplitude.copy()
        peak = self.peak.copy()
        width = self.width.copy()

        if self.reliability >= np.random.uniform(0.0, 1.0, 1):
            response = amplitude * gaussian_pdf(stimulus, peak, width)
        else:
            response = np.zeros_like(stimulus)

        if self.noise is not None:
            response += (np.random.normal(0, np.std(response), response.shape[0]) * self.noise)

        return response


def gaussian_pdf(x, mu, sigma):
    """


    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    z = sigma * np.sqrt(2*np.pi)
    z = 1/z
    numerator = (x - mu)**2
    denominator = sigma**2
    denominator *= -2
    return z * np.exp(numerator / denominator)


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


if __name__ == "__main__":

    x = np.arange(-15, 40, 0.1)
    ys = generate_sample_neuron(x, 1, 0, np.sqrt(0.2), 0, 2, 0.1, 5)

