from __future__ import annotations
from typing import Iterable, Callable, Tuple, List
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def calculate_firing_rates(spike_probability_matrix: np.ndarray, frame_rate: float = 30.0, in_place: bool = False) \
        -> np.ndarray:
    """
    Calculate firing rates

    :param spike_probability_matrix: matrix of n neuron x m samples where each element is the probability of a spike
    :type spike_probability_matrix: numpy.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: firing matrix of n neurons x m samples where each element is a binary indicating presence of spike event
    :rtype: numpy.ndarray
    """
    if in_place:
        firing_matrix = spike_probability_matrix
    else:
        firing_matrix = spike_probability_matrix.copy()

    firing_matrix *= frame_rate

    return firing_matrix


def calculate_mean_firing_rates(firing_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate mean firing rate

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
        instantaneous firing rate
    :type firing_matrix: numpy.ndarray
    :return: 1-D vector of mean firing rates
    :rtype: numpy.ndarray
    """
    return np.nanmean(firing_matrix, axis=1)


def collect_waveforms(traces: np.ndarray, event_indices: Iterable[Iterable[int]], pre: int = 150, post: int = 450) \
        -> Tuple[np.ndarray]:
    """
    Collect waveforms for each event

    :param traces: a matrix of M neurons x N samples
    :type traces: numpy.ndarray
    :param event_indices: a list of events
    :type event_indices: Iterable[Iterable[int]]
    :param pre: number of pre-event frames
    :type pre: int
    :param post: number of post-event frames
    :type post: int
    :return: a matrix of M events x N samples
    :rtype: Tuple[numpy.ndarray]
    """
    return tuple([_collect_waveforms(traces[neuron, :], event_indices[neuron], pre, post)
                  for neuron in range(traces.shape[0])])


def convert_tau(tau: float, dt: float) -> float:
    """
    Converts a discrete tau to a continuous tau

    :param tau: decay constant
    :type dt: float
    :param dt: time step (s)
    :type dt: float
    :return: continuous tau (s)
    :rtype: float
    """
    ctau = np.roots(np.hstack([1, -tau]))
    if ctau < 0:
        ctau = 0
    ctau = np.log(ctau) / dt
    return -1 / ctau


def estimate_event_durations(traces: np.ndarray, event_indices: Iterable[Iterable[int]], tau: Iterable[float],
                             share_peaks: bool = False, noise_threshold: Iterable[float] = None) -> Tuple[List[float]]:
    """
    Estimate the duration of an event by calculating the decay from the peak to noise threshold

    using the decay to the noise_threshold

    :param traces: An M neuron by N sample matrix
    :type traces: numpy.ndarray
    :param event_indices: An Iterable of length M neurons containing sequences of frames identified as event onsets
    :type event_indices: Iterable[Iterable[int]]
    :param tau: An Iterable containing the continuous tau value for each neuron
    :type tau: Iterable[float]
    :param share_peaks: whether to constrain peak calculation such that the peak for any two events cannot be shared.
    :type share_peaks: bool = False
    :param noise_threshold: value of traces considered noise
    :type noise_threshold: float = None
    :return: An iterable of length M containing a sequence with a duration for each event
    :rtype: Tuple[List[float]]
    """
    num_neurons, num_frames = traces.shape

    event_durations = []  # this is fine, don't optimize yet

    if not noise_threshold:  # calculate noise if not provided
        noise_threshold = 2 * np.std(traces, axis=-1)

    for neuron in range(num_neurons):
        events = event_indices[neuron]  # get events for this neuron
        durations = []  # this is durations for this single neuron
        for event in events:
            if event >= frames - 2:
                break  # too close to the edge!
            pointer = event + 1
            last_value = traces[neuron, event].copy()
            current_value = traces[neuron, pointer].copy()
            while current_value > last_value:
                if not share_peaks and pointer in events:
                    last_value = current_value.copy()
                    break
                elif pointer == frames - 1:
                    last_value = current_value.copy()
                    pointer += 1
                    break
                else:
                    pointer += 1
                    last_value = current_value.copy()
                    current_value = traces[neuron, :].copy()
            if pointer < frames:
                durations.append(_estimate_event_duration(last_value, tau, noise_threshold[neuron]))
        event_durations.append(durations)
    return tuple(event_durations)


def get_num_events(event_indices: Iterable[Iterable[int]]) -> np.ndarray:
    """
    Determines the number of events for each neuron in the event indices

    :param event_indices: An iterable of length M neurons containing a sequence with a duration for each event
    :type event_indices: Iterable[Iterable[int]]
    :return: A 1-D vector of length M neurons containing the number of events for each neuron
    :rtype: numpy.ndarray
    """
    return np.array([len(event_index) for event_index in event_indices])


def get_inter_event_intervals(event_indices: Iterable[Iterable[int]], frame_rate: float = 30.0) -> Tuple[np.ndarray]:
    """
    Calculate the inter event intervals for each neuron in the event indices

    :param event_indices: An iterable of length M containing a sequence with a duration for each event
    :type event_indices: Iterable[Iterable[int]]
    :param frame_rate: frame_rate for trace matrix
    :type frame_rate: float
    :return: An iterable of length M neurons containing the inter-event intervals for each event in the sequence
    :rtype: Tuple[numpy.ndarray]
    """
    return tuple([np.diff(events) / frame_rate for events in event_indices])


def get_event_onset_intensities(traces: np.ndarray, event_indices: Iterable[Iterable[int]]):
    """
    Retrieve the signal intensity at event onset for each neuron in the event indices

    :param traces: An M neuron by N sample matrix
    :type traces: numpy.ndarray
    :param event_indices: An iterable of length M containing a sequence with a duration for each event
    :type event_indices: Iterable[Iterable[int]]
    :return: An iterable of length M neurons containing the onset intensities for each event in the sequence
    :rtype: Tuple[numpy.ndarray]
    """
    intensities = []
    for neuron in range(traces.shape[0]):
        intensities.append([traces[neuron, event] for event in event_indices[neuron]])
    return tuple(intensities)


def identify_events(traces: np.ndarray, timeout: int = 15, frame_rate: float = 30.0, smooth: bool = True,
                    force_nonneg: bool = True) \
        -> Tuple[List[Int]]:
    """
    Identify event onset for each neuron using the smoothed, non-negative first-time derivative. The threshold for noise
    is considered 1/2th the standard deviation of the derivative.

    :param traces: An M neuron by N sample matrix
    :type traces: numpy.ndarray
    :param timeout: timeout distance for peak finding (frames)
    :type timeout: int
    :param frame_rate: frame rate / time step for trace matrix
    :type frame_rate: float
    :param smooth: boolean indicating whether to smooth first-time derivative
    :type smooth: bool = True
    :param force_nonneg: boolean indicating whether to enforce non-negativity constraint on first-time derivative
    :type force_nonneg: bool = True
    :return: An iterable where each element contains a sequence of frames identified as event onsets
    :rtype: Tuple[List[int]]
    """
    delta = np.zeros_like(traces)
    delta[..., 1:] = np.diff(traces, axis=-1)

    if smooth:
        gaussian_filter1d(delta, sigma=frame_rate, output=delta)

    if force_nonneg:
        delta[delta[..., :] <= 0] = 0

    noise_threshold = np.std(delta, axis=-1) / 2

    if delta.ndim == 1:
        return find_peaks(delta, prominence=noise_threshold, distance=timeout)[0].tolist()
    else:
        return [find_peaks(delta[neuron, :], prominence=noise_threshold[neuron], distance=timeout)[0].tolist()
                for neuron in range(traces.shape[0])]


def normalize_firing_rates(firing_matrix: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Normalize firing rates by scaling to a max of 1.0. Non-negativity constrained.

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
        instantaneous firing rate
    :type firing_matrix: numpy.ndarray
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: normalized firing rate matrix of n neurons x m samples
    :rtype: numpy.ndarray
    """
    if in_place:
        normalized_matrix = firing_matrix
    else:
        normalized_matrix = firing_matrix.copy()

    normalized_matrix /= np.max(normalized_matrix, axis=0)
    normalized_matrix[normalized_matrix <= 0] = 0
    return normalized_matrix


def scale_waveforms(waveforms: Iterable[np.ndarray], scaler: Callable = StandardScaler) -> np.ndarray:
    """
    Scale waveforms for cross-neuron comparisons

    :param waveforms: An Iterable of M events by N samples matrices of waveforms
    :type waveforms: numpy.ndarray
    :param scaler: sklearn preprocessing object
    :type scaler: Callable
    :return: An Iterable of M event by N samples scaled matrices of waveforms
    :rtype: Iterable[numpy.ndarray]
    """
    return tuple([_scale_waveforms(waves, scaler) for waves in waveforms])


def _collect_waveforms(trace: np.ndarray, event_index: Iterable[int], pre: int = 150, post: int = 450) -> np.ndarray:
    """
    Collect waveforms for each event

    :param trace: a fluorescent trace
    :type trace: numpy.ndarray
    :param event_index: a list of events
    :type event_index: Iterable[int]
    :param pre: number of pre-event frames
    :type pre: int
    :param post: number of post-event frames
    :type post: int
    :return: a matrix of M events x N samples
    :rtype: numpy.ndarray
    """
    waveforms = []  # Pre-allocate empty list. Not need to preallocate numpy array this is fine.
    frames_in_waveform = pre + post + 1
    total_frames = trace.shape[0]
    for event in event_index:
        if event - pre < 0:
            wv = trace[0: post + 1]
            wv = np.pad(wv, pad_width=(frames_in_waveform - wv.shape[0], 0), mode="constant")
        elif event + post > total_frames:
            wv = trace[event - pre:]
            wv = np.pad(wv, pad_width=(0, frames_in_waveform - wv.shape[0]), mode="constant")
        else:
            wv = trace[event - pre: event + post + 1]
        waveforms.append(wv)
    return np.vstack(waveforms)


def _estimate_event_duration(initial_value: float, tau: float, noise_threshold: float) -> float:
    """
    Estimate the duration of an event using the decay to the noise_threshold

    :param initial_value: starting value
    :type initial_value: float
    :param tau: decay constant (s)
    :type tau: float
    :param noise_threshold: target value
    :type noise_threshold: float
    :return: estimated duration of event
    :rtype: float
    """
    return -1 * tau * np.log(noise_threshold/initial_value)


def _scale_waveforms(waveforms: np.ndarray, scaler: Callable) -> np.ndarray:
    """
    Scale waveforms for cross-neuron comparisons

    :param waveforms: an M events by N samples matrix of waveforms
    :type waveforms: numpy.ndarray
    :param scaler: sklearn preprocessing object
    :type scaler: Callable
    :return: an M events by N samples scaled matrix of waveforms
    :rtype: numpy.ndarray
    """
    waves = []  # Again this is fine, don't premature optimize
    for wave in range(waveforms.shape[0]):
        scaling = scaler()
        x = np.reshape(waveforms[wave, :], (-1, 1))
        waves.append(scaling.fit_transform(x))
    return np.transpose(np.hstack(waves))
    