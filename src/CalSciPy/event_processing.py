from __future__ import annotations
from typing import Iterable, Callable, Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def calculate_firing_rates(spike_probability_matrix: np.ndarray, frame_rate: float = 30.0, in_place: bool = False) \
        -> np.ndarray:
    """
    Calculate firing rates

    :param spike_probability_matrix: matrix of n neuron x m samples where each element is the probability of a spike
    :param frame_rate: frame rate of dataset
    :param in_place: boolean indicating whether to perform calculation in-place
    :returns: firing matrix of n neurons x m samples where each element is a binary indicating presence of spike event
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
    | instantaneous firing rate
    :returns: 1-D vector of mean firing rates
    """
    return np.nanmean(firing_matrix, axis=1)


def collect_waveforms(traces: np.ndarray, event_indices: Iterable[Iterable[int]], pre: int = 150, post: int = 450) \
        -> Tuple[np.ndarray]:
    """
    Collect waveforms for each event

    :param traces: a matrix of M neurons x N samples
    :param event_indices: a list of events
    :param pre: number of pre-event frames
    :param post: number of post-event frames
    :returns: a matrix of M events x N samples
    """
    return tuple([_collect_waveforms(traces[neuron, :], event_indices[neuron], pre, post)
                  for neuron in range(traces.shape[0])])


def convert_tau(tau: float, dt: float) -> float:
    """
    Converts a discrete tau to a continuous tau

    :param tau: decay constant
    :param dt: time step (s)
    :returns: continuous tau (s)
    """
    ctau = np.roots(np.hstack([1, -tau]))
    if ctau < 0:
        ctau = 0
    ctau = np.log(ctau) / dt
    return -1 / ctau


def get_num_events(event_indices: Iterable[Iterable[int]]) -> np.ndarray:
    """
    Determines the number of events for each neuron in the event indices

    :param event_indices: An iterable of length M neurons containing a sequence with a duration for each event
    :returns: A 1-D vector of length M neurons containing the number of events for each neuron
    """
    return np.array([len(event_index) for event_index in event_indices])


def get_inter_event_intervals(event_indices: Iterable[Iterable[int]], frame_rate: float = 30.0) -> Tuple[np.ndarray]:
    """
    Calculate the inter event intervals for each neuron in the event indices

    :param event_indices: An iterable of length M containing a sequence with a duration for each event
    :param frame_rate: frame_rate for trace matrix
    :returns: An iterable of length M neurons containing the inter-event intervals for each event in the sequence
    """
    return tuple([np.diff(events) / frame_rate for events in event_indices])


def get_event_onset_intensities(traces: np.ndarray, event_indices: Iterable[Iterable[int]]) -> Tuple[np.ndarray]:
    """
    Retrieve the signal intensity at event onset for each neuron in the event indices

    :param traces: An M neuron by N sample matrix
    :param event_indices: An iterable of length M containing a sequence with a duration for each event
    :returns: An iterable of length M neurons containing the onset intensities for each event in the sequence
    """
    intensities = []
    for neuron in range(traces.shape[0]):
        intensities.append([traces[neuron, event] for event in event_indices[neuron]])
    return tuple(intensities)


def identify_events(traces: np.ndarray, timeout: int = 15, frame_rate: float = 30.0, smooth: bool = True,
                    force_nonneg: bool = True) \
        -> Tuple[List[int]]:
    """
    Identify event onset for each neuron using the smoothed, non-negative first-time derivative. The threshold for noise
    is considered 1/2th the standard deviation of the derivative.

    :param traces: An M neuron by N sample matrix
    :param timeout: timeout distance for peak finding (frames)
    :param frame_rate: frame rate / time step for trace matrix
    :param smooth: boolean indicating whether to smooth first-time derivative
    :param force_nonneg: boolean indicating whether to enforce non-negativity constraint on first-time derivative
    :returns: An iterable where each element contains a sequence of frames identified as event onsets
    """
    delta = np.zeros_like(traces)
    delta[..., 1:] = np.diff(traces, axis=-1)

    if smooth:
        gaussian_filter1d(delta, sigma=frame_rate, output=delta)

    if force_nonneg:
        delta[delta[..., :] <= 0] = 0

    noise_threshold = np.nanstd(delta, axis=-1) / 2

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
    :param in_place: boolean indicating whether to perform calculation in-place
    :returns: normalized firing rate matrix of n neurons x m samples
    """
    if in_place:
        normalized_matrix = firing_matrix
    else:
        normalized_matrix = firing_matrix.copy()

    normalized_matrix /= np.nanmax(normalized_matrix, axis=0)
    normalized_matrix[normalized_matrix <= 0] = 0
    return normalized_matrix


def scale_waveforms(waveforms: Iterable[np.ndarray], scaler: Callable = StandardScaler) -> np.ndarray:
    """
    Scale waveforms for cross-neuron comparisons

    :param waveforms: An Iterable of M events by N samples matrices of waveforms
    :param scaler: sklearn preprocessing object
    :returns: An Iterable of M event by N samples scaled matrices of waveforms
    """
    return tuple([_scale_waveforms(waves, scaler) for waves in waveforms])


def bin_data(data: Union[pd.DataFrame, np.ndarray, Iterable], bin_length: int, fun: Callable) \
        -> Union[pd.DataFrame, np.ndarray]:

    # record if pd
    if isinstance(data, (pd.DataFrame, pd.Series, pd.Index)):
        is_pd = True
    elif isinstance(data, (np.ndarray, Iterable)):
        is_pd = False
    else:
        raise TypeError(print(f"Argument data must be DataFrame, Series, Index, numpy array, or Iterable not "
                              f"{type(data)}"))

    # convert / copy if necessary
    if is_pd is True:
        data_ = data.copy(deep=True)
        index_name = data_.index.name
    else:
        data_ = pd.DataFrame(data=data, columns=[str(x) for x in range(data.shape[1])])
        data_.index.name = "Index"
        index_name = "Index"

    idx = data_.index.to_numpy()

    bins = np.arange(idx[0], idx[-1], bin_length)

    if bins[-1] == idx[-1]:
        bins[-1] += 1
    else:
        bins = np.append(bins, idx[-1] + 1)

    data_.reset_index(drop=False, inplace=True)

    if fun == "median":
        data_ = data_.groupby(pd.cut(data_[index_name], bins=bins, include_lowest=True, right=False)).median()
    elif fun == "mean":
        data_ = data_.groupby(pd.cut(data_[index_name], bins=bins, include_lowest=True, right=False)).mean()
    elif fun == "sum":
        data_ = data_.groupby(pd.cut(data_[index_name], bins=bins, include_lowest=True, right=False)).sum()
    elif fun == "std":
        data_ = data_.groupby(pd.cut(data_[index_name], bins=bins, include_lowest=True, right=False)).std()
    else:
        data_ = data_.groupby(pd.cut(data_[index_name], bins=bins, include_lowest=True, right=False)).apply(fun)

    data_.drop(labels=index_name, axis="columns", inplace=True)

    if is_pd:
        return data_
    else:
        return data_.to_numpy(copy=True)


def _collect_waveforms(trace: np.ndarray, event_index: Iterable[int], pre: int = 150, post: int = 450) -> np.ndarray:
    """
    Collect waveforms for each event

    :param trace: a fluorescent trace
    :param event_index: a list of events
    :param pre: number of pre-event frames
    :param post: number of post-event frames
    :returns: a matrix of M events x N samples
    """
    waveforms = []  # Pre-allocate empty list. Not need to preallocate numpy array this is fine.
    frames_in_waveform = pre + post + 1
    total_frames = trace.shape[0]
    for event in event_index:
        if event - pre < 0:
            wv = trace[0: post + 1]
            wv = np.pad(wv, pad_width=(frames_in_waveform - wv.shape[0], 0), mode="constant", constant_values=np.nan)
        elif event + post >= total_frames:
            wv = trace[event - pre:]
            wv = np.pad(wv, pad_width=(0, frames_in_waveform - wv.shape[0]), mode="constant", constant_values=np.nan)
        else:
            wv = trace[event - pre: event + post + 1]
        waveforms.append(wv)
    return np.vstack(waveforms)


def _scale_waveforms(waveforms: np.ndarray, scaler: Callable) -> np.ndarray:
    """
    Scale waveforms for cross-neuron comparisons

    :param waveforms: an M events by N samples matrix of waveforms
    :param scaler: sklearn preprocessing object
    :returns: an M events by N samples scaled matrix of waveforms
    """
    waves = []  # Again this is fine, don't premature optimize
    for wave in range(waveforms.shape[0]):
        scaling = scaler()
        x = np.reshape(waveforms[wave, :], (-1, 1))
        waves.append(scaling.fit_transform(x))
    return np.transpose(np.hstack(waves))
