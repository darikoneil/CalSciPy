from __future__ import annotations
from math import hypot
from typing import Tuple, Iterable
import pickle as pkl
from pathlib import Path
from itertools import product
from scipy.stats import linregress, pearsonr, spearmanr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

from CalSciPy.color_scheme import COLORS
from CalSciPy.organization.experiment import Experiment
# noinspection PyProtectedMember
from CalSciPy._user import select_directory, verbose_copying
from CalSciPy.organization.files import FileTree
# noinspection PyProtectedMember
from CalSciPy._validators import convert_permitted_types_to_required

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import gridspec


class TraceFearExperiment(Experiment):
    def __init__(self, name: str, base_directory: Path, **kwargs):
        """
        Trace Fear Experiment mix-in

        :param name: name of experiment
        :type name: str
        :param base_directory: base directory of mouse
        :type base_directory: Path
        :key mix_ins: an iterable of mix-ins in string or object form
        """
        # noinspection PyArgumentList
        super().__init__(name, base_directory, **kwargs)

    def collect_data(self) -> None:
        """
        Implementation of abstract method for data collection

        :rtype: Experiment
        """
        behavior_directory = \
            select_directory(title="Select folder containing raw behavioral data")
        verbose_copying(Path(behavior_directory), self.file_tree.get("behavior")(),
                        content_string="behavior data")
        self.reindex()
        super().collect_data()

    def analyze_data(self) -> None:
        """
        Implementation of abstract method for analyzing data

        :rtype: Experiment
        """

        # reindex to grab any files produced from imaging experiment
        self.reindex()

        # first data directly from behavior recording
        behavior_data = load_behavior_data(self.file_tree)
        behavior_data.to_csv(self.file_tree.get("results")().joinpath(f"{self._name}_behavior.csv"))

        # next data from behavioral state machine
        stimulus_file = self.file_tree.get("behavior").find_matching_files("*Stimulus*")[-1]
        stimulus_data = load_stimulus_data(stimulus_file)
        stimulus_data.to_csv(self.file_tree.get("results")().joinpath(f"{self._name}_stimulus.csv"))

        # load acquisition data
        try:
            acquisition_data = pd.read_csv(self.file_tree.get("results")("acquisition"), index_col=0)
        except Exception:
            acquisition_data = None

        # load position data
        position_data = load_video_position(self.file_tree)
        position_data.to_csv(self.file_tree.get("results")().joinpath(f"{self._name}_position.csv"))

        # next generate the aligned data
        aligned_data = align_data(acquisition_data=acquisition_data,
                                  behavior_data=behavior_data,
                                  stimulus_data=stimulus_data,
                                  position_data=position_data)

        aligned_data.to_csv(self.file_tree.get("results")().joinpath(f"{self._name}_aligned.csv"))

        # max position change
        trials = aligned_data[aligned_data["CS+"] == 1]
        CS_plus = np.unique(trials["Trial Group"])
        max_position_change = calculate_max_pos(aligned_data)
        max_position_change["Valence"] = np.repeat("CS-", max_position_change.shape[0])
        idx = np.where(max_position_change["Trial Group"].isin(CS_plus))[0]
        for i in idx:
            max_position_change["Valence"].iloc[i] = "CS+"
        max_position_change.to_csv(self.file_tree.get("results")().joinpath(f"{self._name}_max_position_change.csv"))

        super().analyze_data()

    def generate_class_files(self) -> None:
        """
        Implementation of abstract method for generating file sets specific to mix-in

        :rtype: Experiment
        """
        self.file_tree.add_path("behavior")
        super().generate_class_files()


def align_acquisition_and_stimulus(acquisition_data: pd.DataFrame, stimulus_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function aligns acquisition data and stimulus info

    :param acquisition_data: data acquired by prairieview
    :type acquisition_data: pandas.DataFrame
    :param stimulus_data: stimulus data
    :type stimulus_data: pandas.DataFrame
    :return: aligned data
    :rtype: pandas.DataFrame
    """

    acquisition_data = binarize_acquisition_data(acquisition_data.copy(deep=True))

    trials_aq = np.where(acquisition_data["Trial"] == 1)[0]
    trials_stim = np.where(stimulus_data["Trial"] == 1)[0]

    # drop redundant column
    acquisition_data = acquisition_data.drop(labels="Trial", axis=1)

    # ensure unique column names / air puff format
    acquisition_data.rename(columns={"UCS": "Air"}, inplace=True)

    # find first trial start
    first_peak_aq = trials_aq[0]
    first_peak_stim = trials_stim[0]

    # determine offset
    offset = first_peak_aq - first_peak_stim

    # alignment
    if offset > 0:
        # stim needs shifted right
        time = np.arange(offset, offset + stimulus_data.shape[0], 1)
        time = pd.DataFrame(data=time, index=stimulus_data.index, columns=["Time (ms)"])
        stimulus_data = stimulus_data.copy(deep=True)
        stimulus_data = stimulus_data.join(time)
        stimulus_data.set_index("Time (ms)", drop=True, inplace=True)
        stimulus_data.sort_index(inplace=True)
        aligned_data = acquisition_data.copy(deep=True)
        aligned_data = aligned_data.join(stimulus_data)
    elif offset < 0:
        # aq needs shifted right
        time = np.arange(np.abs(offset, ), np.abs(offset) + trials_aq.shape[0], 1)
        time = pd.DataFrame(data=time, index=acquisition_data.index, columns=["Relative Time (ms)"])
        acquisition_data = acquisition_data.copy(deep=True)
        acquisition_data = acquisition_data.join(time)
        acquisition_data.set_index("Relative Time (ms)", drop=True, inplace=True)
        acquisition_data.sort_index(inplace=True)
        aligned_data = stimulus_data.copy(deep=True)
        aligned_data = aligned_data.join(acquisition_data)
        aligned_data.index.name = "Time (ms)"
    else:
        # offset is zero, already aligned
        aligned_data = stimulus_data.copy(deep=True)
        aligned_data.index.name = "Time (ms)"
        aligned_data = aligned_data.join(acquisition_data)

    # sort names
    column_names = sorted(aligned_data.columns.tolist())

    return aligned_data[column_names]


def align_behavior_data(aligned_data: pd.DataFrame, behavior_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function aligns the aligned acquisition and stimulus data with behavior data

    :param aligned_data: aligned acquisition and stimulus data
    :type aligned_data: pandas.DataFrame
    :param behavior_data: behavior data
    :type behavior_data: pandas.DataFrame
    :return: aligned data
    :rtype: pandas.DataFrame
    """
    # stages to align. order matters!
    stages = ["Pretrial", "Intertrial", "Habituation", "Trial"]

    # forward fill to make sure all labeled
    bd = behavior_data.copy(deep=True)
    bd.ffill(inplace=True)

    # fill nan to make sure indicator vectors' don't jump from nan to 1
    # and thus break logic of checking derivative for stage start/end
    ad = aligned_data.copy(deep=True)
    ad.fillna(0, inplace=True)

    # preallocate
    stage_data_list = []

    # for each stage
    for stage in stages:
        # get indices for behavior data
        stage_vector = np.zeros_like(bd[stage])
        stage_vector[1:] = np.diff(bd[stage])
        starts = np.where(stage_vector >= 1)[0]
        ends = np.where(stage_vector <= -1)[0]

        # handle edge cases
        if bd[stage][0] == 1:
            starts = np.append(starts, 0)
        if ends.shape[0] == starts.shape[0] - 1:
            ends = np.append(ends, bd.shape[0] + 1)

        # determine how many occurrences of this stage
        behavior_occur = starts.shape[0]

        # get indices for aligned data
        aligned_vector = np.zeros_like(ad[stage])
        aligned_vector[1:] = np.diff(ad[stage])
        aligned_starts = np.where(aligned_vector >= 1)[0]
        aligned_ends = np.where(aligned_vector <= -1)[0]

        # handle edge cases
        if ad[stage][0] == 1:
            aligned_starts = np.append(aligned_starts, 0)
        if aligned_ends.shape[0] == aligned_starts.shape[0] - 1:
            aligned_ends = np.append(aligned_ends, ad.shape[0] + 1)

        # determine how many occurrences of this stage
        aligned_occur = aligned_starts.shape[0]

        # because pre-trials can be dropped if premature, we need to index which ones to drop since the stimulus info
        # only indexes the "true" pre-trials for better or worse
        if stage == "Pretrial" and aligned_occur != behavior_occur:
            # we can find the botched pre-trials by easily by looking
            # for ones that are less than 99% of the aligned duration
            # it's not a perfect measures but what are the chances of of a mouse failing in the final 100 ms?...
            behavior_durations = [ends[trial] - starts[trial] for trial in range(behavior_occur)]
            aligned_duration = np.median([aligned_ends[trial] - aligned_starts[trial] for trial
                                          in range(aligned_occur)])
            # index of true pre-trials. re-used for intertrials because there is no way to know directly there
            true_idx = [trial for trial in range(behavior_occur) if behavior_durations[trial] / aligned_duration
                        >= 0.99]

            # keep only true pretrials
            starts = np.array([starts[idx] for idx in true_idx])
            ends = np.array([ends[idx] for idx in true_idx])

            # re-determine how many occurrences of this stage
            behavior_occur = starts.shape[0]

        elif stage == "Intertrial" and aligned_occur != behavior_occur:

            # since pretrial is before this, we can save the true_idx from above... bad design, but ultimately
            # the simplicity is worth the readability/maintainability cost

            # noinspection PyUnboundLocalVariable
            starts = np.array([starts[idx] for idx in true_idx])
            ends = np.array([ends[idx] for idx in true_idx])

            # re-determine how many occurrences of this stage
            behavior_occur = starts.shape[0]

        # ensure number of occurrences match
        assert (aligned_occur == behavior_occur), f"Behavior & Aligned Data Have Unequal Number of {stage} Occurrences"

        for one_occurrence in range(behavior_occur):

            # get data from behavior data
            start = starts[one_occurrence]
            end = ends[one_occurrence] - 1

            # get index from aligned data
            aligned_start = aligned_starts[one_occurrence]
            aligned_end = aligned_ends[one_occurrence] - 1

            if one_occurrence == behavior_occur - 1:
                # handle edge case
                if end - start > aligned_end - aligned_start:
                    end = start + aligned_end - aligned_start

            # get data
            stage_data = bd.loc[start:end, :].copy(deep=True)
            stage_frames = ad.loc[aligned_start:aligned_end, :].copy(deep=True)

            # match samples
            time = np.round(np.linspace(stage_frames.index[0], stage_frames.index[-1], stage_data.shape[0]))
            time = pd.DataFrame(data=time, index=stage_data.index, columns=["Relative Time (ms)"])
            stage_data = stage_data.join(time)
            stage_data.set_index("Relative Time (ms)", drop=True, inplace=True)
            stage_data.sort_index(inplace=True)
            stage_data.index.name = "Time (ms)"
            stage_data_list.append(stage_data)

    # concatenate all stages into one dataframe
    stage_data = pd.concat(stage_data_list)
    stage_data.sort_index(inplace=True)

    # rename any duplicate column ids - see prune_ad function to clean up the resulting duplicates
    # we keep here for the ability to easily double-check any sync issues
    aligned_names = ad.columns.tolist()
    column_names = stage_data.columns.tolist()
    new_names = [name if name not in aligned_names else "".join([name, "2"]) for name in column_names]
    stage_data.rename(columns={name: new_name for name, new_name in zip(column_names, new_names)}, inplace=True)

    # finally make the aligned dataframe
    final_alignment = aligned_data.copy(deep=True)

    return final_alignment.join(stage_data)


def align_position(aligned_data: pd.DataFrame, position_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function aligns the position data to the aligned data

    :param aligned_data:
    :param position_data:
    :return: aligned data
    """
    # get trial vector
    posd = position_data.copy(deep=True)
    posd_trials = posd["Trial"].to_numpy()

    # fill nan to make sure indicator vectors' don't jump from nan to 1
    # and thus break logic of checking derivative for stage start/end
    ad = aligned_data.copy(deep=True)
    ad.fillna(0, inplace=True)

    # determine how many trials
    num_trials_posd = np.max(posd_trials) + 1

    # get indices for aligned data
    aligned_vector = np.zeros_like(ad["Trial"])
    aligned_vector[1:] = np.diff(ad["Trial"])
    aligned_starts = np.where(aligned_vector >= 1)[0]
    aligned_ends = np.where(aligned_vector <= -1)[0]

    # handle edge cases
    if ad["Trial"][0] == 1:
        aligned_starts = np.append(aligned_starts, 0)
    if aligned_ends.shape[0] == aligned_starts.shape[0] - 1:
        aligned_ends = np.append(aligned_ends, ad.shape[0] + 1)

    # determine how many occurrences of this stage
    aligned_occur = aligned_starts.shape[0]

    # ensure number of trials match
    assert (aligned_occur == num_trials_posd), "Number of trials must match"

    # pre-allocate
    posd_list = []

    for one_trial in range(num_trials_posd):

        # get pd idx
        idx = np.where(posd_trials == one_trial)[0]

        # get index from aligned data
        aligned_start = aligned_starts[one_trial]
        aligned_end = aligned_ends[one_trial] - 1

        # get data
        posd_data = posd.iloc[idx].copy(deep=True)
        ad_data = ad.loc[aligned_start:aligned_end, :].copy(deep=True)

        # match samples
        time = np.round(np.linspace(ad_data.index[0], ad_data.index[-1], posd_data.shape[0]))
        time = pd.DataFrame(data=time, index=posd_data.index, columns=["Time (ms)"])
        posd_data = posd_data.join(time)
        posd_data.set_index("Time (ms)", drop=True, inplace=True)
        posd_data.sort_index(inplace=True)
        posd_data = posd_data.rename(columns={"Trial": "Trial3"})
        posd_data = posd_data.reindex(ad_data.index)
        posd_data.interpolate(method="pchip", inplace=True)
        posd_list.append(posd_data)

    # concatenate all stages into one dataframe
    posd_data = pd.concat(posd_list)
    posd_data.sort_index(inplace=True)

    # keep only position
    posd_data.rename(columns={"position relative baseline (mm)": "Position (mm)"}, inplace=True)
    posd_data = posd_data["Position (mm)"]

    # finally join
    final_alignment = aligned_data.copy(deep=True)
    final_alignment = final_alignment.join(posd_data)

    return final_alignment


def align_data(acquisition_data: pd.DataFrame, behavior_data: pd.DataFrame, stimulus_data: pd.DataFrame,
               position_data: pd.DataFrame) -> pd.DataFrame:

    # first align acquisition & stimulus data
    if acquisition_data is not None:
        aligned_data = align_acquisition_and_stimulus(acquisition_data, stimulus_data)
    else:
        aligned_data = stimulus_data

    # next align behavior data
    aligned_data = align_behavior_data(aligned_data, behavior_data)

    # then align position data
    aligned_data = align_position(aligned_data, position_data)

    # finally return data without duplicate redundant columns
    aligned_data = prune_aligned_data(aligned_data)

    return aligned_data


# noinspection DuplicatedCode
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_analog_data(file: Path, channel_ids: Tuple[str] = ("Sync", "Motor Position", "Dummy", "Force")) \
        -> pd.DataFrame:
    """
        Loads analog data from behavioral recording

        :param file: filepath
        :type file: pathlib.Path
        :param channel_ids: string ids of each channel in recording
        :type channel_ids: Tuple[str] = ("Sync", "Motor Position", "Dummy", "Force")
        :return: analog data from behavioral recording
        :rtype: pandas.DataFrame
        """
    analog = np.load(file)

    # try to ensure everything is same time & float,
    # since NaN values are weird for integers, etc
    analog = analog.astype(np.float64)

    if len(analog.shape) == 2:
        channels, samples = analog.shape
    else:
        channels = 1
        samples = analog.shape[-1]

    assert (channels == len(channel_ids)), \
        "Number of identified channels does not match the number of provided channel ids"

    time = np.arange(0, samples, 1)
    time = pd.Index(data=time, name="Time (ms)")

    return pd.DataFrame(data=analog.T, index=time, columns=channel_ids)


def binarize_acquisition_data(acquisition_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts acquisition data to binary indicator vectors

    :param acquisition_data: acquisition data
    :type acquisition_data: pandas.DataFrame
    :return: binarized acquisition data
    :rtype: pandas.DataFrame
    """
    # form binary trial indicator vectors
    trial = np.zeros_like(acquisition_data["Trial"])
    trial[np.where(acquisition_data["Trial"] >= 3)[0]] = 1

    # form binary ucs indicator vector
    ucs = np.zeros_like(acquisition_data["UCS"])
    ucs[np.where(acquisition_data["UCS"] >= 4.95)[0]] = 1
    # reinsert
    acquisition_data["Trial"] = trial
    acquisition_data["UCS"] = ucs

    return acquisition_data


def load_behavior_data(file_tree: FileTree) -> pd.DataFrame:
    analog_file = file_tree.get("behavior").find_matching_files("*Analog*")[-1]
    analog_data = load_analog_data(analog_file)

    digital_file = file_tree.get("behavior").find_matching_files("*Digital*")[-1]
    digital_data = load_digital_data(digital_file)

    state_file = file_tree.get("behavior").find_matching_files("*State*")[-1]
    state_data = load_state_data(state_file)

    return sync_behavior_data(analog_data, digital_data, state_data)


# noinspection DuplicatedCode
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_digital_data(file: Path, channel_ids: Tuple[str] = ("Gate", )) -> pd.DataFrame:
    """
    Loads digital data from behavioral recording

    :param file: filepath
    :type file: pathlib.Path
    :param channel_ids: string ids of each channel in recording
    :type channel_ids: Tuple[str] = ("Gate", )
    :return: digital data from behavioral recording
    :rtype: pandas.DataFrame
    """
    digital = np.load(file)

    # try to keep everything as same time & float,
    # since NaN values are weird for integers
    digital = digital.astype(np.float64)

    if len(digital.shape) == 2:
        channels, samples = digital.shape
    else:
        channels = 1
        samples = digital.shape[-1]

    assert (channels == len(channel_ids)), \
        "Number of identified channels does not match the number of provided channel ids"

    time = np.arange(0, samples, 1)
    time = pd.Index(data=time, name="Time (ms)")

    return pd.DataFrame(data=digital.T, index=time, columns=channel_ids)


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_state_data(file: Path) -> pd.DataFrame:
    """
    Load states from behavioral recording

    :param file: filepath
    :type file: pathlib.Path
    :return: state data
    :rtype: pandas.DataFrame
    """
    state = np.load(file)
    state = np.array(["".join([chr(_) for _ in state[s]]) for s in range(state.shape[0])])
    samples = state.shape[0]
    # then convert to integer code
    unique_states = np.unique(state).shape[0]
    state_dictionary = {key: value for key, value in zip(np.unique(state).tolist(), range(unique_states))}  # noqa: C416
    state_matrix = np.zeros((samples, unique_states), dtype=np.float64)
    for key, value in state_dictionary.items():
        idx = np.where(state == key)[0]
        state_matrix[idx, value] = 1
    column_names = [key.capitalize() for key in state_dictionary.keys()]
    # now make time index in ms
    state_time = np.arange(0, samples * 100, 100)
    state_idx = pd.Index(data=state_time, name="Time (ms)")
    return pd.DataFrame(data=state_matrix, index=state_idx, columns=column_names)


def load_stimulus_data(stimulus_file: str) -> pd.DataFrame:
    """
    Load stimulus info from behavioral state machine

    :param file_tree: filetree
    :type file_tree: FileTree
    :return: stimulus info
    :rtype: pandas.DataFrame
    """
    try:
        with open(stimulus_file, "r") as f:
            stimulus_info = pkl.load(f)
    except TypeError:
        with open(stimulus_file, "rb") as f:
            stimulus_info = pkl.load(f, encoding="bytes")
            stimulus_info = {key.decode("utf-8"): value for key, value in stimulus_info.items()}

    return organize_stimulus_data(stimulus_info)


def organize_stimulus_data(stimulus_info: dict, trace_duration: int = 10000) -> pd.DataFrame:
    # Here are all states -- unfortunately we're hardcoding these
    stages = ["Habituation", "Intertrial", "Pretrial", "Trial"]

    stage_sets = {
        "Habituation": ("habStart", "habEnd"),
        "Intertrial": ("interStart", "interEnd"),
        "Pretrial": ("preStart", "preEnd"),
        "Trial": ("trialStart", "trialEnd"),
    }

    stims = ["CS+", "CS-", "UCS"]

    stim_sets = {
        "CS+": ("cs_plus_start", "cs_plus_end"),
        "CS-": ("cs_minus_start", "cs_minus_end"),
        "UCS": ("ucsStart", "ucsEnd"),
    }

    states = (*stages, *stims)
    sets = {**stage_sets, **stim_sets}

    # finally make separate lists for cs+, cs-, ucs
    cs_index = np.array(stimulus_info.get("stimulusTypes"))
    cs_plus = np.where(cs_index == 0)[0]
    cs_minus = np.where(cs_index == 1)[0]
    ucs = np.where(cs_index == 0)[0]

    stimulus_info["cs_plus_start"] = np.array(stimulus_info.get("csStart"))[cs_plus].tolist()
    stimulus_info["cs_plus_end"] = np.array(stimulus_info.get("csEnd"))[cs_plus].tolist()
    stimulus_info["cs_minus_start"] = np.array(stimulus_info.get("csStart"))[cs_minus].tolist()
    stimulus_info["cs_minus_end"] = np.array(stimulus_info.get("csEnd"))[cs_minus].tolist()
    stimulus_info["ucsStart"] = np.array(stimulus_info.get("ucsStart"))[ucs].tolist()
    stimulus_info["ucsEnd"] = np.array(stimulus_info.get("ucsEnd"))[ucs].tolist()

    # first make all data relative to first recorded stimulus/stage
    offset = stimulus_info.get("habStart")[0] * 1000
    end = stimulus_info.get("trialEnd")[-1] * 1000
    end -= offset

    # get relative time index
    time = np.arange(0, end, 1)
    time = pd.Index(data=time, name="Relative Time (ms)")

    # fill indicator matrix
    indicator_matrix = np.zeros((time.shape[0], len(states)), dtype=np.float64)
    for state, idx in zip(states, range(len(states))):
        state_set = sets.get(state)
        for start, end in zip(stimulus_info.get(state_set[0]), stimulus_info.get(state_set[1])):
            # convert to ms
            start *= 1000
            end *= 1000
            # make timing relative
            start -= offset
            end -= offset
            # coerce to int
            start = int(start)
            end = int(end)
            indicator_matrix[start:end, idx] = 1

    stimulus_data = pd.DataFrame(data=indicator_matrix, index=time, columns=states)

    # add trial group indicator
    trial_vector = stimulus_data["Trial"].to_numpy(copy=True)  # binary vector indicating trial
    delta = np.zeros_like(trial_vector)
    delta[1:] = np.diff(trial_vector)  # can get start/end times from first-derivative + 1
    start = np.where(delta >= 1)[0]
    end = np.where(delta <= -1)[0]
    if end.shape[0] == start.shape[0] - 1:
        end = np.append(end, delta.shape[0])
    trials = start.shape[0]
    trial_vector[:] = np.nan
    for trial in range(trials):
        trial_vector[start[trial]:end[trial]] = trial
    trial_group = pd.DataFrame(data=trial_vector, index=time, columns=["Trial Group"])
    trial_group.bfill(inplace=True)  # fill backwards to collect other stages
    trial_group["Trial Group"][stimulus_data["Habituation"] == 1] = np.nan  # nan out habituation
    stimulus_data = stimulus_data.join(trial_group)

    # add extrapolated indicators

    labels = ["Trace", "Response"]
    timings = [(0, 10000), (10000, 15000)]

    parameters = zip(labels, timings)
    conditions = ["+", "-"]

    for (label, timing), condition in product(parameters, conditions):
        indicator_name = "".join([label, condition])
        reference_name = "".join(["CS", condition])
        start_offset, duration = timing
        delta = np.zeros_like(stimulus_data[reference_name])
        indicator_vector = np.zeros_like(delta)
        delta[1:] = np.diff(stimulus_data[reference_name])
        ends = np.where(delta <= -1)[0]
        for end in ends:
            start = end + start_offset
            stop = start + duration
            indicator_vector[start:stop] = 1
        stimulus_data[indicator_name] = indicator_vector

    valence_vector = np.empty_like(stimulus_data["Trial"])
    valence_vector[:] = np.nan
    # cs_plus
    for trial in cs_plus:
        valence_vector[np.where((stimulus_data["Trial"] == 1) & (stimulus_data["Trial Group"] == trial))[0]] = 1
    # cs minus
    for trial in cs_minus:
        valence_vector[np.where((stimulus_data["Trial"] == 1) & (stimulus_data["Trial Group"] == trial))[0]] = 0

    stimulus_data["Valence"] = valence_vector

    column_names = sorted(stimulus_data.columns.tolist())

    return stimulus_data[column_names]


def prune_aligned_data(aligned_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function prunes some duplicate or unnecessary columns from the aligned data

    :param aligned_data: aligned data containing acquisition, behavior, and stimulus data
    :type aligned_data: pandas.DataFrame
    :return: aligned data
    :rtype: pandas.DataFrame
    """
    # drop some labels
    column_names = aligned_data.columns.tolist()
    drop_labels = ["Habituation2", "Pretrial2", "Intertrial2", "Trial2", "Retracting", "Releasing",
                   "Retractingforcooldown", "Sync", "Setup", "Trial3", "is_trial"]
    keeper_names = [name for name in column_names if name not in drop_labels]
    aligned_data = aligned_data[keeper_names]

    # sort
    column_names = sorted(aligned_data.columns.to_list())
    return aligned_data[column_names]


def sync_behavior_data(analog_data: pd.DataFrame, digital_data: pd.DataFrame, state_data: pd.DataFrame) \
        -> pd.DataFrame:

    behavior_data = analog_data.copy(deep=True)

    assert (behavior_data.shape[0] == digital_data.shape[0]), \
        "Analog & Digital Data must contain same number of samples"
    behavior_data = behavior_data.join(digital_data)

    # adjust gate vector from 1 == egress to 1 == ingress
    gate_vector = np.zeros_like(behavior_data["Gate"])
    gate_vector[np.where(behavior_data["Gate"] == 0)[0]] = 1
    behavior_data["Gate"] = gate_vector

    try:
        assert (behavior_data.shape[0] == state_data.shape[0] * 100)
    except AssertionError:
        target_length = int(behavior_data.shape[0] / 100)
        current_length = state_data.shape[0]
        end_frame = []
        for _ in range(target_length - current_length):
            end_frame.append(state_data.to_numpy()[-1])
        time = np.arange(0, target_length * 100, 100)
        time = pd.Index(data=time, name="Time (ms)")
        state_data = pd.DataFrame(data=np.vstack([state_data.to_numpy(), end_frame]), index=time,
                                  columns=list(state_data.columns))

    state_data = state_data.reindex(behavior_data.index)
    behavior_data = behavior_data.join(state_data)

    column_names = sorted(behavior_data.columns.tolist())
    column_names = [name for name in column_names if name not in ["0", "Dummy"]]
    behavior_data = behavior_data[column_names]

    return behavior_data


"""
DEEP LAB CUT
"""


def load_frame_ids(file: Path) -> np.ndarray:
    """
    This function loads the frame ids mapping each video frame to its respective trial

    :param file: path to file
    :return: an array of frame ids
    """
    # load
    frame_ids = np.load(file)

    # make sure zero-indexed
    while np.min(frame_ids) >= 1:
        frame_ids -= 1

    return frame_ids


def load_video_trial_data(file: Path) -> pd.DataFrame:
    """
    This function loads video trial data

    :param file: path to file
    :return: a dataframe
    """
    # Load
    trial_data = pd.read_csv(file, header=2, index_col=0)

    # Label Index
    trial_data.index.name = "Frame #"

    # Zero-Index Marker Labels
    labels = ["x", "y", "likelihood"]
    column_names = trial_data.columns.to_list()
    num_markers = len(column_names) // 3
    new_names = ["".join([label, str(marker)]) for marker, label in product(range(num_markers), labels)]
    trial_data.rename(columns={name: new_name for name, new_name in zip(column_names, new_names)}, inplace=True)

    return trial_data


def load_video_position(file_tree: FileTree, xnil=(58, 762), ynil=(47, 40), empirical_length=140.0) \
        -> pd.DataFrame:
    """
    This function loads video position estimated by deep lab cut

    :param file_tree: file tree
    :param xnil:
    :param ynil:
    :param empirical_length:
    :return: a dataframe
    """
    # load frame ids
    pre_trial_frame_ids = file_tree.get("behavior").find_matching_files("*PreTrial_FramesIDs*npy")[0]
    pre_trial_frame_ids = load_frame_ids(pre_trial_frame_ids)
    trial_frame_file = file_tree.get("behavior").find_matching_files("*Trial_FrameIDs*npy")[0]
    trial_frame_ids = load_frame_ids(trial_frame_file)

    # load data
    pre_trial_file = file_tree.get("behavior").find_matching_files("*PRETRIALSDLC_resnet*csv*")[0]
    pre_trial_data = load_video_trial_data(pre_trial_file)
    trial_file = file_tree.get("behavior").find_matching_files("*_TRIALSDLC_resnet*csv*")[0]
    trial_data = load_video_trial_data(trial_file)

    # assign data to specific trials
    pre_trial_vector = pd.DataFrame(data=pre_trial_frame_ids, index=pre_trial_data.index, columns=["Trial"])
    pre_trial_label = pd.DataFrame(data=np.repeat(False, pre_trial_frame_ids.shape[0]), index=pre_trial_data.index, columns=["is_trial"])
    trial_vector = pd.DataFrame(data=trial_frame_ids, index=trial_data.index, columns=["Trial"])
    trial_label = pd.DataFrame(data=np.repeat(True, trial_frame_ids.shape[0]), index=trial_data.index, columns=["is_trial"])

    # join with trial data
    pre_trial_data = pre_trial_data.join(pre_trial_vector)
    pre_trial_data = pre_trial_data.join(pre_trial_label)
    trial_data = trial_data.join(trial_vector)
    trial_data = trial_data.join(trial_label)

    # access integrity
    num_bad_frames_pre_trial = np.max([np.where(pre_trial_data[likelihood_label].to_numpy() <= 0.99)[0].shape[0]
                                        for likelihood_label in pre_trial_data.columns.to_list()
                                        if "likelihood" in likelihood_label])
    num_bad_frames_trial = np.max([np.where(trial_data[likelihood_label].to_numpy() <= 0.99)[0].shape[0]
                             for likelihood_label in trial_data.columns.to_list()
                             if "likelihood" in likelihood_label])
    assert (num_bad_frames_pre_trial + num_bad_frames_trial <= np.round(trial_data.shape[0] * 0.005)), \
        f"Inadequate Position Estimation: {num_bad_frames_pre_trial+num_bad_frames_trial=}"

    # remove bad frames if necessary
    likelihood_labels = [likelihood_label for likelihood_label in trial_data.columns.to_list()
                         if "likelihood" in likelihood_label]
    for likelihood_label in likelihood_labels:
        pre_trial_likelihood_vector = pre_trial_data[likelihood_label].to_numpy(copy=True)
        pre_trial_likelihood_vector[np.where(pre_trial_likelihood_vector <= 0.95)[0]] = np.nan
        pre_trial_data[likelihood_label] = pre_trial_likelihood_vector
        trial_likelihood_vector = trial_data[likelihood_label].to_numpy(copy=True)
        trial_likelihood_vector[np.where(trial_likelihood_vector <= 0.95)[0]] = np.nan
        trial_data[likelihood_label] = trial_likelihood_vector

    # merge
    merged_trials = []
    for trial in range(np.max(trial_frame_ids) + 1):
        pre_trial_data_ = pre_trial_data[pre_trial_data["Trial"] == trial]
        trial_data_ = trial_data[trial_data["Trial"] == trial]
        merged_trials.append(pd.concat([pre_trial_data_, trial_data_], ignore_index=True))
    merged_data = pd.concat(merged_trials, ignore_index=True)

    # fit burrow position
    merged_data = fit_burrow_position(merged_data)

    # vectorize
    merged_data = vectorize_coordinates(merged_data)

    # load calibration
    with open(file_tree.get("behavior").find_matching_files("*calibration*")[0], "r") as file:
        lines = file.readlines()
    for line in lines:
        exec(line)

    # transform to mm
    trial_data = transform_to_physical(merged_data, xnil, ynil)

    # joint
    trial_data = average_markers(merged_data)

    trial_data = relative_pre_baseline(trial_data)

    return trial_data


def fit_burrow_position(trial_data):

    def burrow_objective(x_, a_, b_):
        return a_ * x_ + b_

    num_markers = np.sum([1 for name in trial_data.columns.to_list() if "x" in name])

    for marker in range(num_markers):
        x_label = "".join(["x", str(marker)])
        y_label = "".join(["y", str(marker)])
        x = trial_data[x_label].to_numpy(copy=True)
        y = trial_data[y_label].to_numpy(copy=True)
        # noinspection PyTupleAssignmentBalance
        ab, _ = curve_fit(burrow_objective, x, y)
        a, b = ab
        estimate_y = burrow_objective(x, a, b)
        trial_data[x_label] = x
        trial_data[y_label] = estimate_y

    return trial_data


def vectorize_coordinates(trial_data):
    num_markers = np.sum([1 for name in trial_data.columns.to_list() if "x" in name])

    for marker in range(num_markers):
        x_label = "".join(["x", str(marker)])
        y_label = "".join(["y", str(marker)])
        new_label = "".join(["position", str(marker)])
        x = trial_data[x_label].to_numpy(copy=True)
        y = trial_data[y_label].to_numpy(copy=True)
        h = [hypot(x, y) for x, y in zip(x.tolist(), y.tolist())]
        trial_data[new_label] = h

    return trial_data


def transform_to_physical(trial_data, xnil, ynil, empirical_length: float = 128):
    xl, xr = xnil
    yl, yr = ynil
    x = np.abs(xl - xr)
    y = np.abs(yl - yr)
    h = hypot(x, y)
    scaler = MinMaxScaler(feature_range=(0, empirical_length)).fit(np.array([0, h]).reshape(-1, 1))

    position_labels = [name for name in trial_data.columns.to_list() if "position" in name]
    for position in position_labels:
        trial_data[position] = scaler.transform(trial_data[position].to_numpy(copy=True).reshape(-1, 1))

    return trial_data


def average_markers(trial_data):
    position_labels = [name for name in trial_data.columns.to_list() if "position" in name]
    position_data = trial_data[position_labels].to_numpy(copy=True)
    position_data = np.mean(position_data, axis=1)
    trial_data["Position (mm)"] = position_data
    return trial_data


def relative_total_baseline(trial_data):
    joint_baseline = trial_data["joint position (mm)"].to_numpy(copy=True)
    baseline_data = trial_data[trial_data["is_trial"] == False]
    baseline_data = baseline_data["joint position (mm)"].to_numpy(copy=True)
    baseline = np.nanpercentile(baseline_data, 92)
    joint_baseline -= baseline
    joint_baseline *= -1
    trial_data["position relative baseline (mm)"] = joint_baseline
    return trial_data


def relative_pre_baseline(trial_data):
    for trial_ in range(np.max(trial_data["Trial"])+1):
        trial_idx = np.where(trial_data["Trial"]==trial_)[0]
        trial_data_ = trial_data[trial_data["Trial"]==trial_]
        joint_baseline = trial_data_["Position (mm)"].to_numpy(copy=True)
        baseline = np.median(joint_baseline[456-150:456])
        joint_baseline -= baseline
        trial_data.iloc[trial_idx, -1] = joint_baseline
    return trial_data


def calculate_max_pos(trial_data):
    dsp = []
    for trial_ in range(int(np.max(trial_data["Trial Group"])+1)):
        trial_pos = trial_data[trial_data["Trial Group"] == trial_]
        first_trial = int(np.where(trial_pos["Trial"].to_numpy() == 1)[0][0])
        trial_pos = trial_pos["Position (mm)"].to_numpy()
        dsp_ = np.nanmin(trial_pos[first_trial:])
        dsp.append(dsp_)
    dsp = np.vstack(dsp)
    max_pos_change = pd.DataFrame(data=dsp, columns=["Max Position Change (mm)"])
    max_pos_change["Trial Group"] = np.arange(0, dsp.shape[0])
    return max_pos_change


def file_tag_conversion(files):
    for filename in files:
        sub_str = str(filename.name).split("_")
        if len(sub_str[-4]) == 1:
            sub_str[-4] = "0" + sub_str[-4]
            filename.rename(filename.with_name("".join([sub_str[i] if i == 0 else "_" + sub_str[i] for i in range(len(sub_str))])))


"""
Analysis functions
"""


