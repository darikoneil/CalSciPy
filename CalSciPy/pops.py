from __future__ import annotations
from typing import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from memoization import cached, CachingAlgorithmFlag
from binner import bin_data


def cache_population(population):
    attrs = (
        "neurons",
        "total_samples",
        "trials",
        "variables",
        "condition",
        "bin_length",
        "bin_method"
    )
    return hash(f"{[getattr(population, attr) for attr in attrs]}")


def cache_population_reshape(population, data):
    attrs = (
        "neurons",
        "total_samples",
        "trials",
        "variables",
        "condition",
        "bin_length",
        "bin_method"
    )
    pop_cache = tuple([getattr(population, attr) if attr != "variables"
                  else tuple(getattr(population, attr))
                  for attr in attrs])
    data_cache = data.shape

    return hash(pop_cache), hash(data_cache)


class Population:
    def __init__(self,
                 features: np.ndarray,
                 labels: pd.DataFrame,
                 bin_length: int = 1,
                 bin_method: str = "mean",
                 condition: str = None,
                 meta: Mapping = None):

        # preallocate properties
        self._condition = None
        self._meta = None

        # init
        self._features = features
        self._labels = labels
        self.bin_length = bin_length
        self.bin_method = bin_method
        self.condition = condition
        self.meta = meta

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, value):
        self._condition = value

    @property
    def responses(self):
        return self._reshape_data(self.features)

    @property
    def indicators(self):
        return self._reshape_data(self.labels)

    @property
    def features(self):
        return self._organize_features()

    @property
    def labels(self):
        return self._organize_labels()

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        if value is not None:
            self._meta = MappingProxyType(value)

    @property
    def neurons(self):
        return self._features.shape[1]

    @property
    def samples_per_trial(self):
        return self.features.shape[-1]

    @property
    def total_samples(self):
        return self._labels.shape[0]

    @property
    def trials(self):
        return self._features.shape[0]

    @property
    def trial_conditions(self):
        if self.condition is None:
            return [0] * self.trials
        else:
            condition_index = self.variables.index(self.condition)
            condition_vector = self.labels[:, condition_index, :]
            return np.nanmedian(condition_vector, axis=-1)

    @property
    def trial_types(self):
        return np.unique(self.trial_conditions).tolist()

    @property
    def variables(self):
        return self._labels.columns.tolist()

    @cached(custom_key_maker=cache_population, max_size=16, algorithm=CachingAlgorithmFlag.FIFO)
    def _organize_features(self):
        if self.bin_length > 1:
            return np.dstack([bin_data(self._features[trial, :, :].T, self.bin_length, self.bin_method)
                              for trial in range(self.trials)]).swapaxes(0, 2)
        else:
            return self._features

    @cached(custom_key_maker=cache_population, max_size=16, algorithm=CachingAlgorithmFlag.FIFO)
    def _organize_labels(self):
        trial_index = self._labels.columns.tolist().index("Trial #")
        labels = self._labels.to_numpy(copy=True)
        unique_variables = len(self.variables)
        samples_per_trial = self.total_samples // self.trials

        organized_labels = np.zeros((self.trials, unique_variables, samples_per_trial))

        for trial in range(self.trials):
            organized_labels[trial, :, :] = labels[np.where(labels[:, trial_index] == trial)[0], :].T

        return np.dstack([bin_data(organized_labels[trial, :, :].T, self.bin_length, fun="median")
                          for trial in range(self.trials)]).swapaxes(0, 2)

    @cached(custom_key_maker=cache_population_reshape, max_size=32, algorithm=CachingAlgorithmFlag.FIFO)
    def _reshape_data(self, data):
        if self.condition is None:
            return data.reshape(1, *data.shape)
        else:
            reshaped_data = np.zeros((len(self.trial_types), self.trials//len(self.trial_types), *data.shape[1:]),
                                     dtype=data.dtype)

            for idx, trial_type in enumerate(self.trial_types):
                reshaped_data[idx, :, :, :] = data[np.where(self.trial_conditions == trial_type)[0], :, :]

        return reshaped_data

    def __str__(self):
        return f"Neural population containing {self.neurons} neurons over {self.trials} trials"


class Pseudopopulation:
    def __init__(self,
                 populations=None,
                 bin_length=1,
                 bin_method="mean",
                 condition=None):

        # pre-allocate
        self._condition = None
        self._bin_length = None
        self._bin_method = None

        self.populations = populations if populations is not None else list()
        self.bin_length = bin_length
        self.bin_method = bin_method
        self.condition = condition

    @property
    def bin_length(self):
        return self._bin_length

    @bin_length.setter
    def bin_length(self, value):
        if value is not None and self.populations is not None:
            # update all bin_lengths
            for population in self.populations:
                population.bin_length = value
            self._bin_length = value

    @property
    def bin_method(self):
        return self._bin_method

    @bin_method.setter
    def bin_method(self, value):
        if value is not None and self.populations is not None:
            # update all bin methods
            for population in self.populations:
                population.bin_method = value
            self._bin_method = value

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, value):
        if value is not None and self.populations is not None:
            # set conditions for all pops
            for population in self.populations:
                population.condition = value
            self._condition = value

    @property
    def features(self):
        if self.total_populations >= 1:
            sample_slice = self._slice_trial()
            data = [population.responses[:, :, :, :sample_slice] for population in self.populations]
            if self.total_populations == 1:
                return data[0]
            else:
                return np.concatenate(data, axis=2)
        else:
            return None

    @property
    def labels(self):
        if self.total_populations >= 1:
            sample_slice = self._slice_trial()
            return self.populations[0].indicators[:, :, :, :sample_slice]
        else:
            return None

    @property
    def neuron_counts(self):
        return [population.neurons for population in self.populations]

    @property
    def total_neurons(self):
        return np.sum(self.neuron_counts)

    @property
    def total_populations(self):
        return len(self.populations)

    @property
    def total_trials(self):
        return np.sum(self.trial_counts)

    @property
    def trial_counts(self):
        return [population.trials for population in self.populations]

    @property
    def trial_types(self):
        trial_types = [population.trial_types for population in self.populations]
        assert(all(one_pop_types == trial_types[0] for one_pop_types in trial_types))
        return trial_types[0]

    @property
    def variables(self):
        variables = [population.variables for population in self.populations]
        assert(all(one_pop_var == variables[0] for one_pop_var in variables))
        return variables[0]

    def _slice_trial(self):
        trial_lengths = {population.samples_per_trial for population in self.populations}
        if len(trial_lengths) > 1:
            return min(trial_lengths)
        else:
            return list(trial_lengths)[0]

    def add_population(self,
                       features,
                       labels):
        self.populations.append(Population(features, labels, self.bin_length, self.bin_method, self.condition))

    def __str__(self):
        return f"Pseudopopulation containing a total of {self.total_neurons} over " \
               f"{self.total_trials} time-locked trials across {self.total_populations} datasets"
