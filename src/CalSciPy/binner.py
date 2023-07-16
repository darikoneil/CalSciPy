from __future__ import annotations
from typing import Union, Iterable, Callable
import numpy as np
import pandas as pd


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
