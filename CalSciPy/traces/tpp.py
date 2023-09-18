from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.ndimage.filters import gaussian_filter1d
from numba import njit
from tqdm import tqdm

from .._calculations import sliding_window



