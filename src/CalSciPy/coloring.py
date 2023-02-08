import os
from typing import Tuple, Optional, List, Union
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from .io_tools import save_video


class ImageColoring:
    def __init__(self, images: np.ndarray, rois: np.ndarray):
        self.images = images
        self.rois = rois
        self.background = None
        