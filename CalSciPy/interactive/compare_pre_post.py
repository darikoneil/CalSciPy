from CalSciPy.organization import Mouse
from CalSciPy.io_tools import load_binary
import numpy as np
from CalSciPy.interactive.compare_image_overlay import RegistrationComparison


mouse = Mouse.load()

pre_ops = np.load(mouse.prescreen_0.get("results")("ops"), allow_pickle=True).item()
post_ops = np.load(mouse.stimulation_0.get("results")("ops"), allow_pickle=True).item()

pre_corr = pre_ops.get("Vcorr")
post_corr = post_ops.get("Vcorr")

c = RegistrationComparison(pre_corr, post_corr)
