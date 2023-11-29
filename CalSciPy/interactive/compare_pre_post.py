from CalSciPy.organization import Mouse
from CalSciPy.io_tools import load_binary
import numpy as np
from compare_image_overlay import RegistrationComparison


mouse = Mouse.load()

pre = load_binary(mouse.prescreen_0.get("results")(), mapped=True, mode="r")

pre_mean = np.mean(pre, axis=0)

post = load_binary(mouse.stimulation_0.get("results")(), mapped=True, mode="r")

post_mean = np.mean(post, axis=0)

pre_ops = np.load(mouse.prescreen_0.get("results")("ops"), allow_pickle=True).item()
post_ops = np.load(mouse.stimulation_0.get("results")("ops"), allow_pickle=True).item()

pre_corr = pre_ops.get("Vcorr")
post_corr = post_ops.get("Vcorr")

c = RegistrationComparison(pre_corr, post_corr)
