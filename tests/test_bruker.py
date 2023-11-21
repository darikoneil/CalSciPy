
from pathlib import Path

from CalSciPy.bruker.loaders import load_bruker_tifs


folder = Path.cwd().joinpath(
    "tests\\testing_samples\\datasets\\1channels_4planes_100frames_256height_256width\\bruker_folder")

images = load_bruker_tifs(folder, channel=None, plane=None, verbose=True)
