import numpy as np
from pathlib import Path
from collections import ChainMap
from CalSciPy.optogenetics import Photostimulation
from CalSciPy.bruker.protocols import generate_galvo_point_list, generate_marked_points_protocol

from scratch_dev.visualize_optogenetics import *

# parameters
protocol_folder = Path.cwd()

data_folder = protocol_folder.joinpath("tests").joinpath("testing_samples").joinpath("suite2p")

name = "darik_test_protocol"

point_parameters = {"uncaging_laser_power": 1000, "spiral_revolutions": 50.0, "duration": 100.0, "z": 19.96}

# create experiment
photostimulation = Photostimulation.import_rois(folder=data_folder)

# make ensembles
photostimulation.add_photostimulation_group([0, ], name="Point 1")
photostimulation.add_photostimulation_group([10, 5, 6, 11, 12], name="Ensemble 1")
photostimulation.add_photostimulation_group([29], name="Point 2")
photostimulation.add_photostimulation_group([2, 3, 7, 8, 9, 16, 17], name="Ensemble 2")


view_rois(photostimulation)
view_roi_overlay(photostimulation)
view_targets(photostimulation, photostimulation.stimulated_neurons)
view_target_overlay(photostimulation, photostimulation.stimulated_neurons)


mpl, gpl = generate_marked_points_protocol(photostimulation,
                                           targets_only=True,
                                           parameters=point_parameters,
                                           name=name,
                                           file_path=protocol_folder)
