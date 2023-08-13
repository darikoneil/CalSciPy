import numpy as np
from pathlib import Path
from collections import ChainMap
from CalSciPy.optogenetics import Photostimulation
from CalSciPy.bruker.protocols import generate_galvo_point_list

# from scratch.visualize_optogenetics import *

# folder = Path("D:\\DEM2\\preexposure\\results")

# parameters
protocol_folder = Path("C:\\Users\\YUSTE\\Desktop")
data_folder = Path("D:\\EM_E\\joint_analysis2\\results")
name = "darik_test_protocol"
point_parameters = {"uncaging_laser_power": 1000, "spiral_revolutions": 50.0, "duration": 100.0, "z": 19.96}

# create experiment
photostimulation = Photostimulation.import_rois(folder=data_folder)

# make ensembles
photostimulation.add_photostimulation_group([0, 1, 5, 10, 15], name="Ensemble 1")
photostimulation.add_photostimulation_group([10, 5, 6, 11, 12], name="Ensemble 2")
photostimulation.add_photostimulation_group([2, 3, 7, 8, 9, 16, 17], name="Ensemble 3")
photostimulation.add_photostimulation_group([0, ], name="")

# make galvo point list
gpl = generate_galvo_point_list(photostimulation,
                                targets_only=True,
                                parameters=point_parameters,
                                name=name,
                                file_path=protocol_folder)


from CalSciPy.bruker.xml_objects import GalvoPoint, GalvoPointList, _BrukerObject, GalvoPointGroup, MarkPointSeriesElements, MarkPointElement, GalvoPointElement, Point
from CalSciPy.bruker.protocols import _revise_indices_targets_only, _check_parameters,_scale_gpl_parameters


mpl = MarkPointSeriesElements(marks=None,
                              iterations=photostimulation.sequence.repetitions,
                              iteration_delay=photostimulation.sequence.delay)

mpe = MarkPointElement(points=(0, 1))

gpe = GalvoPointElement()

####


targets_only = True
if targets_only:
    idx, permitted_points, rois, groups = _revise_indices_targets_only(photostimulation)
else:
    idx = np.arange(photostimulation.neurons).tolist()
    permitted_points = np.arange(photostimulation.neurons).tolist()
    rois = photostimulation.rois
    groups = photostimulation.sequence

###

g = groups[0]
x = _check_parameters(MarkPointElement, point_parameters)
_scale_gpl_parameters(x)

gp = {"points": tuple(g.ordered_index)}

gp = ChainMap(x, gp)
mpe = MarkPointElement(**gp)

