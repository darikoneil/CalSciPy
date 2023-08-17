from pathlib import Path
import numpy as np
from collections import ChainMap


from CalSciPy.opto import Photostimulation, Group
from CalSciPy.bruker.factories import BrukerXMLFactory
from CalSciPy.bruker.xml_objects import MarkPointSeriesElements, MarkPointElement, GalvoPointElement


folder = Path("D:\\DEM2\\preexposure\\results")

ps = Photostimulation.import_suite2p(folder)

params = {"uncaging_laser_power": 69, "spiral_revolutions": 50, "duration": 100, "z": 0.0}

gpl = ps.generate_galvo_point_list(params)

galvo_point = gpl.galvo_points[0]

factory = BrukerXMLFactory()

gpe = GalvoPointElement()

mpe = MarkPointElement(points=(gpe, ))

mpse = MarkPointSeriesElements(marks=(mpe, ))

lines = factory.constructor(mpse)


def gen(ps, gpl, factory):
    order = np.arange(0, 750, 50).tolist()
    num_neuro\ns = len(order)
    params = {"initial_delay": 1,
              "spiral_revolutions": 50,
              "duration": 100,
              "inter_point_delay": 0.12
              }

    gpe = []
    for neuron in order:
        p = {"points": "".join(["Point ", str(neuron)]), "indices": neuron}
        gpe.append(GalvoPointElement(**ChainMap(p, params)))

    mpes = []
    for g in gpe:
        m = MarkPointElement(points=(g, ), uncaging_laser_power=1000)
        mpes.append(m)

    series = MarkPointSeriesElements(marks=tuple(mpes))

    lines = factory.constructor(series)

    file = open("C:\\Users\\Desktop\\TestProtocol.xml", "w+")
    for line in lines:
        file.write(line)
    file.close()
