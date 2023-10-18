from __future__ import annotations

import sys

import numpy as np

from CalSciPy.optics.psf import PSF

from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QSlider, QGridLayout
from PyQt6.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import seaborn as sns
matplotlib.use("QtAgg")


class XYPlane(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self, psf: PSF = None):
        super().__init__()

        # pre-allocate
        self.central = None
        self.layout = None
        self.plane = 0
        self.psf = psf
        self.z_plane = None
        self.xy_plane = None
        self.xz_plane = None
        self.yz_plane = None
        self.slider = None

        # make figures
        self.init_z_plane()
        self.init_xy_plane()
        self.init_xz_plane()
        self.init_yz_plane()
        self.init_slider()
        self.global_design()
        self.show()

    def global_design(self):
        self.layout = QGridLayout()
        self.layout.addWidget(self.xy_plane, 0, 0)
        self.layout.addWidget(self.slider, 2, 0)
        self.layout.addWidget(self.z_plane, 0, 1)
        self.layout.addWidget(self.xz_plane, 1, 0)
        self.layout.addWidget(self.yz_plane, 1, 1)
        self.central = QWidget()
        self.central.setLayout(self.layout)
        self.setCentralWidget(self.central)

    def init_xy_plane(self):
        self.xy_plane = XYPlane(self)
        self.xy_plane.axes.imshow(self.psf.denoised[self.psf.z_max, :, :], cmap="coolwarm")

    def init_xz_plane(self):
        self.xz_plane = XYPlane(self)
        self.xz_plane.axes.imshow(self.psf.stack[:, self.psf.y_max, :], cmap="coolwarm")

    def init_yz_plane(self):
        self.yz_plane = XYPlane(self)
        self.yz_plane.axes.imshow(self.psf.stack[:, :, self.psf.x_max], cmap="coolwarm")

    def init_z_plane(self):
        self.z_plane = XYPlane(self)
        self.z_plane.axes.imshow(self.psf.stack[self.plane, :, :], cmap="coolwarm")

    def init_slider(self):
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.psf.planes - 1)
        self.slider.setSingleStep(1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_z_plane)
        self.slider.sliderMoved.connect(self.update_z_plane)
        #self.slider.sliderPressed.connect(self.update_z_plane)
        #self.slider.sliderReleased.connect(self.update_z_plane)

    def indicate(self):
        print(f"{self.slider.value()=}")

    def update_z_plane(self, value):
        #val = self.slider.value()
        print(f"{value=}")
        if 0 <= value < (self.psf.planes - 1):
            #print(f"ZPLANE")
            self.z_plane.axes.cla()
            self.z_plane.axes.imshow(self.psf.stack[value, :, :], cmap="coolwarm")
            self.z_plane.draw()


if __name__ == "__main__":
    psf = PSF(np.load("C:\\Users\\YUSTE\\Desktop\\PSF.npy"), scaling=(0.5, 0.1, 0.1))
    app = QApplication(sys.argv)
    w = MainWindow(psf)
    app.exec()
