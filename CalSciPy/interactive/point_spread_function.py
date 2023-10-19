from __future__ import annotations
from typing import TypedDict

import sys

import numpy as np

from CalSciPy.optics.psf import PSF

from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QSlider, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import seaborn as sns
matplotlib.use("QtAgg")


class Color(QWidget):
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)


class EmbeddedPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class EmbeddedSharedPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        gs = GridSpec(5, 4)
        self.axes = [fig.add_subplot(gs[:4, :]), ]
        self.axes.append(fig.add_subplot(gs[4, :], sharex=self.axes[0]))
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self, psf: PSF = None):
        super().__init__()

        # validate or initialize empty psf
        self.psf = psf

        # pre-allocate widget slots
        self.load_button = None

        # pre-allocate embedded plots
        self.fwhm_xy = {}
        self.fwhm_yx = {}
        self.fwhm_xz = {}
        self.fwhm_yz = {}
        self.xy = {}
        self.xyz = {}
        self.xz = {}
        self.yz = {}

        # initialize included widgets
        self.initialize_load_button()
        self.initialize_xy()
        self.initialize_xz()
        self.initialize_xyz()
        self.initialize_yz()
        self.initialize_slider()

        # organize and construct main window
        self.setWindowTitle("CalSciPy - Interactive Point Spread Function")
        self.set_layouts()

        # launch
        self.show()

    def initialize_load_button(self):
        button = QPushButton("Load Stack")
        button.setCheckable(True)
        button.clicked.connect(self.load_button_pressed)
        self.load_button = button

    def initialize_slider(self):
        slider = QSlider(Qt.Orientation.Vertical)
        slider.setRange(0, self.psf.planes - 1)
        slider.setSingleStep(1)
        slider.setValue(0)
        slider.sliderMoved.connect(self.xyz_update)
        slider.valueChanged.connect(self.xyz_update)
        self.xyz["slider"] = slider

    def initialize_fwhm_xy(self):
        widget = EmbeddedPlot(self)
        widget.axes.set_title(f"FWHM XY: {self.psf.fwhm[0]}")
        self.fwhm_xy["image"] = widget.axes.plot(self.psf.x_scale, self.psf.calculate_intensities(self.psf.x_fit,
                                                                                                  self.psf.x_scale))
        self.fwhm_xy["widget"] = widget

    def initialize_xy(self):
        intensities = self.psf.calculate_intensities(self.psf.x_fit, self.psf.x_scale)

        intensities /= np.max(intensities)

        widget = EmbeddedSharedPlot(self)
        widget.axes[0].set_title("XY Plane (Z-Max)")
        widget.axes[0].set_xticks([])
        widget.axes[0].set_yticks([])
        widget.axes[1].plot(self.psf.x_scale, intensities)
        widget.axes[1].set_box_aspect(0.25)
        self.xy["image"] = widget.axes[0].imshow(self.psf.stack[self.psf.z_max, :, :],
                                                 cmap="coolwarm",
                                                 extent=[np.min(self.psf.x_scale), np.max(self.psf.x_scale),
                                                         np.min(self.psf.y_scale), np.max(self.psf.y_scale)])
        self.xy["widget"] = widget

    def initialize_xz(self):
        widget = EmbeddedPlot(self)
        widget.axes.set_title("XZ Plane (Y-Max)")
        widget.axes.set_xticks([])
        widget.axes.set_yticks([])
        self.xz["image"] = widget.axes.imshow(self.psf.stack[:, self.psf.y_max, :].T, cmap="coolwarm")
        self.xz["widget"] = widget

    def initialize_xyz(self):
        widget = EmbeddedPlot(self)
        widget.axes.set_title("XY Plane")
        widget.axes.set_xticks([])
        widget.axes.set_yticks([])
        self.xyz["image"] = widget.axes.imshow(self.psf.stack[self.psf.z_max, :, :], cmap="coolwarm")
        self.xyz["widget"] = widget

    def initialize_yz(self):
        widget = EmbeddedPlot(self)
        widget.axes.set_title("YZ Plane (X-Max)")
        widget.axes.set_xticks([])
        widget.axes.set_yticks([])
        self.yz["image"] = widget.axes.imshow(self.psf.stack[:, :, self.psf.x_max].T, cmap="coolwarm")
        self.yz["widget"] = widget

    def load_button_pressed(self):
        print("Load button pressed...")

    def set_layouts(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        lower_left_layout = QHBoxLayout()
        right_layout = QVBoxLayout()
        upper_right_layout = QHBoxLayout()

        lower_left_layout.addWidget(self.xz.get("widget"))
        lower_left_layout.addWidget(self.yz.get("widget"))

        upper_left_layout = QVBoxLayout()
        upper_left_layout.addWidget(self.xy.get("widget"))
        upper_left_layout.addWidget(self.fwhm_xy.get("widget"))

        left_layout.addLayout(upper_left_layout)
        left_layout.addLayout(lower_left_layout)

        lower_right_layout = QVBoxLayout()
        lower_right_layout.addWidget(self.load_button)

        upper_right_layout.addWidget(self.xyz.get("widget"))
        upper_right_layout.addWidget(self.xyz.get("slider"))

        right_layout.addLayout(upper_right_layout)
        right_layout.addLayout(lower_right_layout)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def xyz_update(self, value):
        widget = self.xyz.get("widget")
        widget.axes.cla()
        widget.axes.imshow(self.psf.stack[value, :, :], cmap="coolwarm")
        widget.axes.set_xticks([])
        widget.axes.set_yticks([])
        widget.axes.set_title("XY Plane")
        widget.draw()


if __name__ == "__main__":
    psf_ = PSF(np.load("C:\\Users\\YUSTE\\Desktop\\PSF.npy"), scaling=(0.5, 0.1, 0.1))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(psf_)
    app.exec()
