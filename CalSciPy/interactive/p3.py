class MainWindow2(QMainWindow):
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
        if 0 <= value < (self.psf.planes - 1):
            self.z_plane.axes.cla()
            self.z_plane.axes.imshow(self.psf.stack[value, :, :], cmap="coolwarm")
            self.z_plane.draw()


def interactive_psf(psf):
    fig = plt.figure()
    gs = GridSpec(2, 2)
    # current plane
    ax1 = plt.subplot(gs[0, 0])
    # center plane
    ax2 = plt.subplot(gs[0, 1])
    # XZ
    ax3 = plt.subplot(gs[1, 0])
    # YZ
    ax4 = plt.subplot(gs[1, 1])

    current_plane = psf.z_max

    # current plane
    ax1.imshow(psf.stack[current_plane, :, :], cmap="coolwarm")
    ax1.set_title("Current Plane")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # center plane
    ax2.imshow(psf.stack[psf.z_max, :, :], cmap="coolwarm")
    ax2.set_title("Center Plane")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(psf.stack[:, :, psf.x_max], cmap="coolwarm")
    ax3.set_title("XZ")
    ax3.set_xticks([])

    ax4.imshow(psf.stack[:, psf.y_max, :], cmap="coolwarm")
    ax4.set_title("YZ")
    ax4.set_xticks([])

    def update():
        cp = plane_slider.value()
        ax1.imshow(psf.stack[cp, :, :], cmap="coolwarm")

    plane_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
    plane_slider.setRange(0, psf.planes)
    plane_slider.setSingleStep(1)
    plane_slider.setValue(psf.z_max)
    plane_slider.valueChanged.connect(update)
    # plane_slider.
    vbox = QtWidgets.QVBoxLayout()
    vbox.addWidget(plane_slider)
    fig.canvas.setLayout(vbox)

    return fig
