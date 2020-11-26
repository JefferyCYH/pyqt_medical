from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import nibabel as nib
import numpy as np


def load_nii(path):
    image = nib.load(path)
    affine = image.affine
    image = np.asarray(image.dataobj)
    return image, affine

class LGEView(QWidget):
    switch_window = pyqtSignal()

    def __init__(self):
        print('开始')
        super().__init__()
        # self.setWindowTitle('LGE image')
        self.depth = SubLayout('depth')

        layout = QVBoxLayout()
        layout.addLayout(self.depth.layout)
        self.setLayout(layout)


    def change_image(self, image):
        LGE_arr, affine = load_nii(image)
        depth_max = LGE_arr.shape[2]

        self.depth.LGE_arr = LGE_arr
        self.depth.setdepth_max(self.depth.slideblock, depth_max)
        return


class SubLayout(QVBoxLayout):
    def __init__(self, tag):
        super().__init__()
        self.LGE_arr = None
        self.value = 0

        layout1 = QHBoxLayout()
        self.plotCanvas = PlotCanvas(self, width=10, height=10)
        layout1.addWidget(self.plotCanvas)

        layout2 = QHBoxLayout()
        self.slideblock = QSlider(Qt.Horizontal)
        self.slideblock.setMinimum(0)
        # self.slideblock.setMaximum(10)  #depth_max
        self.slideblock.setSingleStep(1)
        self.slideblock.valueChanged['int'].connect(self.changevalue)
        layout2.addWidget(self.slideblock)

        layout3 = QHBoxLayout()
        self.label = QLabel()
        layout3.addWidget(self.label)

        self.layout = QVBoxLayout()
        self.layout.addLayout(layout1)
        self.layout.addLayout(layout2)
        self.layout.addLayout(layout3)


    def changevalue(self, value):
        self.slideblock.setValue(value)
        if self.LGE_arr is not None:
            self.plotCanvas.plot(self.LGE_arr, value, cmap='gray')
            self.label.setText('Slice:' + str(value))

    def setdepth_max(self, slideblock, maximum):
        slideblock.setMaximum(maximum)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=2, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('black')
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, nii_arr, value, cmap='gray'):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        slice = nii_arr[:, :, value - 1]
        slice = np.transpose(slice)
        ax.imshow(slice, cmap=cmap)
        self.draw()