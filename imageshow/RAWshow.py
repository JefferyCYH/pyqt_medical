from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np



class RAWView(QWidget):
    switch_window = pyqtSignal()

    def __init__(self):
        super().__init__()
        # self.setWindowTitle('LGE image')
        self.sublayout = SubLayout('depth')

        layout = QVBoxLayout()
        layout.addLayout(self.sublayout.layout)
        self.setLayout(layout)


    def change_image(self, image):
        RAW_arr = image
        self.sublayout.RAW_arr = RAW_arr
        self.sublayout.windowcenter = (np.max(RAW_arr) + np.min(RAW_arr))/2
        self.sublayout.windowwidth = (np.max(RAW_arr) - np.min(RAW_arr))/2
        self.sublayout.changevalue(0)


class SubLayout(QVBoxLayout):
    def __init__(self, tag):
        super().__init__()
        self.RAW_arr = None
        self.slice = 0
        self.windowwidth = 0
        self.windowcenter = 0

        layout1 = QHBoxLayout()
        self.plotCanvas = PlotCanvas(self, width=20, height=20)
        layout1.addWidget(self.plotCanvas)

        layout2 = QHBoxLayout()
        self.slideslice = QSlider(Qt.Horizontal)
        self.slideslice.setMinimum(0)
        self.slideslice.setMaximum(199)
        self.slideslice.setSingleStep(1)
        self.slideslice.valueChanged['int'].connect(self.changevalue)
        # self.slideslice.sliderMoved.connect(self.changevalue)
        layout2.addWidget(self.slideslice)

        layout3 = QHBoxLayout()
        self.slidewidth = QSlider(Qt.Horizontal)
        self.slidewidth.setMinimum(1)
        self.slidewidth.setMaximum(1200)
        self.slidewidth.setSingleStep(1)
        self.slidewidth.valueChanged['int'].connect(self.changewidth)
        layout3.addWidget(self.slidewidth)

        layout4 = QHBoxLayout()
        self.slidecenter = QSlider(Qt.Horizontal)
        self.slidecenter.setMinimum(1)
        self.slidecenter.setMaximum(2400)
        self.slidecenter.setSingleStep(1)
        self.slidecenter.valueChanged['int'].connect(self.changecenter)
        layout4.addWidget(self.slidecenter)

        layout5 = QHBoxLayout()
        self.label = QLabel()
        layout5.addWidget(self.label)

        self.layout = QVBoxLayout()
        self.layout.addLayout(layout1)
        self.layout.addLayout(layout2)
        self.layout.addLayout(layout3)
        self.layout.addLayout(layout4)
        self.layout.addLayout(layout5)


    def changevalue(self, value):
        self.slideslice.setValue(value)
        self.slice = value
        if self.RAW_arr is not None:
            self.plotCanvas.plot(self.RAW_arr, self.slice, self.windowcenter, self.windowwidth, cmap='gray')
            self.label.setText('Slice:' + str(self.slice) + '窗宽:' + str(self.windowwidth) + '窗位:' + str(self.windowcenter))
        else:
            message_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            message_box.exec_()

    def changewidth(self, value):
        self.slidewidth.setValue(value)
        self.windowwidth = float(value)
        if self.RAW_arr is not None:
            self.plotCanvas.plot(self.RAW_arr, self.slice, self.windowcenter, self.windowwidth, cmap='gray')
            self.label.setText('Slice:' + str(self.slice) + '窗宽:' + str(self.windowwidth) + '窗位:' + str(self.windowcenter))
        else:
            message_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            message_box.exec_()

    def changecenter(self, value):
        self.slidecenter.setValue(value)
        self.windowcenter = float(value)
        if self.RAW_arr is not None:
            self.plotCanvas.plot(self.RAW_arr, self.slice, self.windowcenter, self.windowwidth, cmap='gray')
            self.label.setText('Slice:' + str(self.slice) + '窗宽:' + str(self.windowwidth) + '窗位:' + str(self.windowcenter))
        else:
            message_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            message_box.exec_()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=2, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('black')
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, arr, value, windowcenter, windowwidth, cmap='gray'):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        slice = arr[value, :, :]
        slice = np.transpose(slice)
        slicediv = np.max(slice)-np.min(slice)
        slice = np.where(slice > windowcenter+windowwidth, windowcenter+windowwidth, slice)
        slice = np.where(slice < windowcenter-windowwidth, windowcenter-windowwidth, slice)
        slice = (slice-np.min(slice))*slicediv/windowwidth*2
        slice.astype(np.float64)
        ax.imshow(slice, cmap=cmap)
        self.draw()
