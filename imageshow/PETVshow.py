import sys, os
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import qdarkstyle
import scipy.misc
import pdb


class PETViewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viewer-demo")
        self.sagittal = SubLayout("sagittal")

        layout = QVBoxLayout()
        layout.addLayout(self.sagittal.layout)
        self.setLayout(layout)

    def change_image(self, image):
        pet_image = sitk.ReadImage(image)

        pet_arr = sitk.GetArrayFromImage(pet_image)
        if len(pet_arr.shape) != 4:
            self.sagittal.label.setText('file mismatch!')
            return

        sagittal_slice_max = pet_arr.shape[0]
        coronal_slice_max = pet_arr.shape[1]

        self.sagittal.pet_arr = pet_arr
        self.sagittal.mri_arr = pet_arr
        self.sagittal.wm_arr = pet_arr
        self.sagittal.rm_arr = pet_arr

        self.sagittal.scrollbar_setmaximum(self.sagittal.scrollbar_time, sagittal_slice_max)
        self.sagittal.scrollbar_setmaximum(self.sagittal.scrollbar_slice, coronal_slice_max)


class SubLayout(PyQt5.QtWidgets.QVBoxLayout):
    def __init__(self, tag):
        super().__init__()

        self.pet_arr = None
        self.mri_arr = None
        self.wm_arr = None
        self.rm_arr = None
        self.axis1_value = 0
        self.axis2_value = 0
        layout1 = QHBoxLayout()
        self.plotCanvas1 = PlotCanvas(self, width=2, height=2)
        # self.plotCanvas2 = PlotCanvas(self, width=2, height=2)
        # self.plotCanvas3 = PlotCanvas(self, width=2, height=2)
        # self.plotCanvas4 = PlotCanvas(self, width=2, height=2)
        layout1.addWidget(self.plotCanvas1)
        # layout1.addWidget(self.plotCanvas2)
        # layout1.addWidget(self.plotCanvas3)
        # layout1.addWidget(self.plotCanvas4)

        layout2 = QHBoxLayout()
        self.scrollbar_time = QScrollBar(Qt.Horizontal)
        self.scrollbar_time.sliderMoved.connect(lambda: self.sliderMoved("scrollbar_{}".format(tag)))
        layout2.addWidget(self.scrollbar_time)

        layout3 = QHBoxLayout()
        self.scrollbar_slice = QScrollBar(Qt.Horizontal)
        self.scrollbar_slice.sliderMoved.connect(lambda: self.sliderMoved2("scrollbar_{}".format(tag)))
        layout3.addWidget(self.scrollbar_slice)

        layout4 = QHBoxLayout()
        self.label = QLabel()
        layout4.addWidget(self.label)

        self.layout = QVBoxLayout()
        self.layout.addLayout(layout1)
        self.layout.addLayout(layout2)
        self.layout.addLayout(layout3)
        self.layout.addLayout(layout4)

    def sliderMoved(self, scrollbarName):
        self.axis1_value = self.sender().value()
        if not self.pet_arr is None:
            self.plotCanvas1.plot(scrollbarName, self.pet_arr, self.axis1_value, self.axis2_value, cmap='gray')
            # self.plotCanvas2.plot(scrollbarName, self.mri_arr, self.axis1_value, self.axis2_value, cmap='gray')
            # self.plotCanvas3.plot(scrollbarName, self.wm_arr, self.axis1_value, self.axis2_value, cmap='gray')
            # self.plotCanvas4.plot(scrollbarName, self.rm_arr, self.axis1_value, self.axis2_value, cmap='jet')
            self.label.setText('Frame:' + str(self.axis1_value) + 'Slice:' + str(self.axis2_value))
        else:
            message_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            message_box.exec_()

    def sliderMoved2(self, scrollbarName):
        self.axis2_value = self.sender().value()
        if not self.pet_arr is None:
            self.plotCanvas1.plot(scrollbarName, self.pet_arr, self.axis1_value, self.axis2_value, cmap='gray')
            # self.plotCanvas2.plot(scrollbarName, self.mri_arr, self.axis1_value, self.axis2_value, cmap='gray')
            # self.plotCanvas3.plot(scrollbarName, self.wm_arr, self.axis1_value, self.axis2_value, cmap='gray')
            # self.plotCanvas4.plot(scrollbarName, self.rm_arr, self.axis1_value, self.axis2_value, cmap='jet')
            self.label.setText('Frame:' + str(self.axis1_value) + 'Slice:' + str(self.axis2_value))
        else:
            message_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            message_box.exec_()

    def scrollbar_setmaximum(self, scrollbar, maximum):
        scrollbar.setMaximum(maximum)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('black')
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, scrollbarName, nii_arr, value1, value, cmap='jet'):

        self.fig.clear()
        ax = self.figure.add_subplot(111)

        # 依据scrollbar.value值，调整某一维の数值

        if scrollbarName == 'scrollbar_sagittal':
            slice = nii_arr[value1 - 1, value - 1, :, :]

        elif scrollbarName == 'scrollbar_coronal':
            slice = nii_arr[:, value - 1, :]
            slice = slice[::-1]
        elif scrollbarName == 'scrollbar_transverse':
            slice = nii_arr[:, :, value - 1]
            slice = slice[::-1]
        else:
            print("{} not support.".format(scrollbarName))
            pass

        ax.imshow(slice, cmap=cmap)
        self.draw()