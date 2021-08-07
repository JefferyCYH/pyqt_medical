from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from LGEprocess.graphicsView_LGE import GraphicsView_LGE


def load_nii(path):
    image = nib.load(path)
    affine = image.affine
    image = np.asarray(image.dataobj)
    return image, affine

class LGEView(QWidget):
    switch_window = pyqtSignal()

    def __init__(self):
        super().__init__()
        # self.setWindowTitle('LGE image')
        self.depth = SubLayout('depth')
        # self._empty = True
        # self.graphicsView_lge = GraphicsView_LGE(self)

        layout = QVBoxLayout()
        layout.addLayout(self.depth.layout)
        self.setLayout(layout)

<<<<<<< Updated upstream
=======
    # def contextMenuEvent(self, event):
    #     print('save')
    #     # if not self.has_photo():
    #     #     print('no')
    #     #     return
    #     # print('save')
    #     menu = QMenu()
    #     save_action = QAction('另存为', self)
    #     save_action.triggered.connect(self.save_current)  # 传递额外值
    #     menu.addAction(save_action)
    #     menu.exec(QCursor.pos())
    #
    # def save_current(self,img):
    #     print('另存为')
    #     file_name = QFileDialog.getSaveFileName(self, '另存为', './', 'Image files(*.nii.gz)')[0]
    #     print(file_name)
    #     if file_name:
    #         sitk.WriteImage(img,file_name)

    # def has_photo(self):
    #     return not self._empty

    def update_image(self, img):
        self._empty = False
        print(img.shape)
        # self.img=img
        depth_max = img.shape[2]
        if len(img.shape) != 3:
            self.depth.label.setText('file mismatch!')
            return
        self.depth.LGE_arr = img
        self.depth.setdepth_max(self.depth.slideblock, depth_max)

>>>>>>> Stashed changes

    def change_image(self, image):
        LGE_arr, affine = load_nii(image)
        depth_max = LGE_arr.shape[2]
        if len(LGE_arr.shape) != 3:
            self.depth.label.setText('file mismatch!')
            return
        self.depth.LGE_arr = LGE_arr
        self.depth.setdepth_max(self.depth.slideblock, depth_max)


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
        else:
            message_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            message_box.exec_()
            

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

