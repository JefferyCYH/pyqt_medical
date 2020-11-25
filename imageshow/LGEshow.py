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
        print('chang the image in LGEshow')
        LGE_arr, affine = load_nii(image)
        depth_max = LGE_arr.shape[2]
        print('LGE_arr.shape[2]', LGE_arr.shape[2])

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
        print('改变value')
        print(self.slideblock.value())
        self.slideblock.setValue(value)
        self.plotCanvas.plot(self.LGE_arr, value, cmap='gray')
        self.label.setText('Slice:' + str(value))


    def setdepth_max(self, slideblock, maximum):
        print('maximum:', maximum)
        slideblock.setMaximum(maximum)


class PlotCanvas(FigureCanvas):
    print('创建plotcanvas')

    def __init__(self, parent=None, width=2, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('black')
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, nii_arr, value, cmap='gray'):
        print('plot')
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        slice = nii_arr[:, :, value - 1]
        slice = np.transpose(slice)
        ax.imshow(slice, cmap=cmap)
        print('xxxxxxxx')
        self.draw()
        print('结束')


        # for i in range(value):
        #     print('画图')
        #     slice = nii_arr[:, :, value-1]
        #     slice = np.transpose(slice)
        #     # slice = slice[::-1]
        #     # slice = nib.viewers.OrthoSlicer3D(slice).show()
        #     ax.imshow(slice, cmap=cmap)
        #     self.draw()
        #     print('结束')

# if __name__ == '__main__':
#     LGE_file = './Case_P099.nii.gz'
#     LGE_file_GT = './Case_P099_GT.nii.gz'
#     app = QApplication(sys.argv)
#     app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
#     example = LGEViewer(LGE_file)
#     example.show()
#     sys.exit(app.exec_())