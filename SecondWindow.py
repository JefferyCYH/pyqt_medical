import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from custom.stackedWidget import StackedWidget
from custom.treeView import FileSystemTreeView
from custom.listWidgets import FuncListWidget, UsedListWidget
from custom.graphicsView import GraphicsView

from LGEprocess.stackedWidget_LGE import StackedWidget_LGE
from LGEprocess.listWidgets_LGE import FuncListWidget_LGE, UsedListWidget_LGE
from LGEprocess.graphicsView_LGE import GraphicsView_LGE

from FDCMRprocess.stackedWidget_4D import StackedWidget_4D
from FDCMRprocess.listWidgets_4D import FuncListWidget_4D, UsedListWidget_4D
from FDCMRprocess.graphicsView_4D import GraphicsView_4D

from RAWprocess.stackedWidget_RAW import StackedWidget_RAW
from RAWprocess.listWidgets_RAW import FuncListWidget_RAW, UsedListWidget_RAW
from RAWprocess.graphicsView_RAW import GraphicsView_RAW
from RAWprocess.treeView_RAW import FileSystemTreeView_RAW


from imageshow.LGEshow import LGEView
from imageshow.PETVshow import PETViewer
from imageshow.RAWshow import RAWView


class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.datatype = "png"
        self.tool_bar = self.addToolBar('工具栏')
        self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "向右旋转90", self)
        self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "向左旋转90°", self)
        self.action_histogram = QAction(QIcon("icons/直方图.png"), "直方图", self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.action_histogram.triggered.connect(self.histogram)
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate, self.action_histogram))

        self.useListWidget = UsedListWidget(self)
        self.funcListWidget = FuncListWidget(self)
        self.stackedWidget = StackedWidget(self)
        self.fileSystemTreeView = FileSystemTreeView(self)
        self.graphicsView = GraphicsView(self)

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.useListWidget)
        self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        self.setCentralWidget(self.graphicsView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        self.setWindowTitle('医学图像处理')
        self.setWindowIcon(QIcon('icons/main.png'))
        self.src_img = None
        self.cur_img = None

    def update_image(self):
        print('update')
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.update_image(img)

    def change_image(self, img):
        print('change')
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.change_image(img)

    def process_image(self):
        print('process')
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    def right_rotate(self):
        self.graphicsView.rotate(90)

    def left_rotate(self):
        self.graphicsView.rotate(-90)


    def histogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.cur_img], [i], None, [256], [0, 256])
            histr = histr.flatten()
            plt.plot(range(256), histr, color=col)
            plt.xlim([0, 256])
        plt.show()



class MyApp_LGE(QMainWindow):
    def __init__(self):
        super(MyApp_LGE, self).__init__()
        self.datatype = "lge"
        self.tool_bar = self.addToolBar('工具栏')
        self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "向右旋转90", self)
        self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "向左旋转90°", self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate))

        self.useListWidget_LGE = UsedListWidget_LGE(self)
        self.funcListWidget_LGE = FuncListWidget_LGE(self)
        self.stackedWidget_LGE = StackedWidget_LGE(self)
        self.fileSystemTreeView = FileSystemTreeView(self)
        self.graphicsView_lge = GraphicsView_LGE(self)
        self.LGEView = LGEView()

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget_LGE)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.useListWidget_LGE)
        self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget_LGE)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        self.setCentralWidget(self.LGEView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        self.setWindowTitle('医学图像处理')
        self.setWindowIcon(QIcon('icons/main.png'))
        self.src_img = None
        self.cur_img = None


    def update_image(self):
        print('update')
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        # self.graphicsView.update_image(img)
        self.LGEView.change_image(img)

    def change_image(self, img):
        print('src_img is True')
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        # self.graphicsView.change_image(img)
        self.LGEView.change_image(img)

    def change_label(self, label):
        self.src_img = label
        img = self.process_image()
        self.cur_img = img
        # self.graphicsView.change_image(img)
        self.LGEView.change_label(label)

    def process_image(self):
        print('process')
        img = self.src_img.copy()
        for i in range(self.useListWidget_LGE.count()):
            img = self.useListWidget_LGE.item(i)(img)
        return img

    def right_rotate(self):
        # self.graphicsView.rotate(90)
        self.LGEView.rotate(90)

    def left_rotate(self):
        # self.graphicsView.rotate(-90)
        self.LGEView.rotate(-90)

class MyApp_PETV(QMainWindow):
    def __init__(self):
        super(MyApp_PETV, self).__init__()
        self.datatype = "petv"
        self.tool_bar = self.addToolBar('工具栏')
        self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "向右旋转90", self)
        self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "向左旋转90°", self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate))

        self.UsedListWidget_4D = UsedListWidget_4D(self)
        self.FuncListWidget_4D = FuncListWidget_4D(self)
        self.StackedWidget_4D= StackedWidget_4D(self)
        self.fileSystemTreeView = FileSystemTreeView(self)
        self.PETViewer = PETViewer()

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.FuncListWidget_4D)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.UsedListWidget_4D)
        self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.StackedWidget_4D)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        self.setCentralWidget(self.PETViewer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        self.setWindowTitle('医学图像处理')
        self.setWindowIcon(QIcon('icons/main.png'))
        self.src_img = None
        self.cur_img = None
        self.label=None
    
    def change_label(self,label):
        self.label=label[0:-7]+'_gt.nii.gz'

        print(self.label)

    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        print(img.shape)
        self.PETViewer.change_image(img)

    def change_image(self, img):
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        # self.FuncListWidget_4D.change_image(img)
        self.PETViewer.change_image(img)

    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.UsedListWidget_4D.count()):
            img = self.UsedListWidget_4D.item(i)(img)
        return img

    def right_rotate(self):
        self.PETViewer.rotate(90)

    def left_rotate(self):
        # self.graphicsView.rotate(-90)
        self.PETViewer.rotate(-90)

class MyApp_RAW(QMainWindow):
    def __init__(self):
        super(MyApp_RAW, self).__init__()
        self.datatype = "raw"
        self.tool_bar = self.addToolBar('工具栏')
        self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "向右旋转90", self)
        self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "向左旋转90°", self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate))

        self.useListWidget = UsedListWidget_RAW(self)
        self.funcListWidget = FuncListWidget_RAW(self)
        self.stackedWidget = StackedWidget_RAW(self)
        self.fileSystemTreeView = FileSystemTreeView_RAW(self)
        # self.RAWView = GraphicsView_RAW()
        self.RAWView = RAWView()

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.useListWidget)
        self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        # self.setCentralWidget(self.graphicsView)
        self.setCentralWidget(self.RAWView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        self.setWindowTitle('医学图像处理')
        self.setWindowIcon(QIcon('icons/main.png'))
        self.src_img = None
        self.cur_img = None


    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        # self.graphicsView.update_image(img)
        self.RAWView.change_image(img)

    def change_image(self, img):
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        # self.graphicsView.change_image(img)
        self.RAWView.change_image(img)

    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    def right_rotate(self):
        # self.graphicsView.rotate(90)
        self.RAWView.rotate(90)

    def left_rotate(self):
        # self.graphicsView.rotate(-90)
        self.RAWView.rotate(-90)
