import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

def load_nii(path):
    image = nib.load(path)
    affine = image.affine
    image = np.asarray(image.dataobj)
    return image, affine


class FileSystemTreeView(QTreeView, QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setRootPath('.')
        self.setModel(self.fileSystemModel)
        # 隐藏size,date等列
        self.setColumnWidth(0, 200)
        self.setColumnHidden(1, True)
        self.setColumnHidden(2, True)
        self.setColumnHidden(3, True)
        # 不显示标题栏
        self.header().hide()
        # 设置动画
        self.setAnimated(True)
        # 选中不显示虚线
        self.setFocusPolicy(Qt.NoFocus)
        self.doubleClicked.connect(self.select_image)
        self.setMinimumWidth(200)

    def select_image(self, file_index):
        file_name = self.fileSystemModel.filePath(file_index)
        if file_name.endswith(('.jpg', '.png', '.bmp')) and self.mainwindow.datatype == "png":
            src_img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
            self.mainwindow.change_image(src_img)
        elif file_name.endswith('.nii.gz') and self.mainwindow.datatype == "lge":
            src_img_name = file_name
            src_img, affine = load_nii(file_name)
            print(src_img.shape)
            # src_img = sitk.ReadImage(file_name)
            # src_img=sitk.GetArrayFromImage(src_img)
            self.mainwindow.change_image(src_img)
        elif file_name.endswith('.nii.gz') and self.mainwindow.datatype == "petv":
            src_img = file_name
            self.mainwindow.change_image(src_img)
        elif file_name.endswith('.raw') and self.mainwindow.datatype == "raw":
            src_img = file_name
            self.mainwindow.change_image(src_img)

        