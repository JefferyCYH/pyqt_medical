import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
from RAWprocess.flags_RAW import *
from RAWprocess.eval import seg


class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(100, 60))  # size

    def get_params(self):
        protected = [v for v in dir(self) if v.startswith('_') and not v.startswith('__')]
        param = {}
        for v in protected:
            param[v.replace('_', '', 1)] = self.__getattribute__(v)
        return param

    def update_params(self, param):
        for k, v in param.items():
            if '_' + k in dir(self):
                self.__setattr__('_' + k, v)


class SegItem(MyItem):
    def __init__(self, parent=None):
        super(SegItem, self).__init__('分割', parent=parent)

    def __call__(self, img):
        img = seg(img)
        return img

class GrayscaleItem(MyItem):
    def __init__(self, parent=None):
        super(GrayscaleItem, self).__init__('灰度调节', parent=parent)
        self._kgray = 0

    def __call__(self, img):
        img = img+self._kgray
        img = np.where(img > 2048, 2048, img)
        img = np.where(img < 0, 0, img)
        return img

class ContrastItem(MyItem):
    def __init__(self, parent=None):
        super(ContrastItem, self).__init__('对比度调节', parent=parent)
        self._kcontrast = 1

    def __call__(self, img):
        img = img*self._kcontrast
        img = np.where(img > 2048, 2048, img)
        img = np.where(img < 0, 0, img)
        return img

class CutItem(MyItem):
    def __init__(self, parent=None):
        super(CutItem, self).__init__('图像切割', parent=parent)
        self._kdown = 0
        self._kback = 0
        self._kleft = 0
        self._kup = 199
        self._kfor = 159
        self._kright = 159

    def __call__(self, img):
        img = img[self._kdown:self._kup, self._kback:self._kfor, self._kleft:self._kright]
        return img

class SaveItem(MyItem):
    def __init__(self, parent=None):
        super(SaveItem, self).__init__('图像保存', parent=parent)

    def __call__(self, img):
        img.astype(np.int16).tofile("save.raw")
        return img