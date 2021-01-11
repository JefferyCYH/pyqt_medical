import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
from LGEprocess.flags_LGE import *


class MyItem_LGE(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem_LGE, self).__init__(name, parent=parent)
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(60, 60))  # size
        print('MyItem_LGE')

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

class LabelItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(LabelItem, self).__init__('添加GT', parent=parent)


    def __call__(self, label):
        # blank = np.zeros(img.shape, img.dtype)
        # img = cv2.addWeighted(img, self._alpha, blank, 1 - self._alpha, self._beta)
        return label

class NormItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(NormItem, self).__init__('归一化', parent=parent)

    def __call__(self, img):

        return img

class LightItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(LightItem, self).__init__('亮度', parent=parent)

    def __call__(self, img):

        return img

class ROIItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(ROIItem, self).__init__('ROI提取', parent=parent)

    def __call__(self, img):

        return img



class SegItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(SegItem, self).__init__('分割', parent=parent)

    def __call__(self, img):

        return img



