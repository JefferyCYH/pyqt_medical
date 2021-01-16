import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
from LGEprocess.flags_LGE import *
from skimage import exposure, img_as_float
import torch.utils.data as Datas
from LGEprocess import Network as Network
import os
import torch
def Seg(img):
    print(img.shape)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda:0")
    print(torch.__version__)
    data=img
    dataloder = Datas.DataLoader(dataset=data, batch_size=1, shuffle=False)
    Segnet = Network.DenseBiasNet(n_channels=1, n_classes=4).to(device)

    pretrained_dict = torch.load('./model/net_epoch_source-Seg-Network.pkl', map_location='cpu')
    model_dict = Segnet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    Segnet.load_state_dict(model_dict)

    with torch.no_grad():
        for epoch in range(1):
            for step, (img) in enumerate(dataloder):
                print(img.shape)
                img=img.to(device).float()
                print(img.shape)
                img=Segnet(img)
    img= img[0, 1, :, :, :] * 1 + img[0, 2, :, :, :] * 2 + img[0, 3, :, :, :] * 3
    img = img.data.cpu().numpy()
    print(img.shape)
    return img
def normor(image):
    image -=image.mean()
    image /=image.std()
    return image
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
        max = img.max()
        min = img.min()
        img = (img - min) / (max - min)
        return img

class LightItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(LightItem, self).__init__('亮度', parent=parent)
        self.alpha = 1

    def __call__(self, img):
        img = img_as_float(img)
        if (self.alpha <=1 & self.alpha >0):
            img = exposure.adjust_gamma(img, self.alpha)  # 图片调暗
        elif (self.alpha > 1):
            img = exposure.adjust_gamma(img, 0.5)  # 图片调亮
        else:
            print('请输入大于0的数字！')
        return img

class ROIItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(ROIItem, self).__init__('ROI提取', parent=parent)

    def __call__(self, img):


        return img


class SegItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(SegItem, self).__init__('分割', parent=parent)
        print('img')
    def __call__(self, img):
        img = np.transpose(img, (2, 1, 0))  # xyz-zyx
        img=normor(img)
        img = img[np.newaxis,np.newaxis, :, :, :]
        # print(img.shape)
        img=Seg(img)
        img=np.transpose(img,(2,1,0))#zyx-xyz
        print(img.shape)

        return img



