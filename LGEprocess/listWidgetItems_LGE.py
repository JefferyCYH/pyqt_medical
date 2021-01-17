import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
from LGEprocess.flags_LGE import *
from skimage import exposure, img_as_float
import torch.utils.data as Datas
from LGEprocess import Network as Network
import nibabel as nib
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

def Reg(mov,fix):
    print(mov.shape)
    print(fix.shape)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = torch.device("cuda:0")
    print(torch.__version__)
    data = mov,fix
    dataloder = Datas.DataLoader(dataset=data, batch_size=2, shuffle=False)
    Flownet = Network.VXm(2).to(device)
    ##
    pretrained_dict = torch.load('./model/net_epoch_source-Flow-Network.pkl')
    model_dict = Flownet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    Flownet.load_state_dict(model_dict)

    with torch.no_grad():
        for epoch in range(1):
            for step, (mov,fix) in enumerate(dataloder):
                print(mov.shape)
                print(fix.shape)
                mov = mov.to(device).float()
                fix = fix.to(device).float()
                print(mov.shape)
                print(fix.shape)
                flow_field_x1, mov_fix, flow_field_x2, es_source = Flownet(fix, mov, fix)
                mov_fix = mov_fix[0, 0, :, :, :].data.cpu().numpy()
                print(mov_fix.shape)
                return mov_fix

def load_nii(path):
    image = nib.load(path)
    affine = image.affine
    image = np.asarray(image.dataobj)
    return image, affine

def normor(image):
    image -=image.mean()
    image /=image.std()
    return image

def crop_img(label_es, img, box_height=128, box_width=128):
    shape=label_es.shape()
    a = label_es.nonzero()
    a_x = a[0]
    a_x_middle = np.median(a[0])
    a_height = max((a_x)) - min((a_x)) + 1

    assert a_height < box_height, 'height小了'
    a_x_start = int(a_x_middle - box_height / 2)
    a_x_end = int(a_x_middle + box_height / 2)

    a_y = a[1]
    a_y_middle = np.median(a_y)
    a_width = max(a_y) - min(a_y) + 1
    # print(a_width,a_height)
    assert a_width < box_width, 'width小了'
    a_y_start = int(a_y_middle - box_width / 2)
    a_y_end = int(a_y_middle + box_width / 2)

    img_1 = img[a_x_start:a_x_end, a_y_start:a_y_end, :]
    #plt.imshow(img_1[:,:,5], cmap='gray')
    return img_1

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

class RegItem(MyItem_LGE):
    def __init__(self, parent=None):
        super(RegItem, self).__init__('配准', parent=parent)

    def __call__(self, img):
        path='./image/_es.nii.gz'
        fix=nib.load(path).get_data()
        img = np.transpose(img, (2, 1, 0))  # xyz-zyx
        img = normor(img)
        img = img[np.newaxis, np.newaxis, :, :, :]
        fix = np.transpose(fix, (2, 1, 0))  # xyz-zyx
        fix = normor(fix)
        fix = fix[np.newaxis, np.newaxis, :, :, :]
        mov=img

        img=Reg(mov,fix)
        img = np.transpose(img, (2, 1, 0))  # zyx-xyz
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



