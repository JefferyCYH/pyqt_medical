import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
import torch.utils.data as Datas
from FDCMRprocess import Network as Network
import nibabel as nib
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import SimpleITK as sitk
import torch
def threshold(data):
 data[np.where((0.5 > data))] = 0
 data[np.where((0.5 <= data) & (data < 1.5))] = 1
 data[np.where((1.5 <= data) & (data < 2.5))] = 2
 data[np.where((2.5 <= data) & (data < 3.5))] = 3
 return data

def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res
def dice(img1, img2, labels=None, nargout=1):
 '''
 Dice [1] volume overlap metric

 The default is to *not* return a measure for the background layer (label = 0)

 [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
 Ecology 26.3 (1945): 297-302.

 Parameters
 ----------
 vol1 : nd array. The first volume (e.g. predicted volume)
 vol2 : nd array. The second volume (e.g. "true" volume)
 labels : optional vector of labels on which to compute Dice.
     If this is not provided, Dice is computed on all non-background (non-0) labels
 nargout : optional control of output arguments. if 1, output Dice measure(s).
     if 2, output tuple of (Dice, labels)

 Output
 ------
 if nargout == 1 : dice : vector of dice measures for each labels
 if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
     dice was computed
 '''
 if labels is None:
  labels = np.unique(np.concatenate((img1, img2)))  # 输出一维数组
  labels = np.delete(labels, np.where(labels == 0))  # remove background

 dicem = np.zeros(len(labels))
 for idx, lab in enumerate(labels):
  top = 2 * np.sum(np.logical_and(img1 == lab, img2 == lab))
  bottom = np.sum(img1 == lab) + np.sum(img2 == lab)
  bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon. 机器最小的正数
  dicem[idx] = top / bottom

 if nargout == 1:
  return dicem
 else:
  return (dicem, labels)
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

    print("crop_label",label_es.shape)
    print("crop_img",img.shape)
    a = label_es.nonzero()
    a_x = a[0]
    a_x_middle = np.median(a[0])
    a_height = max((a_x)) - min((a_x)) + 1

    assert a_height < box_height, 'height小了'
    a_x_start = max(0,int(a_x_middle - box_height / 2))
    if(int(a_x_middle - box_height / 2)>=0):
        a_x_end = int(a_x_middle + box_height / 2)
    else:
        a_x_end=box_height

    a_y = a[1]
    a_y_middle = np.median(a_y)
    a_width = max(a_y) - min(a_y) + 1
    # print(a_width,a_height)
    assert a_width < box_width, 'width小了'
    a_y_start = max(0,int(a_y_middle - box_width / 2))
    if(int(a_y_middle - box_width / 2)>=0):
        a_y_end = int(a_y_middle + box_width / 2)
    else:
        a_y_end=box_width

    img_1 = img[a_x_start:a_x_end, a_y_start:a_y_end, :]
    label_res=label_es[a_x_start:a_x_end, a_y_start:a_y_end, :]
    print('img1',img_1.shape)
    #plt.imshow(img_1[:,:,5], cmap='gray')
    return img_1,label_res

class MyItem_4D(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem_4D, self).__init__(name, parent=parent)
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(60, 60))  # size
        print('MyItem_4D')

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

class LabelItem(MyItem_4D):
    def __init__(self, parent=None):
        super(LabelItem, self).__init__('添加GT', parent=parent)

    def __call__(self, label):

        # self.label = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件","./", "All Files (*);;Text Files (*.txt)")
        # blank = np.zeros(img.shape, img.dtype)
        # img = cv2.addWeighted(img, self._alpha, blank, 1 - self._alpha, self._beta)
        return label

class NormItem(MyItem_4D):
    def __init__(self, parent=None):
        super(NormItem, self).__init__('归一化', parent=parent)

    def __call__(self, img):
        max = img.max()
        min = img.min()
        img = (img - min) / (max - min)
        return img

class LightItem(MyItem_4D):
    def __init__(self, parent=None):
        super(LightItem, self).__init__('Dice', parent=parent)


    def __call__(self, img,label):

        test_img=img[0,:,:,:]
        test_img=threshold(test_img)
        test_img=convert_to_one_hot(test_img)
        test_label=threshold(label)
        test_label=convert_to_one_hot(test_label)
        test_img=test_img[np.newaxis,:,:,:,:]
        test_label = test_label[np.newaxis, :, :, :, :]
        print("test_labelshape", test_label.shape)
        print("test_imgshape", test_img.shape)
        mm1 = test_img[0, 1, :, :, :] * 1 + test_img[0, 2, :, :, :] * 2 + test_img[0, 3, :, :, :] * 3
        mm2 = test_label[0, 1, :, :, :] * 1 + test_label[0, 2, :, :, :] * 2 + test_label[0, 3, :, :, :] * 3
        testdice = dice(mm1, mm2, nargout=1)
        print(testdice)
        self.showmsg(testdice)


        # self.label_1 = QLabel(self)
        # self.label_1.setText(self,'右心室：'+str(testdice[0])+'左心肌：'+str(testdice[1])+'左心室：'+str(testdice[2]))


        return img,label

    def showmsg(self,testdice):
        QMessageBox.information(None, 'Dice计算',
                                '右心室：' + str(testdice[0]) + '左心肌：' + str(testdice[1]) + '左心室：' + str(testdice[2]),
                                QMessageBox.Yes | QMessageBox.No)

class ROIItem(MyItem_4D):
    def __init__(self, parent=None):
        super(ROIItem, self).__init__('ROI提取', parent=parent)


    def __call__(self, img,label):

        # label_path=self.label_path_edit.text()

        # label_path='./image/patient001_frame01_gt.nii.gz'

        # label=nib.load(label_path).get_data()
        print("label_shape",label.shape)
        print("img_shape",img.shape)
        t,x,y,z=img.shape

        use_img=np.zeros((t,128,128,z))
        use_label=np.zeros((128,128,z))
        for i in range(0,t):
            cur_img,use_label=crop_img(label_es=label,img=img[i,:,:,:],box_height=128,box_width=128)
            use_img[i,:,:,:]=cur_img


        return use_img, use_label
            # label=use_label





class RegItem(MyItem_4D):
    def __init__(self, parent=None):
        super(RegItem, self).__init__('配准', parent=parent)

    def __call__(self, img,label):
        # path='./image/_es.nii.gz'
        # fix=nib.load(path).get_data()
        # pet_image = sitk.ReadImage(img)
        # pet_arr = sitk.GetArrayFromImage(pet_image)
        img = np.transpose(img, (0,3, 2, 1))  # txyz-tzyx
        fix=img[-1,:,:,:]
        fix = fix[np.newaxis, np.newaxis, :, :, :]
        t, z, y, x = img.shape
        use_img = np.zeros((t, x, y, z))
        for i in range(0, t):
            cur_img = img[i, :, :, :]
            cur_img = cur_img[np.newaxis, np.newaxis, :, :, :]
            mov=cur_img
            wrap=Reg(mov,fix)
            wrap = np.transpose(wrap, (2, 1, 0))  # zyx-xyz
            use_img[i, :, :, :] = wrap
        return use_img,label

class SegItem(MyItem_4D):
    def __init__(self, parent=None):
        super(SegItem, self).__init__('分割', parent=parent)
    def __call__(self, img,label):
        img = np.transpose(img, (0,3, 2, 1))  # txyz-tzyx
        t, z, y, x = img.shape
        use_img=np.zeros((t,x,y,z))
        for i in range(0, t):
            cur_img=img[i,:,:,:]
            # cur_img=normor(cur_img)
            cur_img = cur_img[np.newaxis,np.newaxis, :, :, :]
            print("cur_img:shape",img.shape)
            cur_img=Seg(cur_img)
            cur_img=np.transpose(cur_img,(2,1,0))#zyx-xyz
            use_img[i,:,:,:]=cur_img

        return use_img,label



