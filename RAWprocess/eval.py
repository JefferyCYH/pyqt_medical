import numpy as np
import torch
import torch.optim
import torch.nn as nn
from .models.fewshot import FewShotSeg

def getMask(label):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask_1 = torch.where(label == 1,
                          torch.ones_like(label), torch.zeros_like(label))
    fg_mask_2 = torch.where(label == 2,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label == 0,
                          torch.ones_like(label), torch.zeros_like(label))

    return {'fg_mask1': fg_mask_1,
            'fg_mask2': fg_mask_2,
            'bg_mask': bg_mask}
def seg(query_img):
    model = FewShotSeg()
    model = nn.DataParallel(model.cuda(), device_ids=[0])
    model.load_state_dict(torch.load('./RAWprocess/paras.pth', map_location='cpu'))
    model.eval()

    supp_img = np.fromfile('./RAWprocess/image/supp_img.raw', dtype=np.int16)
    supp_img = np.where(supp_img < 900, 900, supp_img)
    supp_img = np.where(supp_img > 1400, 1400, supp_img)
    supp_img = np.reshape(supp_img, (1, 1, 200, 160, 160)).astype(np.float32)
    supp_img = (supp_img - 900)/500
    supp_img = torch.tensor(supp_img)
    supp_label = np.fromfile('./RAWprocess/image/supp_label.raw', dtype=np.int16)
    supp_label = np.reshape(supp_label, (1, 200, 160, 160))
    supp_label = torch.tensor(supp_label)
    supp_mask = getMask(supp_label)
    supp_imgs = [[supp_img], [supp_img]]
    supp_fg_masks = [[supp_mask['fg_mask1']], [supp_mask['fg_mask2']]]
    supp_bg_masks = [[supp_mask['bg_mask']], [supp_mask['bg_mask']]]

    # query_img = np.fromfile('./image/query_img.raw', dtype=np.int16)
    query_img = np.where(query_img < 900, 900, query_img)
    query_img = np.where(query_img > 1400, 1400, query_img)
    query_img = np.reshape(query_img, (1, 1, 200, 160, 160)).astype(np.float32)
    query_img = (query_img - 900)/500
    query_img = torch.tensor(query_img)
    query_imgs = [query_img]

    query_pred, _, _, _ = model(supp_imgs, supp_fg_masks, supp_bg_masks, query_imgs)

    pred = query_pred[0].detach().cpu().numpy().argmax(axis=0).astype(np.int16)
    return pred
    pred.tofile("pred.raw")