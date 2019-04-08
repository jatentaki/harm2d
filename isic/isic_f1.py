import torch, sys
import scipy.ndimage as ndi
import numpy as np
from torch_dimcheck import dimchecked

sys.path.append('..')
from utils import cut_to_match

def postprocess(img):
    label, n = ndi.label(img)
    if n == 0:
        return None
    hist, _ = np.histogram(label.flatten(), bins=np.arange(n+2) - 0.5)
    biggest_id = hist[1:].argmax() + 1
    biggest_blob = label == biggest_id
    filled = ndi.binary_fill_holes(biggest_blob)

    return filled

class IsicF1:
    name = 'IsicF1'
    def __init__(self, n_thresholds=10, range=(0.2, 0.8)):
        self.f_1 = np.zeros(n_thresholds, dtype=np.float32)
        self.thresholds = np.linspace(*range, n_thresholds + 2)[1:-1]
        self.n_processed = 0

    def process_one_img(self, prediction, mask, target):
        mask = mask.cpu().numpy() != 0.
        target = target.cpu().numpy().astype(np.bool)
        sigmoids = torch.sigmoid(prediction).cpu().numpy()
#        sigmoids = prediction.cpu().numpy()

        target = target[mask]

        for th_i, th in enumerate(self.thresholds):
            binary = sigmoids > th
            postprocessed = postprocess(binary)
            if postprocessed is None:
                continue
            binary = postprocessed[mask]
            
            tp =  binary & target
            fp =  binary & ~target
            #tn = ~binary & ~target
            fn = ~binary &  target

            tp = tp.sum()
            fp = fp.sum()
            fn = fn.sum()

            f_1 = 2 * tp / (2 * tp + fn + fp)

            if f_1 >= 0.65:
                self.f_1[th_i] += f_1

        self.n_processed += 1
        
    def __call__(self, prediction, mask, target):
        prediction = prediction[0]
        mask = cut_to_match(prediction, mask[0], n_pref=0)
        target = cut_to_match(prediction, target[0], n_pref=0)
        self.process_one_img(prediction, mask, target)
        return 0

    def get_dict(self):
        argmax = self.f_1.argmax()
        f_1 = self.f_1[argmax] / self.n_processed
        thres = self.thresholds[argmax]

        return {
            'isic_f1': f_1,
            'isic_thres': thres
        }

def process_one_img(prediction, mask, target, thresholds, logit_input=True):
    mask = mask.cpu().numpy() != 0.
    target = target.cpu().numpy().astype(np.bool)
    if logit_input:
        sigmoids = torch.sigmoid(prediction)
    else:
        sigmoids = prediction
    sigmoids = sigmoids.cpu().numpy()

    target = target[mask]

    ious = np.zeros_like(thresholds)
    for th_i, th in enumerate(thresholds):
        binary = sigmoids > th
        postprocessed = postprocess(binary)
        if postprocessed is None:
            continue
        binary = postprocessed[mask]
        
        inter = (binary & target).sum()
        union_2 = binary.sum() + target.sum()

        iou = inter / (union_2 - inter)

        iou = 0. if iou < 0.65 else iou

        ious[th_i] = iou
    
    return ious

class IsicIoU:
    name = 'IsicIoU'
    def __init__(self, n_thresholds=10, range=(0.2, 0.8)):
        self.iou = np.zeros(n_thresholds, dtype=np.float32)
        self.thresholds = np.linspace(*range, n_thresholds + 2)[1:-1]
        self.n_processed = 0

        
    def __call__(self, prediction, mask, target):
        prediction = prediction[0]
        mask = cut_to_match(prediction, mask[0], n_pref=0)
        target = cut_to_match(prediction, target[0], n_pref=0)

        ious = process_one_img(prediction, mask, target, self.thresholds)
        self.iou += ious
        self.n_processed += 1
        return 0

    def get_dict(self):
        argmax = self.iou.argmax()
        iou = self.iou[argmax] / self.n_processed
        thres = self.thresholds[argmax]

        return {
            'isic_iou': iou,
            'isic_thres': thres
        }

if __name__ == '__main__':
    import os, imageio
    import matplotlib.pyplot as plt

    difficult = '../../difficult'
    labels = '~/Storage/jatentaki/Datasets/isic2018/lbls'
    imgs = '~/Storage/jatentaki/Datasets/isic2018/imgs'

    for file in os.listdir(difficult):
        f = torch.from_numpy(np.load(os.path.join(difficult, file))['seg'])
        lbl = imageio.imread(os.path.join(labels, file[:-4] + '_segmentation.png'))
        img = imageio.imread(os.path.join(imgs, file[:-3] + 'jpg'))
        f = (torch.sigmoid(f) > 0.5).numpy()
        p = postprocess(f).astype(np.uint8)
        plt.imshow(p)
        plt.figure()
        plt.imshow(lbl)
        plt.figure()
        plt.imshow(img)
        plt.show()

#    p = 'edge_loss/harmonic_eval/'
#
#    f1 = IsicIoU()
#
#    for i in range(10):
#        pred = np.load(p + f'{i}/heatmap.npy')[None]
#        gt = np.load(p + f'{i}/g_truth.npy')[None]
#        mask = np.ones_like(gt)
#
#        pred = torch.from_numpy(pred)
#        gt = torch.from_numpy(gt)
#        mask = torch.from_numpy(mask)
#
#        f1(pred, mask, gt)
#    print(f1.get_dict())
