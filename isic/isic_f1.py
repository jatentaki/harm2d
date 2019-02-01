import torch, sys
import scipy.ndimage as ndi
import numpy as np
from torch_dimcheck import dimchecked

sys.path.append('..')
from utils import cut_to_match

class IsicF1:
    name = 'IsicF1'
    def __init__(self, n_thresholds=10, range=(0.2, 0.8)):
        self.f_1 = np.zeros(n_thresholds, dtype=np.float32)
        self.thresholds = np.linspace(*range, n_thresholds + 2)[1:-1]
        self.n_processed = 0

    def postprocess(self, img):
        label, n = ndi.label(img)
        if n == 0:
            return None
        hist, _ = np.histogram(label.flatten(), bins=np.arange(n+2) - 0.5)
        biggest_id = hist[1:].argmax() + 1
        biggest_blob = label == biggest_id
        filled = ndi.binary_fill_holes(biggest_blob)

        return filled

    def process_one_img(self, prediction, mask, target):
        mask = mask.cpu().numpy() != 0.
        target = target.cpu().numpy().astype(np.bool)
        sigmoids = torch.sigmoid(prediction).cpu().numpy()
#        sigmoids = prediction.cpu().numpy()

        target = target[mask]

        for th_i, th in enumerate(self.thresholds):
            binary = sigmoids > th
            postprocessed = self.postprocess(binary)
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

if __name__ == '__main__':
    p = 'edge_loss/harmonic_eval/'

    f1 = IsicF1()

    for i in range(10):
        pred = np.load(p + f'{i}/heatmap.npy')[None]
        gt = np.load(p + f'{i}/g_truth.npy')[None]
        mask = np.ones_like(gt)

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
        mask = torch.from_numpy(mask)

        f1(pred, mask, gt)
    print(f1.get_dict())
