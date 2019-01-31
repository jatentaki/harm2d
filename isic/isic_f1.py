import torch
from torch_dimcheck import dimchecked

class IsicF1:
    def __init__(self, n_thresholds=100, eps=0.001):
        self.f_1 = torch.zeros(n_thresholds, dtype=torch.float32)
        self.thresholds = torch.linspace(0., 1., n_thresholds + 2)[1:-1]
        self.n_processed = 0

    def process_one_img(self, prediction, mask, target):
        mask = mask != 0.
        target = target.to(torch.uint8)[mask]
        prediction = prediction[mask]

        sigmoids = torch.sigmoid(prediction)

        for th_i, th in enumerate(self.thresholds):
            binary = sigmoids > th
            
            tp =  binary &  target
            fp =  binary & ~target
            #tn = ~binary & ~target
            fn = ~binary &  target
            
            tp = tp.sum().item()
            fn = fn.sum().item()
            fp = fn.sum().item()

            f_1 = 2 * tp / (2 * tp + fn + fp)

            if f_1 >= 0.65:
                self.f_1[th_i] += f_1

        self.n_processed += 1
        
    @dimchecked
    @utils.size_adaptive
    def __call__(self, prediction: ['b', 1, 'h', 'w'],
                       mask: ['b', 1, 'h', 'w'],
                       target: ['b', 1, 'h', 'w']):

        for b in range(prediction.size(0)):
            self.process_one_img(prediction[b], mask[b], target[b])

    def get_dict(self):
        argmax = self.isic_f1.argmax().item()
        f_1 = self.isic_f1[argmax].item() / self.n_processed
        thres = self.thresholds[argmax].item()

        return {
            'isic_f1': f_1,
            'isic_thres': thres
        }
            
