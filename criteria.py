import torch
import torch.nn as nn
import numpy as np
import utils

class Criterion(nn.Module):
    def __init__(self, name=None):
        super(Criterion, self).__init__()
        self.name = name if name is not None else type(self).__name__

class Tableau4(Criterion):
    def __init__(self, name='TP FP TN FN', threshold=0.5):
        super(Tableau4, self).__init__()
        self.name = name
        self.threshold = threshold

    def forward(self, x, target, _img):
        positive = (torch.sigmoid(x) > self.threshold)
        target, mask = mateusz_to_target_and_mask(target)

        positive = positive[mask]
        target = target[mask]

        tp = positive & target
        fp = positive & ~target
        tn = ~positive & ~target
        fn = ~positive & target

        return np.array([i.sum().item() for i in [tp, fp, tn, fn]])

class PrecRec(Criterion):
    def __init__(self, n_thresholds=10):
        self.n_thresholds = n_thresholds
        self.thresholds = torch.linspace(0, 1, n_thresholds + 2)[1:-1]
        self.classes = torch.zeros(4, n_thresholds, dtype=torch.int64)

    @utils.size_adaptive
    def __call__(self, prediction, mask, target):
        sigmoids = torch.sigmoid(prediction).cpu()
        mask = mask.to(torch.uint8).cpu()
        target = target[mask].to(torch.uint8).cpu()

        for i, threshold in enumerate(self.thresholds):
            positive = sigmoids > threshold

            positive = positive[mask]

            tp = positive & target
            fp = positive & ~target
            tn = ~positive & ~target
            fn = ~positive & target

            results = torch.tensor(
                [arr.sum().item() for arr in [tp, fp, tn, fn]],
                dtype=torch.int64
            )
            
            self.classes[:, i] += results

    def tp(self):
        return self.classes[0]

    def fp(self):
        return self.classes[1]

    def tn(self):
        return self.classes[2]

    def fn(self):
        return self.classes[3]

    def prec_rec(self):
        prec = self.tp().float() / (self.tp().float() + self.fp().float())
        rec = self.tp().float() / (self.tp().float() + self.fn().float())

        return prec, rec

    def best_f1(self):
        prec, rec = self.prec_rec()
        f1 = 2 * prec * rec / (prec + rec)
        argmax = f1.argmax().item()
        return f1[argmax].item(), self.thresholds[argmax].item()
