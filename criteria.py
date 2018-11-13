import torch
import torch.nn as nn
import numpy as np
import utils

class Criterion(nn.Module):
    def __init__(self, name=None):
        super(Criterion, self).__init__()
        self.name = name if name is not None else type(self).__name__

class Dice(Criterion):
    def __init__(self, name=None, threshold=0.5, eps=1e-4):
        super(Dice, self).__init__(name=name)
        self.threshold = threshold 
        self.eps = eps

    def forward(self, x, target, _img):
        positives = (torch.sigmoid(x) > self.threshold)
        target, mask = mateusz_to_target_and_mask(target)

        positives = positives[mask]
        target = target[mask]

        true_positives = (positives & target).to(torch.float32).sum()
        p_s = positives.to(torch.float32).sum()
        t_s = target.to(torch.float32).sum()

        dice = (2 * true_positives + self.eps) / (p_s + t_s + self.eps)
        return dice.mean()


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

class Histogrammer:
    def __init__(self, bins=20):
        self.n = 0
        self.bins = bins
        self.thresholds = np.linspace(0, 1, bins)
        self.trues = np.zeros(bins)
        self.falses = np.zeros(bins)

    @utils.size_adaptive
    def __call__(self, pred, target, _img):
        target, mask = mateusz_to_target_and_mask(target)
        target, pred = target[mask], pred[mask]

        probs = torch.sigmoid(pred).cpu()
        target = target.cpu() 

        true = torch.histc(
            probs[target], bins=self.bins, min=self.thresholds[0], max=self.thresholds[-1]
        ).numpy()
        false = torch.histc(
            probs[~target], bins=self.bins, min=self.thresholds[0], max=self.thresholds[-1]
        ).numpy()


        avg_true = self.trues * self.n + true
        avg_false = self.falses * self.n + false

        self.n += 1

        self.trues = avg_true / self.n
        self.falses = avg_false / self.n


if __name__ == '__main__':
    import nrrd
    from losses import SizeAdaptiveLoss

    def r(p):
        v, _ = nrrd.read(p + '.nrrd')
        return v

    pred = r('pred')
    img = r('img')
    lbl = r('lbl')

    lbl = torch.from_numpy(lbl)
    pred = torch.from_numpy(pred)

    loss_fn = SizeAdaptiveLoss(Dice())
    loss = loss_fn(pred, lbl)

    print(loss)
