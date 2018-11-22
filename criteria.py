import torch
import torch.nn as nn
import utils

class Criterion(nn.Module):
    def __init__(self, name=None):
        super(Criterion, self).__init__()
        self.name = name if name is not None else type(self).__name__

class PrecRec(Criterion):
    def __init__(self, n_thresholds=10, masked=True):
        self.n_thresholds = n_thresholds
        self.thresholds = torch.linspace(0, 1, n_thresholds + 2)[1:-1].cuda()
        self.classes = torch.zeros(4, n_thresholds, dtype=torch.int64)
        self.masked = masked

    @utils.size_adaptive
    def __call__(self, *args):
        if self.masked:
            prediction, mask, target = args
            mask = mask.to(torch.uint8)
            prediction = prediction[mask]
            target = target[mask]
        else:
            prediction, target = args

        sigmoids = torch.sigmoid(predictions)
        target = target.to(torch.uint8)
        for i, threshold in enumerate(self.thresholds):
            positive = sigmoids > threshold

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
        return self.classes[0].float()

    def fp(self):
        return self.classes[1].float()

    def tn(self):
        return self.classes[2].float()

    def fn(self):
        return self.classes[3].float()

    def prec_rec(self):
        prec = self.tp() / (self.tp() + self.fp())
        rec  = self.tp() / (self.tp() + self.fn())

        return prec, rec

    def iou(self):
        return self.tp() / (self.tp() + self.fp() + self.fn())

    def best_iou(self):
        iou = self.iou()
        argmax = iou.argmax().item()
        return iou[argmax].item(), self.thresholds[argmax].item()

    def best_f1(self):
        prec, rec = self.prec_rec()
        f1 = 2 * prec * rec / (prec + rec)
        argmax = f1.argmax().item()
        return f1[argmax].item(), self.thresholds[argmax].item()

class ISICIoU(Criterion):
    def __init__(self, n_thresholds=10, masked=True):
        self.n_thresholds = n_thresholds
        self.thresholds = torch.linspace(0, 1, n_thresholds + 2)[1:-1].cuda()
        self.results = [utils.AvgMeter() for t in self.thresholds]
        self.masked = masked

    @utils.size_adaptive
    def __call__(self, *args):
        if self.masked:
            prediction, mask, target = args
            mask = mask.to(torch.uint8)
            prediction = prediction[mask]
            target = target[mask]
        else:
            prediction, target = args

        sigmoids = torch.sigmoid(prediction)

        target = target.to(torch.uint8)
        b = args[0].shape[0]
        target = target.reshape(b, -1)
        sigmoids = sigmoids.reshape(b, -1)

        for i, threshold in enumerate(self.thresholds):
            positive = sigmoids > threshold

            tp = (positive & target).sum(dim=1).float()
            fp = (positive & ~target).sum(dim=1).float()
            fn = (~positive & target).sum(dim=1).float()

            ious = tp / (tp + fp + fn)
            ious[ious < 0.65] = 0.

            for iou in ious:
                self.results[i].update(iou)

    def best_iou(self):
        results = torch.tensor([m.avg for m in self.results])
        argmax = results.argmax().item()
        return results[argmax].item(), self.thresholds[argmax].item()
