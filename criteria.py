import torch
import torch.nn as nn
import utils

class Criterion(nn.Module):
    def __init__(self, name=None):
        super(Criterion, self).__init__()
        self.name = name if name is not None else type(self).__name__

class PrecRec(Criterion):
    def __init__(self, n_thresholds=10, masked=True, eps=0.001):
        self.name = 'Precision recall'
        self.n_thresholds = n_thresholds
        self.thresholds = torch.linspace(0, 1, n_thresholds + 2)[1:-1]
        if torch.cuda.is_available():
            self.thresholds = self.thresholds.cuda()

        self.eps = eps
        self.classes = torch.zeros(4, n_thresholds, dtype=torch.int64)
        if not masked:
            raise ValueError("Unmasked PrecRec is deprecated")

    def calculate_tp_fp_tn_fn(self, prediction, mask, target):
        if not prediction.size(0) == 1:
            msg = f'Expected single batch item, got {prediction.size(0)}'
            raise ValueError(msg)

        mask = mask != 0.
        prediction = prediction[mask]
        target = target[mask]

        sigmoids = torch.sigmoid(prediction)
        target = target.to(torch.uint8)
        results = []
        for i, threshold in enumerate(self.thresholds):
            positive = sigmoids > threshold

            tp = positive & target
            fp = positive & ~target
            tn = ~positive & ~target
            fn = ~positive & target

            results.append([arr.sum().item() for arr in [tp, fp, tn, fn]])

        return results

    @utils.size_adaptive
    def __call__(self, prediction, mask, target):
        for batch_i in range(prediction.size(0)):
            results = self.calculate_tp_fp_tn_fn(
                prediction[batch_i], mask[batch_i], target[batch_i]
            )

            for threshold_i, result in enumerate(results):
                result = torch.tensor(result, dtype=torch.int64)
                self.classes[:, threshold_i] += result

        return torch.tensor(0.)

    def reset(self):
        self.classes[:] = 0

    def tp(self):
        return self.classes[0].float()

    def fp(self):
        return self.classes[1].float()

    def tn(self):
        return self.classes[2].float()

    def fn(self):
        return self.classes[3].float()

    def prec_rec(self):
        prec = self.tp() / (self.tp() + self.fp() + self.eps)
        rec  = self.tp() / (self.tp() + self.fn() + self.eps)

        return prec, rec

    def iou(self):
        return self.tp() / (self.tp() + self.fp() + self.fn() + self.eps)

    def best_iou(self):
        iou = self.iou()
        argmax = iou.argmax().item()
        return iou[argmax].item(), self.thresholds[argmax].item()

    def best_f1(self):
        prec, rec = self.prec_rec()
        f1 = 2 * prec * rec / (prec + rec)
        argmax = f1.argmax().item()
        return f1[argmax].item(), self.thresholds[argmax].item()

    def get_dict(self):
        iou, iou_t = self.best_iou()
        f1, f1_t = self.best_f1()

        return {
            'f1': f1,
            'f1_thres': f1_t,
            'iou': iou,
            'iou_thres': iou_t
        }

class IsicPrecRec(PrecRec):
    def __init__(self, n_thresholds=10, masked=True, eps=0.001):
        super(IsicPrecRec, self).__init__(n_thresholds, masked=masked, eps=eps)

        self.isic_f1 = torch.zeros(n_thresholds, dtype=torch.float32)
        self.n_processed = 0

    @utils.size_adaptive
    def __call__(self, prediction, mask, target):
        for batch_i in range(prediction.size(0)):
            results = self.calculate_tp_fp_tn_fn(
                prediction[batch_i], mask[batch_i], target[batch_i]
            )

            for threshold_i, result in enumerate(results):
                result_t = torch.tensor(result, dtype=torch.int64)
                self.classes[:, threshold_i] += result_t

                # ISIC part
                tp, fp, tn, fn = result
                prec = tp / (tp + fp + self.eps)
                rec  = tp / (tp + fn + self.eps)
                f1 = 2 * prec * rec / (prec + rec + self.eps)
                if f1 >= 0.65:
                    self.isic_f1[threshold_i] += f1

            self.n_processed += 1

        return torch.tensor(0.)


    def best_isic_f1(self):
        argmax = self.isic_f1.argmax().item()
        f1 = self.isic_f1[argmax].item() / self.n_processed
        thres = self.thresholds[argmax].item()

        return f1, thres

    def get_dict(self):
        base = super(IsicPrecRec, self).get_dict()
        isic_f1, isic_thres = self.best_isic_f1()

        return {
            **base,
            'isic_f1': isic_f1,
            'isic_thres': isic_thres
        }
            
