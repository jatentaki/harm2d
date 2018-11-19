import torch, functools, inspect
import torch.nn as nn
import torch.nn.functional as F

class BCE(nn.Module):
    def __init__(self, masked=True):
        super(BCE, self).__init__()
        self.masked = masked

    def forward(self, *args):
        if self.masked:
            prediction, mask, target = args
            mask = mask.to(torch.uint8)
            prediction = prediction[mask]
            target = target[mask]

        else:
            prediction, target = args

        return F.binary_cross_entropy_with_logits(prediction, target.to(torch.float32))
