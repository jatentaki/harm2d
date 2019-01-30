import torch, functools, inspect
import torch.nn as nn
import torch.nn.functional as F

class BCE(nn.Module):
    def __init__(self, masked=True):
        super(BCE, self).__init__()
        self.masked = masked

    def forward(self, prediction, mask, target):
        loss = F.binary_cross_entropy_with_logits(
            prediction, target, reduction='none'
        )

        return (loss * mask).mean()
