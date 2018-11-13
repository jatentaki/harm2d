import torch, functools, inspect
import torch.nn as nn
import torch.nn.functional as F

class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, prediction, mask, target):
        target = target.to(torch.float32)
        mask = mask.to(torch.uint8)
        return F.binary_cross_entropy_with_logits(
            prediction[mask], target[mask]
        )
