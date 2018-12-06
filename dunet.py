import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_localize import localized_module

def upscale2x(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


@localized_module
class Conv(nn.Sequential):
    def __init__(self, n_in, n_out, size, **kwargs):
        padding = size // 2

        bn = nn.InstanceNorm2d(n_in)
        relu = nn.ReLU(inplace=True)
        conv = nn.Conv2d(n_in, n_out, size, padding=padding, **kwargs)

        super(Conv, self).__init__(bn, relu, conv)


class Dense(nn.Sequential):
    def __init__(self, n_repetitions, n_in=48):
        sequence = []
        for i in range(n_repetitions):
            in_f = n_in if i == 0 else 48
            one_by_one = Conv(in_f, 192, 1, name=f'conv1x1_{i}')
            three_by_three = Conv(192, 48, 3, name=f'conv3x3_{i}')

            sequence.append(one_by_one)
            sequence.append(three_by_three)
        
        super(Dense, self).__init__(*sequence)


class Transition(nn.Sequential):
    def __init__(self, n=48):
        conv = Conv(n, n, 1)
        pool = nn.AvgPool2d(2)

        super(Transition, self).__init__(conv, pool)


class Upsample(nn.Module):
    def __init__(self, n_up, n_out, n_hor=48):
        super(Upsample, self).__init__()

        self.conv = Conv(n_up + n_hor, n_out, 3)

    def forward(self, x, h):
        x_up = upscale2x(x)
        s = torch.cat([x_up, h], dim=1)

        return self.conv(s)
        

class Dunet(nn.Module):
    def __init__(self):
        super(Dunet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.dense1 = Dense(6, n_in=96)
        self.trans1 = Transition()

        self.dense2 = Dense(12)
        self.trans2 = Transition()
        
        self.dense3 = Dense(36)
        self.trans3 = Transition()
        
        self.dense4 = Dense(24)

        self.up1 = Upsample(48, 768)
        self.up2 = Upsample(768, 384)
        self.up3 = Upsample(384, 96)
        self.up4 = Upsample(96, 96, n_hor=96)

        self.conv_last = Conv(96, 64, 3)
        self.logit_conv = Conv(64, 1, 1)

    def forward(self, x):
        d0 = self.conv1(x)
        t0 = self.pool1(d0)

        d1 = self.dense1(t0)
        t1 = self.trans1(d1)

        d2 = self.dense2(t1)
        t2 = self.trans2(d2)

        d3 = self.dense3(t2)
        t3 = self.trans3(d3)

        d4 = self.dense4(t3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, d0)

        u5 = upscale2x(u4)
        u5 = self.conv_last(u5)
        logits = self.logit_conv(u5)

        return logits


    def l2_params(self):
        return [p for n, p in self.named_parameters() if 'bias' not in n]


    def nr_params(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


if __name__ == '__main__':
    input = torch.randn(2, 3, 224, 224)

    net = Dunet()
    out = net(input)
    assert out.shape == (2, 1, 224, 224)
