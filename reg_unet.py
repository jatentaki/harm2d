import torch, itertools
import torch.nn as nn
import torch.nn.functional as F

from torch_localize import localized_module

from utils import cut_to_match, upsample


def size_is_pow2(t):
    ''' Check if the trailing spatial dimensions are powers of 2 '''
    return all(s % 2 == 0 for s in t.size()[2:])


class AttentionGate(nn.Module):
    def __init__(self, n_features):
        super(AttentionGate, self).__init__()
        self.n_features = n_features
    
        self.conv1 = nn.Conv2d(self.n_features, self.n_features, 1)
        self.conv2 = nn.Conv2d(self.n_features, self.n_features, 1)

    def forward(self, inp):
        y = self.conv1(inp)
        y = F.relu(y)
        y = self.conv2(y)
        g = torch.sigmoid(y)

        return g * inp


class NormalizedConv2(nn.Module):
    def __init__(self, in_, out_, size=5, momentum=0.1, **kwargs):
        super(NormalizedConv2, self).__init__()

        self.normalization = nn.BatchNorm2d(in_, momentum=momentum)
        self.conv = nn.Conv2d(
            in_, out_, size, bias=True, **kwargs
        )

    def forward(self, y):
        y = self.normalization(y)
        y = self.conv(y)

        return y


@localized_module
class UnetDownBlock(nn.Module):
    def __init__(self, in_, out_, size=5, **kwargs):
        super(UnetDownBlock, self).__init__()

        self.in_ = in_
        self.out_ = out_
        
        self.conv1 = NormalizedConv2(in_, out_, size=size, **kwargs)
        self.nonl1 = AttentionGate(out_)

        self.conv2 = NormalizedConv2(out_, out_, size=size, **kwargs)
        self.nonl2 = AttentionGate(out_)
    

    def forward(self, y):
        y = self.conv1(y)
        y = self.nonl1(y)
        y = self.conv2(y)
        y = self.nonl2(y)

        return y


    def __repr__(self):
        fmt = 'UnetDownBlock ({}) {} -> {}'
        msg = fmt.format(self.name, self.in_, self.out_)
        return msg


@localized_module
class UnetUpBlock(nn.Module):
    def __init__(self, bottom, horizontal, out,
                 size=5, last_nonl=True, **kwargs):
        super(UnetUpBlock, self).__init__()

        self.last_nonl = last_nonl
        self.bottom = bottom
        self.horizontal = horizontal
        self.cat = bottom + horizontal
        self.out = out

        self.conv1 = NormalizedConv2(self.cat, self.cat, size, **kwargs)
        self.nonl1 = AttentionGate(self.cat)

        if self.last_nonl:
            self.conv2 = NormalizedConv2(self.cat, self.out, size, **kwargs)
            self.nonl2 = AttentionGate(self.out)
        else:
            self.conv2 = NormalizedConv2(self.cat, 1, size, **kwargs)


    def forward(self, bottom, horizontal):
        horizontal = cut_to_match(bottom, horizontal)
        y = torch.cat([bottom, horizontal], dim=1)
        y = self.conv1(y)
        y = self.nonl1(y)
        y = self.conv2(y)
        if self.last_nonl:
            y = self.nonl2(y)

        return y


    def __repr__(self):
        fmt = 'UnetUpBlock ({}) {} x {} -> {}'
        msg = fmt.format(self.name, self.bottom, self.horizontal, self.out)
        return msg

unet_default_dimensions = [
    1,
    56,
    34,
    24
]

def repr_to_dims(repr):
    return sum(n * (2 * j + 1) for j, n in enumerate(repr))


@localized_module
class Unet(nn.Module):
    def __init__(self, dimensions=unet_default_dimensions, classes=1, **kwargs):
        super(Unet, self).__init__()

        self.dimensions = dimensions
        self.classes = classes

        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dimensions[:-1],
                                              self.dimensions[1:])):

            block = UnetDownBlock(
                d_in, d_out, name='down_{}'.format(i), **kwargs
            )
            self.path_down.append(block)

        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor) in enumerate(zip(self.dimensions[::-1][:-2],
                                               self.dimensions[::-1][1:-1])):

            block = UnetUpBlock(
                d_bot, d_hor, d_hor, name='up_{}'.format(i), **kwargs,
            )
            self.path_up.append(block)

        self.logit_conv = nn.Conv2d(dimensions[1], self.classes, 1)


    def __repr__(self):
        fmt = ('Unet:\n'
               'path_down: [\n'
               '\t{}\n'
               ']\n'
               'path_up: [\n'
               '\t{}\n'
               ']\n'
               'logits: {} -> {}'
              )
        pd = '\n\t'.join(repr(l) for l in self.path_down)
        pu = '\n\t'.join(repr(l) for l in self.path_up)
        msg = fmt.format(pd, pu, self.dimensions[1], self.classes)
        return msg


    def forward(self, inp):
        features = inp
        features_down = []
        for i, layer in enumerate(self.path_down):
            if i != 0:
                if not size_is_pow2(features):
                    fmt = "Trying to downsample feature map of size {}"
                    msg = fmt.format(features.size())
                    raise RuntimeError(msg)

                features = F.avg_pool2d(features, 2)
            features = layer(features)
            features_down.append(features)
        
        f_b = features_down[-1]
        features_horizontal = features_down[-2::-1]
        
        for i, (layer, f_h) in enumerate(zip(self.path_up, features_horizontal)):
            f_b = upsample(f_b, scale_factor=2)
            f_b = layer(f_b, f_h)

        return self.logit_conv(f_b)

    def l2_params(self):
        return [p for n, p in self.named_parameters() if 'bias' not in n]

    def nr_params(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]


if __name__ == '__main__':
    import unittest
    
    class Tests(unittest.TestCase):
        def test_inequal_output_padding(self):
            unet = Unet()
            input = torch.zeros(2, 1, 104, 104)
            output = unet(input)
            self.assertEqual(torch.Size([2, 1, 24, 24]), output.size())
    
    unittest.main()
