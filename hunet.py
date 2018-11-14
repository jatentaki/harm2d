import torch, itertools
import torch.nn as nn
import torch.nn.functional as F

import harmonic
from harmonic import d2

from torch_localize import localized_module
from torch_dimcheck import dimchecked

from utils import cut_to_match

def size_is_pow2(t):
    ''' Check if the trailing spatial dimensions are powers of 2 '''
    return all(s % 2 == 0 for s in t.size()[-2:])


class BNConv2d(nn.Module):
    def __init__(self, repr_in, repr_out, size):
        super(BNConv2d, self).__init__()

        self.bn = d2.BatchNorm2d(repr_in)
        self.conv = d2.HConv2d(repr_in, repr_out, size)

    def forward(self, x):
        return self.conv(self.bn(x))

@localized_module
class UnetDownBlock(nn.Module):
    def __init__(self, in_repr, out_repr, size=5, first_nonl=True):
        super(UnetDownBlock, self).__init__()

        self.first_nonl = first_nonl

        self.in_repr = in_repr
        self.out_repr = out_repr
        
        if first_nonl:
            self.nonl1 = d2.ScalarGate2d(in_repr)
            self.conv1 = BNConv2d(in_repr, out_repr, size)
        else:
            self.conv1 = d2.HConv2d(in_repr, out_repr, size)

        self.nonl2 = d2.ScalarGate2d(out_repr)
        self.conv2 = BNConv2d(out_repr, out_repr, size)
    

    @dimchecked
    def forward(self, y: [2, 'b', 'fi', 'hi', 'wi']
               )      -> [2, 'b', 'fo', 'ho', 'wo']:
        if self.first_nonl:
            y = self.nonl1(y)
        y = self.conv1(y)
        y = self.nonl2(y)
        y = self.conv2(y)

        return y


    def __repr__(self):
        fmt = 'UnetDownBlock ({}) {} -> {}.'
        msg = fmt.format(self.name, self.in_repr, self.out_repr)
        return msg


@localized_module
class UnetMiddleBlock(nn.Module):
    def __init__(self, in_repr, mid_repr, out_repr, size=5):
        super(UnetMiddleBlock, self).__init__()

        self.in_repr = in_repr
        self.mid_repr = mid_repr
        self.out_repr = out_repr

        self.nonl1 = d2.ScalarGate2d(in_repr)
        self.conv1 = BNConv2d(in_repr, mid_repr, size=size)

        self.nonl2 = d2.ScalarGate2d(mid_repr)
        self.conv2 = BNConv2d(mid_repr, out_repr, size=size)

    def __repr__(self):
        fmt = 'UnetMiddleBlock ({}) {} -> {} -> {}.'
        msg = fmt.format(self.name, self.in_repr, self.mid_repr, self.out_repr)
        return msg
        
    @dimchecked
    def forward(self, y: [2, 'b', 'fi', 'hi', 'wi']
               )      -> [2, 'b', 'fo', 'ho', 'wo']:

        y = self.nonl1(y)
        y = self.conv1(y)
        y = self.nonl2(y)
        y = self.conv2(y)

        return y


@localized_module
class UnetUpBlock(nn.Module):
    def __init__(self, bottom_repr, horizontal_repr, out_repr, size=5):
        super(UnetUpBlock, self).__init__()

        self.bottom_repr = bottom_repr
        self.horizontal_repr = horizontal_repr
        self.cat_repr = harmonic.cat_repr(bottom_repr, horizontal_repr)
        self.out_repr = out_repr

        self.nonl1 = d2.ScalarGate2d(self.cat_repr)
        self.conv1 = BNConv2d(self.cat_repr, self.cat_repr, size)

        self.nonl2 = d2.ScalarGate2d(self.cat_repr)
        self.conv2 = BNConv2d(self.cat_repr, self.out_repr, size)


    @dimchecked
    def forward(self, bottom:     [2, 'b', 'fb', 'hb', 'wb'],
                      horizontal: [2, 'b', 'fh', 'hh', 'wh']
               )               -> [2, 'b', 'fo', 'ho', 'wo']:

        horizontal = cut_to_match(bottom, horizontal, n_pref=3)
        y = d2.cat2d(bottom, self.bottom_repr, horizontal, self.horizontal_repr)
        y = self.nonl1(y)
        y = self.conv1(y)
        y = self.nonl2(y)
        y = self.conv2(y)

        return y


    def __repr__(self):
        fmt = 'UnetUpBlock ({}) {} x {} -> {}.'
        msg = fmt.format(self.name, self.bottom_repr, self.horizontal_repr, self.out_repr)
        return msg


aunet_default_down = [
    (1, ),
    (8, 6, 6),
    (12, 4, 2),
    (8, 2, 2)
]

@localized_module
class HUnet(nn.Module):
    def __init__(self, down=aunet_default_down,
                 mid=None, up=None):
        super(HUnet, self).__init__()

        if (mid is None) != (up is None):
            raise ValueError("You must either specify both `mid` and `up` or none of them")

        if mid is None and up is None:
            mid = down[-1]
            up = down[-2::-1]
            down = down[:-1]

        self.down = down
        self.mid = mid
        self.up = up

        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.down[:-1],
                                              self.down[1:])):

            # do not pass the very input of SE3Unet through nonlinearity
            first_nonl = i != 0

            block = UnetDownBlock(
                d_in, d_out, name='down_{}'.format(i), first_nonl=first_nonl,
            )
            self.path_down.append(block)

        self.middle_block = UnetMiddleBlock(
            self.down[-1], self.mid, self.up[0], name='middle'
        )

        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(self.up,
                                                      self.down[::-1],
                                                      self.up[1:])):

            block = UnetUpBlock(
                d_bot, d_hor, d_out, name='up_{}'.format(i),
            )
            self.path_up.append(block)


    def __repr__(self):
        fmt = 'HUnet, layout {},\npath_down:' \
        '[\n\t{}\n],\nmiddle:\t{}\npath_up:[\n\t{}\n]'
        pd = '\n\t'.join(repr(l) for l in self.path_down)
        pu = '\n\t'.join(repr(l) for l in self.path_up)
        msg = fmt.format(self.down + [self.mid] + self.up, pd, repr(self.middle_block), pu)
        return msg


    @dimchecked
    def forward(self, inp: ['b', 'fi', 'hi', 'wi']) -> ['b', 'fo', 'ho', 'wo']:
        features = harmonic.cmplx.from_real(inp)
        features_down = []
        for layer in self.path_down:
            features = layer(features)
            features_down.append(features)

            if not size_is_pow2(features):
                fmt = "Trying to downsample feature map of size {}"
                msg = fmt.format(features.size())
                raise RuntimeError(msg)
            features = d2.avg_pool2d(features, 2)
            
        features = self.middle_block(features)

        for layer, features_horizontal in zip(self.path_up, features_down[-1::-1]):
            features = d2.upsample_2d(features, scale_factor=2)
            features = layer(features, features_horizontal)

        return features[0, ...]

    def l2_params(self):
        return [p for n, p in self.named_parameters() if 'bias' not in n]

    def nr_params(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]
