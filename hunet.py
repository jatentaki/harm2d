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
    def __init__(self, in_repr, out_repr, size=5):
        super(UnetDownBlock, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        
        self.conv1 = BNConv2d(in_repr, out_repr, size)
        self.nonl1 = d2.ScalarGate2d(out_repr)

        self.conv2 = BNConv2d(out_repr, out_repr, size)
        self.nonl2 = d2.ScalarGate2d(out_repr)
    

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hi', 'wi']
               )     -> ([2, 'b', 'fo', 'ho', 'wo'],
                         [2, 'b', 'fo', 'ho', 'wo']):

        y = self.conv1(x)
        y = self.nonl1(y)
        y = self.conv2(y)
        y_gated = self.nonl2(y)

        return y_gated, y


    def __repr__(self):
        fmt = 'UnetDownBlock ({}) {} -> {}.'
        msg = fmt.format(self.name, self.in_repr, self.out_repr)
        return msg


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
        fmt = 'UnetUpBlock ({}) bot {} x hor {} -> {}.'
        msg = fmt.format(self.name, self.bottom_repr, self.horizontal_repr, self.out_repr)
        return msg


aunet_default_down = [
    (1, ),
    (8, 6, 6), # 20
    (12, 4, 2), # 18
    (8, 2, 2) # 12
]

@localized_module
class HUnet(nn.Module):
    def __init__(self, down=aunet_default_down, up=None, classes=1):
        super(HUnet, self).__init__()

        if up is None:
            up = down[-2::-1]

        self.classes = classes
        self.down = down
        self.up = up

        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.down[:-1], self.down[1:])):
            block = UnetDownBlock(d_in, d_out, name='down_{}'.format(i))
            self.path_down.append(block)

        bottoms = [down[-1]] + self.up
        horizontals = self.down[-2::-1]
        outs = self.up

        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bottoms,
                                                      horizontals,
                                                      outs)):
            block = UnetUpBlock(
                d_bot, d_hor, d_out, name='up_{}'.format(i),
            )
            self.path_up.append(block)

        self.logit_nonl = d2.ScalarGate2d(up[-2])
        self.logit_conv = nn.Conv2d(sum(up[-2]), self.classes, 1)


    def __repr__(self):
        fmt = ('HUnet:\n'
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
        msg = fmt.format(pd, pu, self.up[-1], self.classes)
        return msg


    @dimchecked
    def forward(self, inp: ['b', 'fi', 'hi', 'wi']) -> ['b', 'fo', 'ho', 'wo']:
        features_gated = harmonic.cmplx.from_real(inp)
        features_down = []
        for i, layer in enumerate(self.path_down):
            if i != 0:
                if not size_is_pow2(features_gated):
                    fmt = "Trying to downsample feature map of size {}"
                    msg = fmt.format(features_gated.size())
                    raise RuntimeError(msg)

                features_gated = d2.avg_pool2d(features_gated, 2)

            features_gated, features  = layer(features_gated)
            features_down.append(features)

        f_b = features_down[-1]
        features_horizontal = features_down[-2::-1]

        for layer, f_h in zip(self.path_up, features_horizontal):
            f_b = d2.upsample_2d(f_b, scale_factor=2)
            f_b = layer(f_b, f_h)

        f_gated = self.logit_nonl(f_b)
        return self.logit_conv(harmonic.cmplx.magnitude(f_gated))

    def l2_params(self):
        return [p for n, p in self.named_parameters() if 'bias' not in n]

    def nr_params(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]
