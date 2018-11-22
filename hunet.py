import torch, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import harmonic
from harmonic import d2

from torch_localize import localized_module, localized
from torch_dimcheck import dimchecked

from utils import cut_to_match

SAVE_NPY = False

def size_is_pow2(t):
    ''' Check if the trailing spatial dimensions are powers of 2 '''
    return all(s % 2 == 0 for s in t.size()[-2:])


class BNConv2d(nn.Module):
    def __init__(self, repr_in, repr_out, size, radius=None):
        super(BNConv2d, self).__init__()

        self.bn = d2.BatchNorm2d(repr_in)
        self.conv = d2.HConv2d(repr_in, repr_out, size, radius=radius)

    def forward(self, x):
        y = self.bn(x)
        y = self.conv(y)

        return y

class UnetDownBlock(nn.Module):
    def __init__(self, in_repr, out_repr, size=5, radius=None, name=None):
        super(UnetDownBlock, self).__init__()

        self.name = name
        self.in_repr = in_repr
        self.out_repr = out_repr
        
        self.conv1 = BNConv2d(in_repr, out_repr, size, radius=radius)
        self.nonl1 = d2.ScalarGate2d(out_repr, name=self.name + '_gate1')

        self.conv2 = BNConv2d(out_repr, out_repr, size, radius=radius)
        self.nonl2 = d2.ScalarGate2d(out_repr, name=self.name + '_gate2')
    

    @localized
    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hi', 'wi']
               )     -> ([2, 'b', 'fo', 'ho', 'wo'],
                         [2, 'b', 'fo', 'ho', 'wo']):

        y = self.conv1(x)
        y = self.nonl1(y)
        y = self.conv2(y)
        y_gated = self.nonl2(y)

        if SAVE_NPY:
            np.save('fmap_' + self.name + '.npy', y.detach().cpu().numpy())
            np.save(
                'kernels_' + self.name + '.npy',
                self.conv2.conv.synthesize().detach().cpu().numpy()
            )

        return y_gated, y


class UnetUpBlock(nn.Module):
    def __init__(self, bottom_repr, horizontal_repr, out_repr,
                 size=5, radius=None, name=None):

        super(UnetUpBlock, self).__init__()

        self.name = name
        self.bottom_repr = bottom_repr
        self.horizontal_repr = horizontal_repr
        self.cat_repr = harmonic.cat_repr(bottom_repr, horizontal_repr)
        self.out_repr = out_repr

        self.nonl1 = d2.ScalarGate2d(self.cat_repr, name=self.name + '_gate1')
        self.conv1 = BNConv2d(self.cat_repr, self.cat_repr, size, radius=radius)

        self.nonl2 = d2.ScalarGate2d(self.cat_repr, name=self.name + '_gate2')
        self.conv2 = BNConv2d(self.cat_repr, self.out_repr, size, radius=radius)


    @localized
    @dimchecked
    def forward(self, bottom:     [2, 'b', 'fb', 'hb', 'wb'],
                      horizontal: [2, 'b', 'fh', 'hh', 'wh']
               )               -> [2, 'b', 'fo', 'ho', 'wo']:

        horizontal = cut_to_match(bottom, horizontal, n_pref=3)
        y = d2.cat2d(bottom, self.bottom_repr, horizontal, self.horizontal_repr)
        y = self.nonl1(y)
        y = self.conv1(y)
        if SAVE_NPY:
            np.save('fmap_pre_' + self.name + '.npy', y.detach().cpu().numpy())
        y = self.nonl2(y)
        if SAVE_NPY:
            np.save('fmap_post_' + self.name + '.npy', y.detach().cpu().numpy())
        y = self.conv2(y)


        return y


hunet_default_down = [
    (8, 6, 6),
    (12, 4, 2),
    (8, 2, 2)
]
hunet_default_up= [
    (12, 4, 2),
    (20, )
]

@localized_module
class HUnet(nn.Module):
    def __init__(self, in_features=1, out_features=1, up=hunet_default_up,
                 down=hunet_default_down, size=5, radius=None):
        super(HUnet, self).__init__()

        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.up = up
        self.down = down
        self.in_features = in_features
        self.out_features = out_features

        down_dims = [(in_features, )] + down
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            block = UnetDownBlock(
                d_in, d_out, size=size, radius=radius, name='down_{}'.format(i)
            )
            self.path_down.append(block)

        bot_dims = [down[-1]] + up
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            block = UnetUpBlock(
                d_bot, d_hor, d_out, size=size, radius=radius,
                name='up_{}'.format(i)
            )
            self.path_up.append(block)

        self.logit_nonl = d2.ScalarGate2d(up[-1], name='logit_nonl')
        self.logit_conv = nn.Conv2d(sum(up[-1]), self.out_features, 1)

        self.n_params = 0
        for param in self.parameters():
            self.n_params += param.numel()


    @dimchecked
    def forward(self, inp: ['b', 'fi', 'hi', 'wi']) -> ['b', 'fo', 'ho', 'wo']:
        if inp.size(1) != self.in_features:
            fmt = "Expected {} feature channels in input, got {}"
            msg = fmt.format(self.in_features, inp.size(1))
            raise ValueError(msg)

        if SAVE_NPY:
            np.save('img.npy', inp.detach().cpu().numpy())

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
        magnitudes = harmonic.cmplx.magnitude(f_gated)
        return self.logit_conv(magnitudes)

    def l2_params(self):
        return [p for n, p in self.named_parameters() if 'bias' not in n]

    def nr_params(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]
