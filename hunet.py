import torch, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import harmonic
from harmonic import d2

from torch_localize import localized_module, localized
from torch_dimcheck import dimchecked

from utils import cut_to_match


def size_is_pow2(t):
    ''' Check if the trailing spatial dimensions are powers of 2 '''
    return all(s % 2 == 0 for s in t.size()[-2:])


class TrivialUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialUpsample, self).__init__()

    def forward(self, x):
        return d2.upsample_2d(x, scale_factor=2)


class TrivialDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialDownsample, self).__init__()

    def forward(self, x):
        return d2.avg_pool2d(x, 2)

class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x

default_setup = {
    'gate': d2.ScalarGate2d,
    'norm': d2.InstanceNorm2d,
    'upsample': TrivialUpsample,
    'downsample': TrivialDownsample,
    'dropout': NoOp,
    'padding': False
}


class Conv(nn.Sequential):
    def __init__(self, repr_in, repr_out, size, radius=None, setup=default_setup):
        norm = setup['norm'](repr_in)
        nonl = setup['gate'](repr_in)
        dropout = setup['dropout']()

        conv_kwargs = {}
        if setup['padding']:
            conv_kwargs['padding'] = size // 2
        else:
            conv_kwargs['padding'] = 0

        conv = d2.HConv2d(
            repr_in, repr_out, size, radius=radius, conv_kwargs=conv_kwargs
        )

        super(Conv, self).__init__(norm, nonl, dropout, conv)


class Upsample(nn.Sequential):
    def __init__(self, repr, size, radius=None, setup=default_setup):
        conv_kwargs = {
            'stride': 2,
            'output_padding': 1
        }
        norm = setup['norm'](repr)
        nonl = setup['gate'](repr)
        conv = d2.HConv2dTranspose(
            repr, repr, size, radius=radius, conv_kwargs=conv_kwargs
        )

        super(Upsample, self).__init__(norm, nonl, conv)


class Downsample(nn.Sequential):
    def __init__(self, repr, size, radius=None, setup=default_setup):

        conv_kwargs = {
            'stride': 2,
            'padding': size // 2
        }
        norm = setup['norm'](repr)
        nonl = setup['gate'](repr)
        conv = d2.HConv2d(
            repr, repr, size, radius=radius, conv_kwargs=conv_kwargs
        )

        super(Downsample, self).__init__(norm, nonl, conv)

class FirstDownBlock(nn.Sequential):
    def __init__(self, in_repr, out_repr, size=5, radius=None, name=None,
                 setup=default_setup):

        self.name = name
        self.in_repr = in_repr
        self.out_repr = out_repr

        conv_kwargs = {}
        if setup['padding']:
            conv_kwargs['padding'] = size // 2
        else:
            conv_kwargs['padding'] = 0

        conv1 = d2.HConv2d(
            in_repr, out_repr, size=size, radius=radius, conv_kwargs=conv_kwargs
        )
        conv2 = Conv(out_repr, out_repr, size, radius=radius, setup=setup)
 
        super(FirstDownBlock, self).__init__(conv1, conv2)

class UnetDownBlock(nn.Sequential):
    def __init__(self, in_repr, out_repr, size=5, radius=None, name=None,
                 setup=default_setup):

        self.name = name
        self.in_repr = in_repr
        self.out_repr = out_repr
        
        downsample = setup['downsample'](in_repr, size, radius=radius, setup=setup)
        conv1 = Conv(in_repr, out_repr, size, radius=radius, setup=setup)
        conv2 = Conv(out_repr, out_repr, size, radius=radius, setup=setup)

        super(UnetDownBlock, self).__init__(downsample, conv1, conv2)


    @localized
    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hi', 'wi']
               )     ->  [2, 'b', 'fo', 'ho', 'wo']:

        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return super(UnetDownBlock, self).forward(x)


class UnetUpBlock(nn.Module):
    def __init__(self, bottom_repr, horizontal_repr, out_repr,
                 size=5, radius=None, name=None, setup=default_setup):

        super(UnetUpBlock, self).__init__()

        self.name = name
        self.bottom_repr = bottom_repr
        self.horizontal_repr = horizontal_repr
        self.cat_repr = harmonic.cat_repr(bottom_repr, horizontal_repr)
        self.out_repr = out_repr

        self.upsample = setup['upsample'](
            bottom_repr, size, radius=radius, setup=setup
        )

        conv1 = Conv(
            self.cat_repr, self.cat_repr, size, radius=radius, setup=setup
        )

        conv2 = Conv(
            self.cat_repr, self.out_repr, size, radius=radius, setup=setup,
        )

        self.seq = nn.Sequential(conv1, conv2)


    @localized
    @dimchecked
    def forward(self, bot: [2, 'b', 'fb', 'hb', 'wb'],
                      hor: [2, 'b', 'fh', 'hh', 'wh']
               )        -> [2, 'b', 'fo', 'ho', 'wo']:


        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=3)
        combined = d2.cat2d(bot_big, self.bottom_repr, hor, self.horizontal_repr)

        return self.seq(combined)


@localized_module
class HUnet(nn.Module):
    def __init__(self, in_features=1, out_features=1, up=None, down=None,
                 size=5, radius=None, setup=default_setup):

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
            if i == 0:
                block_type = FirstDownBlock
            else:
                block_type = UnetDownBlock

            block = block_type(
                d_in, d_out, size=size, radius=radius, name=f'down_{i}', setup=setup
            )
            self.path_down.append(block)

        bot_dims = [down[-1]] + up
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            block = UnetUpBlock(
                d_bot, d_hor, d_out, size=size, radius=radius,
                name=f'up_{i}', setup=setup
            )
            self.path_up.append(block)

        self.logit_nonl = setup['gate'](up[-1])
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

        features = [harmonic.cmplx.from_real(inp)]
        for i, layer in enumerate(self.path_down):
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        f_gated = self.logit_nonl(f_bot)
        magnitudes = harmonic.cmplx.magnitude(f_gated)
        return self.logit_conv(magnitudes)

    @staticmethod
    def is_regularized(param_name):
        return not ('bias' in param_name or 'angular' in param_name)

    def l2_params(self):
        return [p for n, p in self.named_parameters() \
                if HUnet.is_regularized(n)]

    def nr_params(self):
        return [p for n, p in self.named_parameters() \
                if not HUnet.is_regularized(n)]
