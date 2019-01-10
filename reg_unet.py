import torch, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        r = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False
        )
        return r


class TrivialDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialDownsample, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, 2)


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class ScalarGate(nn.Module):
    def __init__(self, n_features, mult=1):
        super(ScalarGate, self).__init__()
        self.n_features = n_features
        self.mult = mult
    
        if mult == 1:
            self.seq = nn.Sequential(
                nn.Conv2d(self.n_features, self.n_features, 1),
                nn.Sigmoid()
            )
        elif mult == 2:
            self.seq = nn.Sequential(
                nn.Conv2d(self.n_features, self.n_features, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.n_features, self.n_features, 1),
                nn.Sigmoid()
            )

    def forward(self, inp):
        g = self.seq(inp)

        return g * inp


default_setup = {
    'gate': ScalarGate,
    'norm': nn.InstanceNorm2d,
    'upsample': TrivialUpsample,
    'downsample': TrivialDownsample,
    'dropout': NoOp,
}


class Conv(nn.Sequential):
    def __init__(self, in_, out_, size, setup=default_setup):
        norm = setup['norm'](in_)
        nonl = setup['gate'](in_)
        dropout = setup['dropout']()
        conv = nn.Conv2d(in_, out_, size)

        super(Conv, self).__init__(norm, nonl, dropout, conv)


class Upsample(nn.Sequential):
    def __init__(self, n_features, size, setup=default_setup):
        conv_kwargs = {
            'stride': 2,
            'output_padding': 1
        }
        norm = setup['norm'](n_features)
        nonl = setup['gate'](n_features)
        conv = nn.ConvTranspose2d(
            n_features, n_features, size, conv_kwargs=conv_kwargs
        )

        super(Upsample, self).__init__(norm, nonl, conv)


class Downsample(nn.Sequential):
    def __init__(self, n_features, size, setup=default_setup):

        conv_kwargs = {
            'stride': 2,
            'padding': size // 2
        }
        norm = setup['norm'](n_features)
        nonl = setup['gate'](n_features)
        conv = nn.Conv2d(
            n_features, n_features, size, conv_kwargs=conv_kwargs
        )

        super(Downsample, self).__init__(norm, nonl, conv)


class FirstDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, name=None,
                 setup=default_setup):

        self.name = name
        self.in_ = in_
        self.out_ = out_

        conv1 = nn.Conv2d(in_, out_, size)
        conv2 = Conv(out_, out_, size, setup=setup)
 
        super(FirstDownBlock, self).__init__(conv1, conv2)


class UnetDownBlock(nn.Sequential):
    def __init__(self, in_, out_, size=5, name=None,
                 setup=default_setup):

        self.name = name
        self.in_ = in_
        self.out_ = out_
        
        downsample = setup['downsample'](in_, size, setup=setup)
        conv1 = Conv(in_, out_, size, setup=setup)
        conv2 = Conv(out_, out_, size, setup=setup)

        super(UnetDownBlock, self).__init__(downsample, conv1, conv2)


    @localized
    @dimchecked
    def forward(self, x: ['b', 'fi', 'hi', 'wi']
               )     ->  ['b', 'fo', 'ho', 'wo']:

        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return super(UnetDownBlock, self).forward(x)


class UnetUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_,
                 size=5, name=None, setup=default_setup):

        super(UnetUpBlock, self).__init__()

        self.name = name
        self.bottom_ = bottom_
        self.horizontal_ = horizontal_
        self.cat_ = bottom_ + horizontal_
        self.out_ = out_

        self.upsample = setup['upsample'](bottom_, size, setup=setup)

        conv1 = Conv(self.cat_, self.cat_, size, setup=setup)
        conv2 = Conv(self.cat_, self.out_, size, setup=setup,)
        self.seq = nn.Sequential(conv1, conv2)


    @localized
    @dimchecked
    def forward(self, bot: ['b', 'fb', 'hb', 'wb'],
                      hor: ['b', 'fh', 'hh', 'wh']
               )        -> ['b', 'fo', 'ho', 'wo']:


        bot_big = self.upsample(bot)
        hor = cut_to_match(bot_big, hor, n_pref=2)
        combined = torch.cat([bot_big, hor], dim=1)

        return self.seq(combined)


@localized_module
class Unet(nn.Module):
    def __init__(self, in_features=1, out_features=1, up=None, down=None,
                 size=5, setup=default_setup):

        super(Unet, self).__init__()

        if not len(down) == len(up) + 1:
            raise ValueError("`down` must be 1 item longer than `up`")

        self.up = up
        self.down = down
        self.in_features = in_features
        self.out_features = out_features

        down_dims = [in_features] + down
        self.path_down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            if i == 0:
                block_type = FirstDownBlock
            else:
                block_type = UnetDownBlock

            block = block_type(
                d_in, d_out, size=size, name=f'down_{i}', setup=setup
            )
            self.path_down.append(block)

        bot_dims = [down[-1]] + up
        hor_dims = down_dims[-2::-1]
        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(bot_dims, hor_dims, up)):
            block = UnetUpBlock(
                d_bot, d_hor, d_out, size=size, name=f'up_{i}', setup=setup
            )
            self.path_up.append(block)

        self.logit_nonl = setup['gate'](up[-1])
        self.logit_conv = nn.Conv2d(up[-1], self.out_features, 1)

        self.n_params = 0
        for param in self.parameters():
            self.n_params += param.numel()


    @dimchecked
    def forward(self, inp: ['b', 'fi', 'hi', 'wi']) -> ['b', 'fo', 'ho', 'wo']:
        if inp.size(1) != self.in_features:
            fmt = "Expected {} feature channels in input, got {}"
            msg = fmt.format(self.in_features, inp.size(1))
            raise ValueError(msg)

        features = [inp]
        for i, layer in enumerate(self.path_down):
            features.append(layer(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]

        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        f_gated = self.logit_nonl(f_bot)
        return self.logit_conv(f_gated)

    @staticmethod
    def is_regularized(param_name):
        return not ('bias' in param_name or 'angular' in param_name)

    def l2_params(self):
        return [p for n, p in self.named_parameters() \
                if HUnet.is_regularized(n)]

    def nr_params(self):
        return [p for n, p in self.named_parameters() \
                if not HUnet.is_regularized(n)]

if __name__ == '__main__':
    import unittest
    
    class Tests(unittest.TestCase):
        def test_inequal_output_asymmetric(self):
            unet = Unet(in_features=3, out_features=4, down=[16, 32, 64], up=[40, 24])
            input = torch.zeros(2, 3, 104, 104)
            output = unet(input)
            self.assertEqual(torch.Size([2, 4, 24, 24]), output.size())
    
        def test_inequal_output_symmetric(self):
            unet = Unet(down=[16, 32, 64], up=[40, 24])
            input = torch.zeros(2, 1, 104, 104)
            output = unet(input)
            self.assertEqual(torch.Size([2, 1, 24, 24]), output.size())

    unittest.main()
