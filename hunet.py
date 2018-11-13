import torch, itertools
import torch.nn as nn
import torch.nn.functional as F

from harmonic.d3 import HConv3d, BatchNorm3d, ScalarGate3d, avg_pool3d, upsample_3d
from harmonic.cmplx import from_real

from torch_localize import localized_module
from torch_dimcheck import dimchecked

from utils import cut_to_match

def concatenated_representation(r1, r2):
    '''
    calculate the representation signature of concatenation of signatures
    `r1` and `r2`. For example,
    `concatenated_representation((3, 5, 1,), (2, 2,)) == (5, 7, 1,)`
    '''

    return tuple(x + y for x, y in itertools.zip_longest(r1, r2, fillvalue=0))

def size_is_pow2(t):
    ''' Check if the trailing spatial dimensions are powers of 2 '''
    return all(s % 2 == 0 for s in t.size()[2:])

@dimchecked
def merge_representations(
    t1: ['b', 'f1', 'h', 'w', 'd', 2], repr1,
    t2: ['b', 'f2', 'h', 'w', 'd', 2], repr2) -> ['b', 'fo', 'h', 'w', 'd', 2]:
    '''
    concatenate tensor `t1`, containing representations `repr1` with tensor
    `t2` containing representations `repr2`. Returns tensor `t3` of representation
    `concatenated_representation(repr1, repr2)`
    '''

    fmt = "size of `t{}` at axis 1 ({}) does not match its representation ({}, total {})"
    if t1.size(1) != sum(repr1):
        msg = fmt.format(1, t1.size(1), repr1, sum(repr1))
        raise ValueError(msg)

    if t2.size(1) != sum(repr2):
        msg = fmt.format(2, t2.size(1), repr2, sum(repr2))
        raise ValueError(msg)

    blocks = []
    prev1, prev2 = 0, 0
    for i, (n1, n2) in enumerate(zip(repr1, repr1)):
        block1 = t1[:, prev1:prev1+n1, ...]
        block2 = t2[:, prev2:prev2+n2, ...]
        prev1 += n1
        prev2 += n2
        blocks.extend([block1, block2])

    # one of representations must have been exhausted
    if prev1 < t1.size(1) and prev2 < t2.size(1):
        msg = "logical error: neither iterator exhausted"
        raise AssertionError(msg)

    if prev1 < t1.size(1):
        blocks.append(t1[:, prev1:, ...])
    if prev2 < t2.size(1):
        blocks.append(t2[:, prev2:, ...])
    
    return torch.cat(blocks, dim=1)


def h3conv(repr_in, repr_out, size, **kwargs):
    return HConv3d(repr_in, repr_out, size, **kwargs)

@localized_module
class UnetDownBlock(nn.Module):
    def __init__(self, in_repr, out_repr, size=5, first_nonl=True, **kwargs):
        super(UnetDownBlock, self).__init__()

        self.first_nonl = first_nonl

        self.in_repr = in_repr
        self.out_repr = out_repr
        
        if first_nonl:
            self.nonl1 = ScalarGate3d(in_repr)
        self.conv1 = h3conv(in_repr, out_repr, size, **kwargs)

        self.nonl2 = ScalarGate3d(out_repr)
        self.conv2 = h3conv(out_repr, out_repr, size, **kwargs)
    

    @dimchecked
    def forward(self, y: ['b', 'fi', 'hi', 'wi', 'di', 2]
               )      -> ['b', 'fo', 'ho', 'wo', 'do', 2]:
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
    def __init__(self, in_repr, mid_repr, out_repr, size=5, **kwargs):
        super(UnetMiddleBlock, self).__init__()

        self.in_repr = in_repr
        self.mid_repr = mid_repr
        self.out_repr = out_repr

        self.nonl1 = ScalarGate3d(in_repr)
        self.conv1 = h3conv(in_repr, mid_repr, size=size, **kwargs)

        self.nonl2 = ScalarGate3d(mid_repr)
        self.conv2 = h3conv(mid_repr, out_repr, size=size, **kwargs)

    def __repr__(self):
        fmt = 'UnetMiddleBlock ({}) {} -> {} -> {}.'
        msg = fmt.format(self.name, self.in_repr, self.mid_repr, self.out_repr)
        return msg
        
    @dimchecked
    def forward(self, y: ['b', 'fi', 'hi', 'wi', 'di', 2]
               )      -> ['b', 'fo', 'ho', 'wo', 'do', 2]:

        y = self.nonl1(y)
        y = self.conv1(y)
        y = self.nonl2(y)
        y = self.conv2(y)

        return y


@localized_module
class UnetUpBlock(nn.Module):
    def __init__(self, bottom_repr, horizontal_repr, out_repr,
                 size=5, **kwargs):
        super(UnetUpBlock, self).__init__()

        self.bottom_repr = bottom_repr
        self.horizontal_repr = horizontal_repr
        self.cat_repr = concatenated_representation(bottom_repr, horizontal_repr)
        self.out_repr = out_repr

        self.nonl1 = ScalarGate3d(self.cat_repr)
        self.conv1 = h3conv(self.cat_repr, self.cat_repr, size, **kwargs)

        self.nonl2 = ScalarGate3d(self.cat_repr)
        self.conv2 = h3conv(self.cat_repr, self.out_repr, size, **kwargs)


    @dimchecked
    def forward(self, bottom:     ['b', 'fb', 'hb', 'wb', 'db', 2],
                      horizontal: ['b', 'fh', 'hh', 'wh', 'dh', 2]
               )               -> ['b', 'fo', 'ho', 'wo', 'do', 2]:

        horizontal = cut_to_match(bottom, horizontal)
        y = merge_representations(
            bottom, self.bottom_repr, horizontal, self.horizontal_repr
        )
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
                 mid=None, up=None, **kwargs):
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
                d_in, d_out, name='down_{}'.format(i),
                first_nonl=first_nonl, **kwargs
            )
            self.path_down.append(block)

        self.middle_block = UnetMiddleBlock(
            self.down[-1], self.mid, self.up[0],
            name='middle', **kwargs
        )

        self.path_up = nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(zip(self.up,
                                                      self.down[::-1],
                                                      self.up[1:])):

            block = UnetUpBlock(
                d_bot, d_hor, d_out, name='up_{}'.format(i),
                **kwargs
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
    def forward(self, inp: ['b', 'fi', 'hi', 'wi', 'di']) -> ['b', 'fo', 'ho', 'wo', 'do']:
        features = from_real(inp)
        features_down = []
        for layer in self.path_down:
            features = layer(features)
            features_down.append(features)

            if not size_is_pow2(features):
                fmt = "Trying to downsample feature map of size {}"
                msg = fmt.format(features.size())
                raise RuntimeError(msg)
            features = avg_pool3d(features, 2)
            
        features = self.middle_block(features)

        for layer, features_horizontal in zip(self.path_up, features_down[-1::-1]):
            features = upsample_3d(features, scale_factor=2)
            features = layer(features, features_horizontal)

        return features[..., 0]

    def l2_params(self):
        return [p for n, p in self.named_parameters() if 'bias' not in n]

    def nr_params(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]
