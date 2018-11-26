import torch
import torch.nn as nn

from torch_dimcheck import dimchecked
from harmonic.cmplx import magnitude

class _GlobalGate(nn.Module):
    def __init__(self, repr, dim=2):
        super(_GlobalGate, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim value of {} is not allowed".format(dim))

        self.repr = repr
        self._dim = dim

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        total_fmaps = sum(repr)
        self.mean_mat = nn.Linear(total_fmaps, 2 * total_fmaps)
        self.conv1 = conv(total_fmaps, 2 * total_fmaps, 1)
        self.conv2 = conv(2 * total_fmaps, total_fmaps, 1)

    @dimchecked
    def forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        magnitudes = magnitude(x)
        g_x = self.conv1(magnitudes)
        
        means = x.reshape(*x.shape[:3], -1).mean(dim=3)
        mean_magnitudes = magnitude(means)
        g_m = self.mean_mat(mean_magnitudes)

        g = g_x + g_m.reshape(*g_m.shape, *([1] * self._dim))
        g = torch.relu(g)
        g = self.conv2(g)
        g = torch.sigmoid(g)

        return x * g.unsqueeze(0)

    def __repr__(self):
        return f'GlobalGate{self._dim}d(repr={self.repr})'


class GlobalGate2d(_GlobalGate):
    def __init__(self, repr):
        super(GlobalGate2d, self).__init__(repr, dim=2)

    @dimchecked
    def forward(self, x: [2, 'b', 'f', 'h', 'w']) -> [2, 'b', 'f', 'h', 'w']:
        return super(GlobalGate2d, self).forward(x)

if __name__ == '__main__':
    import unittest

    class GlobalGateTests(unittest.TestCase):
        def test_forward(self):
            nonl = GlobalGate2d((3, 6, 0, 1))
            n, h, w = 3, 40, 40
            inputs = torch.randn(2, n, 3 + 6 + 1, h, w)
            output = nonl(inputs)


    unittest.main()
