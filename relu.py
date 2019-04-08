import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from torch_dimcheck import dimchecked
from harmonic.cmplx import magnitude

class CReLU(nn.Module):
    def __init__(self, repr, dim=2, eps=1e-3):
        super(CReLU, self).__init__()
        
        self.repr = repr
        self.eps = eps
        self.dim = 2

        bias = torch.zeros(1, 1, sum(repr), *(self.dim * [1]))
        bias.exponential_(1.17) # exponential so that bias is always positive
        self.bias = nn.Parameter(bias, requires_grad=True)

    @dimchecked
    def forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        return checkpoint(self._forward, x)
        
    @dimchecked
    def _forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        magn = magnitude(x).unsqueeze(0)
        normalized = x / (magn + self.eps)
        magn_biased = magn - self.bias

        zero = x.new_zeros((1, ))
        return torch.where(magn_biased > 0, normalized * magn_biased, zero)

    def __repr__(self):
        return f'CReLU{self.dim}d(repr={self.repr})'

if __name__ == '__main__':
    import unittest

    class CReLUTests(unittest.TestCase):
        def test_forward(self):
            nonl = CReLU((3, 6, 0, 1))
            n, h, w = 3, 40, 40
            inputs = torch.randn(2, n, 3 + 6 + 1, h, w)
            output = nonl(inputs)

            print(nonl.bias[0, 0, 0])
            print(magnitude(inputs[:, 0, 0]))
            print(output[0, 0, 0])


    unittest.main()

#    n = 10000
#    samples = torch.randn(2, n)
#    mags = magnitude(samples)
#
#    thresholds = torch.zeros(n)
#    lambda_ = torch.sqrt(-2. * torch.log(torch.tensor(0.5)))
#    thresholds.exponential_(lambda_)
#    passed = mags > thresholds
#
#    print(lambda_)
#    print(passed.sum().float() / n)
