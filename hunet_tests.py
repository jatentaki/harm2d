import unittest, torch

from harmonic.cmplx import from_real

from hunet import HUnet, UnetMiddleBlock, UnetDownBlock, UnetUpBlock

def rot90(x, k=1, plane=(2, 3)):
    if k == 0:
        return x
    elif k == 1:
        return x.flip(plane[0]).transpose(*plane)
    elif k == 2:
        return x.flip(plane[0]).flip(plane[1])
    elif k == 3:
        return x.flip(plane[1]).transpose(*plane)
    else:
        raise ValueError("k={} is invalid".format(k))

class EquivarianceTests(unittest.TestCase):
    def _diff_rotation(self, net, inp, plane):
        rotation = lambda t: rot90(t, plane=plane)

        rot = rotation(inp)

        base_fwd = net(inp)
        rot_fwd = net(rot)

        return (rotation(base_fwd) - rot_fwd).max().item()

    def _test_equivariance(self, net, inp, real=False):
        plane = (3, 4) if real else (2, 3)
        diff = self._diff_rotation(net, inp, plane=plane)
        self.assertLess(diff, 1e-5)

    def _test_equivariant_output(self, net_builder, train=True, real=False, ins=1):
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True

        torch.set_default_dtype(torch.float64)

        SEED = 0
        torch.manual_seed(SEED)

        net = net_builder().cuda()
        if train:
            net.train()
        else:
            net.eval()

        s = 92
        input = torch.randn(2, ins, s, s).cuda()

        if real:
            input = from_real(input)

        with torch.no_grad():
            self._test_equivariance(net, input, real=real)


class DownBlockTests(EquivarianceTests):
    def _build_block(self):
        class TwoBlocks(torch.nn.Module):
            def __init__(self):
                super(TwoBlocks, self).__init__()

                self.b1 = UnetDownBlock((1, ), (2, 3))
                self.b2 = UnetDownBlock((2, 3), (1, ))

            def forward(self, x):
                return self.b2(self.b1(x))

        return TwoBlocks()

    def test_equivariance_down_block(self):
        self._test_equivariant_output(self._build_block, real=True)

    def test_equivariance_down_block_eval(self):
        self._test_equivariant_output(self._build_block, train=False, real=True)

class UpBlockTests(EquivarianceTests):
    def _build_block(self):
        class ThreeBlocks(torch.nn.Module):
            def __init__(self):
                super(ThreeBlocks, self).__init__()

                bottom = (4, 5, 6)
                horizontal = (2, 3)

                self.block_bottom = UnetDownBlock((1, ), bottom)
                self.block_horizontal = UnetDownBlock((1, ), horizontal)

                self.up = UnetUpBlock(bottom, horizontal, (1, ))

            def forward(self, x):
                b = self.block_bottom(x)
                h = self.block_horizontal(x)

                return self.up(b, h)

        return ThreeBlocks()

    def test_equivariance_up_block(self):
        self._test_equivariant_output(self._build_block, real=True)

    def test_equivariance_up_block_eval(self):
        self._test_equivariant_output(self._build_block, train=False, real=True)


class MiddleBlockTests(EquivarianceTests):
    def test_equivariance_middle_block(self):
        builder = lambda : UnetMiddleBlock((1,), (2, 5, 3), (1,))
        self._test_equivariant_output(builder, real=True)

    def test_equivariance_middle_block_eval(self):
        builder = lambda : UnetMiddleBlock((1,), (2, 5, 3), (1,))
        self._test_equivariant_output(builder, train=False, real=True)
    
class AsymmMiddleBlockTests(EquivarianceTests):
    def _build_block(self):
        class ThreeBlocks(torch.nn.Module):
            def __init__(self):
                super(ThreeBlocks, self).__init__()

                a = (2, 3)
                b = (5, 2)
                c = (3, 3)

                self.d1 = UnetDownBlock((1, ), a)
                self.mid = UnetMiddleBlock(a, b, c)
                self.d2 = UnetDownBlock(c, (1, ))

            def forward(self, x):
                return self.d2(self.mid(self.d1(x)))

        return ThreeBlocks()

    def test_equivariance_middle_block(self):
        self._test_equivariant_output(self._build_block, real=True)
 

class HunetTests(EquivarianceTests):
    def test_equivariance_unet(self):
        builder = lambda : HUnet()
        self._test_equivariant_output(builder)

    def test_equivariance_unet_asymm(self):
        down = [(1,), (4, 2, 2), (3, 3, 1)]
        mid = (2, 3)
        up = [(3, 2, 1), (4, 2, 2), (1, )]

        builder = lambda : HUnet(down=down, mid=mid, up=up)
        self._test_equivariant_output(builder)

    def test_equivariance_unet_asymm_inp(self):
        down = [(3,), (4, 2, 2), (3, 3, 1)]
        mid = (2, 3)
        up = [(3, 3, 1), (4, 2, 2), (1, )]

        builder = lambda : HUnet(down=down, mid=mid, up=up)
        self._test_equivariant_output(builder, ins=3)

unittest.main()
