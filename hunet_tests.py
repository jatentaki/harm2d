import unittest, torch

from harmonic.cmplx import from_real

from hunet import HUnet, UnetMiddleBlock

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

class Tests(unittest.TestCase):
    def _diff_rotation(self, net, inp, plane):
        rotation = lambda t: rot90(t, plane=plane)

        rot = rotation(inp)

        base_fwd = net(inp)
        rot_fwd = net(rot)

        return (rotation(base_fwd) - rot_fwd).max().item()

    def _test_selective_equivariance(self, net, inp):
        diff = self._diff_rotation(net, inp, plane=(3, 4))
        self.assertLess(diff, 1e-5)

        diff = self._diff_rotation(net, inp, plane=(2, 4))
        self.assertGreater(diff, 1)

        diff = self._diff_rotation(net, inp, plane=(2, 3))
        self.assertGreater(diff, 1)

    def _test_equivariant_output(self, net_builder, train=True, real=False):
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
        input = torch.randn(2, 1, s, s, s).cuda()

        if real:
            input = from_real(input)

        with torch.no_grad():
            self._test_selective_equivariance(net, input)

    def test_equivariance_middle_block(self):
        builder = lambda : UnetMiddleBlock((1,), (2, 5, 3), (1,))
        self._test_equivariant_output(builder, real=True)

    def test_equivariance_unet(self):
        builder = lambda : HUnet()
        self._test_equivariant_output(builder)

    def test_equivariance_middle_block_eval(self):
        builder = lambda : UnetMiddleBlock((1,), (2, 5, 3), (1,))
        self._test_equivariant_output(builder, train=False, real=True)
    
    def test_equivariance_unet_eval(self):
        builder = lambda : HUnet()
        self._test_equivariant_output(builder, train=False)

    def test_equivariance_unet_asymm(self):
        down = [(1,), (4, 2, 2), (3, 3, 1)]
        mid = (2,)
        up = [(2, 3, 2), (2, 4, 3), (1,)]

        builder = lambda : HUnet(down=down, mid=mid, up=up)
        self._test_equivariant_output(builder)

unittest.main()
