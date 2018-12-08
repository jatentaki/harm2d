import unittest, torch

from harmonic.cmplx import from_real, magnitude

from hunet import HUnet, Upsample, Downsample

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

    def _test_equivariant_output(self, net_builder, train=True, real=False, ins=1, s=92):
        cuda = torch.cuda.is_available()

        if cuda:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True

        torch.set_default_dtype(torch.float64)

        SEED = 0
        torch.manual_seed(SEED)

        net = net_builder()
        if cuda:
            net = net.cuda()

        if train:
            net.train()
        else:
            net.eval()

        input = torch.randn(2, ins, s, s)
        if cuda:
            input = input.cuda()

        if real:
            input = from_real(input)

        with torch.no_grad():
            self._test_equivariance(net, input, real=real)


class HunetTests(EquivarianceTests):
    def test_equivariance_unet_symm(self):
        down = [(2, 2, 2), (2, 2, 2), (2, 2)]
        up = [(2, 2, 2), (2, 2, 2)]

        builder = lambda : HUnet(down=down, up=up, size=5, radius=2)
        self._test_equivariant_output(builder)

    def test_equivariance_unet_asymm(self):
        down = [(4, 2, 2), (3, 3, 1), (2, 3)]
        up = [(3, 2, 1), (8, 3, 2)]

        builder = lambda : HUnet(down=down, up=up, size=5, radius=2)
        self._test_equivariant_output(builder)

    def test_equivariance_unet_asymm_3_input_channels(self):
        down = [(4, 2, 2), (3, 3, 1), (2, 3)]
        up = [(3, 3, 1), (8, 3, 2)]

        builder = lambda : HUnet(in_features=3, down=down, up=up)
        self._test_equivariant_output(builder, ins=3)


unittest.main()
