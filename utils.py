import time, functools, torch, os
import numpy as np

def upsample(inp, scale_factor=2, mode='bilinear', align_corners=False):
    if not inp.dim() == 4:
        fmt = 'Attempting to upscale a tensor of shape {}'
        msg = fmt.format(inp.size())
        raise ValueError(msg)

    return torch.nn.functional.interpolate(
        inp, scale_factor=scale_factor,
        mode=mode, align_corners=False
    )

def time_it(return_tuple=False):
    def timer(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            start = time.time()
            v = f(*args, **kwargs)
            duration = time.time() - start

            if return_tuple:
                return v, duration
            else:
                print('Call to {} took {:.2f}s'.format(f.__name__, duration))
                return v

        return wrapped
    return timer


class AvgMeter:
    def __init__(self):
        self.n = 0
        self.avg = 0
        self.last = None

    def update(self, val):
        self.last = val
        avg = self.avg * self.n + val
        self.n += 1
        self.avg = avg / self.n

    def reset(self):
        self.n = 0
        self.avg = 0
        self.last = None


def fmt_value(v, prec=3):
    ffmt = '{:.' + str(prec) + 'f}'
    if isinstance(v, float):
        return ffmt.format(v)
    elif isinstance(v, np.ndarray):
        assert len(v.shape) == 1, "cannot format higher dimensional arrays"
        val_fmt = ffmt if v.dtype in [np.float, np.double] else '{}'
        fmt_str = ', '.join(val_fmt for _ in range(v.shape[0]))
        return fmt_str.format(*v)
    else:
        return '{}'.format(v)


def print_dict(d, prec=3):
    return ','.join('{}: {}'.format(k, fmt_value(v, prec=prec)) for k, v in d.items())


def maybe_make_dir(fname_path):
    path, _ = os.path.split(fname_path)
    if not os.path.isdir(path):
        print('Creating new directory', path)
        os.makedirs(path)


def open_file(fname_path, *args, **kwargs):
    maybe_make_dir(fname_path)
    return open(fname_path, *args, **kwargs)


def untorchify(c):
    return c.squeeze().cpu().numpy()


def rotate(t, axes=(2, 3)):
    if len(axes) != 2:
        raise ValueError("`axes` must be length 2, got {}".format(axes))

    return torch.flip(t, (axes[0], )).transpose(*axes)


def unrotate(t):
    return rotate(rotate(rotate(t)))


def cut_to_match(reference, t):
    '''
    Slice tensor `t` along spatial dimensions to match `reference`, by
    picking the central region
    '''

    if reference.size()[2:] == t.size()[2:]:
        # sizes match, no slicing necessary
        return t

    # compute the difference along all spatial axes
    diffs = [s - r for s, r in zip(t.size()[2:], reference.size()[2:])]

    # check if diffs are even, which is necessary if we want a truly centered crop
    if not all(d % 2 == 0 for d in diffs) and all(d >= 0 for d in diffs):
        fmt = "Tried to slice `t` of size {} to match `reference` of size {}"
        msg = fmt.format(t.size(), reference.size())
        raise RuntimeError(msg)

    # pick the full extent of `batch` and `feature` axes
    slices = [slice(None, None), slice(None, None)] 

    # for all remaining pick between diff//2 and size-diff//2
    for d in diffs:
        if d > 0:
            slices.append(slice(d // 2, -(d // 2)))
        elif d == 0:
            slices.append(slice(None, None))

    return t[slices]


def size_adaptive(method):
    @functools.wraps(method)
    def wrapped(self, input, target, *args, **kwargs):
        target = cut_to_match(input, target)

        cut_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cut = cut_to_match(input, arg)
            else:
                cut = arg
            cut_args.append(cut)
        
        return method(self, input, target, *cut_args, **kwargs)

    return wrapped


def size_adaptive_(cls):
    cls.forward = size_adaptive(cls.forward)

    return cls


if __name__ == '__main__':
    import unittest

    class Tests(unittest.TestCase):
        def test_cut_to_match(self):
            src = torch.zeros(2, 3, 10, 10, 8)
            tar = torch.ones(2, 3, 6, 6, 6)

            src[:, :, 2:8, 2:8, 1:7] = 1

            self.assertTrue((cut_to_match(tar, src) == tar).all())

        def test_avg(self):
            a = AvgMeter()

            a.update(1)
            a.update(2)
            a.update(3)

            self.assertEquals(a.avg, 2)

    unittest.main()
