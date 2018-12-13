import torch, random
import torchvision.transforms as T
from PIL import Image

class Lift:
    def __init__(self, f):
        self.f = f

    def __call__(self, *imgs):
        return list(map(self.f, imgs))


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, *imgs):
        outs = imgs

        for transformation in self.transformations:
            outs = list(transformation(*outs))

        return outs


class ResizeTransform(Lift):
    def __init__(self, target_size):
        self.target_size = target_size
        def resize(img):
            resample = Image.NEAREST if img.mode == 'L' else Image.BICUBIC
            return img.resize(self.target_size, resample=resample)

        super(ResizeTransform, self).__init__(resize)

class RandomFlip:
    def __call__(self, *imgs):
        flip_lr = random.randint(0, 1)
        flip_td = random.randint(0, 1)

        def do_flip(img):
            if flip_lr == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_td == 1:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            return img

        return list(map(do_flip, imgs))


class AspectPreservingResizeTransform(Lift):
    def __init__(self, target_size):
        self.w, self.h = target_size

        def resize(img):
            if img.size == (self.w, self.h):
                return img

            w_ratio = self.w / img.size[0]
            h_ratio = self.h / img.size[1]
            ratio = min(w_ratio, h_ratio)

            new_size = int(ratio * img.size[0]), int(ratio * img.size[1])
            interpolation = Image.NEAREST if img.mode == 'L' else Image.BICUBIC
            resized = img.resize(new_size, resample=interpolation)

            assert resized.size[0] <= self.w and resized.size[1] <= self.h

            padded = Image.new(img.mode, (self.w, self.h))
            padded.paste(
                resized,
                (
                    (padded.size[0] - resized.size[0]) // 2,
                    (padded.size[1] - resized.size[1]) // 2
                )
            )

            return padded

        super(AspectPreservingResizeTransform, self).__init__(resize)


class PadTransform(Lift):
    def __init__(self, target_size):
        self.w, self.h = target_size

        def pad(img):
            padded = Image.new(img.mode, (self.w, self.h))
            padded.paste(
                img,
                (
                    (padded.size[0] - img.size[0]) // 2,
                    (padded.size[1] - img.size[1]) // 2
                )
            )

            return padded

        super(PadTransform, self).__init__(pad)

class RandomCropTransform:
    def __init__(self, target_size):
        self.w, self.h = target_size

    def __call__(self, *imgs):
        w, h = imgs[0].shape[1:3]
        assert w >= self.w and h >= self.h
        sw = random.randint(0, w-self.w)
        sh = random.randint(0, h-self.h)
        
        return [img[:, sw:sw+self.w, sh:sh+self.h] for img in imgs]


class CenterCropTransform(Lift):
    def __init__(self, target_size):
        self.w, self.h = target_size

        def crop(img):
            w, h = img.shape[1:3]
            assert w >= self.w and h >= self.h
            
            dw = w - self.w
            dh = h - self.h

            if dw >= 0:
                half_w = int(dw / 2)
                img = img[:, half_w:half_w+self.w, :]
            
            if dh >= 0:
                half_h = int(dh / 2)
                img = img[:, :, half_h:half_h+self.h]

            return img

        super(CenterCropTransform, self).__init__(crop)


class RandomRotate:
    def __call__(self, *imgs):
        angle = random.randint(-180, 180)

        rotated = []
        for img in imgs:
            if img.mode == 'L':
                rotated.append(img.rotate(angle, resample=Image.NEAREST))
            elif img.mode == 'RGB':
                rotated.append(img.rotate(angle, resample=Image.BICUBIC))
            else:
                raise ValueError("Encountered unsupported mode `{}`".format(img.mode))

        return rotated


class Normalize:
    def __init__(self, r=(180., 39.), g=(151., 41.48), b=(139., 45.18)):
        self.r = r
        self.g = g
        self.b = b

        self.mean = torch.tensor([r[0], g[0], b[0]]).reshape(3, 1, 1)
        self.std = torch.tensor([r[1], g[1], b[1]]).reshape(3, 1, 1)

    def __call__(self, img):
        mask = (img == 0.).all(dim=0, keepdim=True).expand(3, -1, -1)
        
        img = img.clone()
        img[~mask] = (img[~mask] - self.mean) / self.std

        return img
