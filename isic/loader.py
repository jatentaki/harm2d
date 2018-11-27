import torch, os, imageio, re, random
import torchvision as tv
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

class ISICNormalize:
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

class ISICDataset:
    def __init__(self, path, img_transform=None,
                 lbl_transform=None, global_transform=None, normalize=False):

        self.normalize = normalize

        self.img_p = path + os.path.sep + 'imgs'
        self.lbl_p = path + os.path.sep + 'lbls'

        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.global_transform = global_transform

        img_re = re.compile('^ISIC_(\d{7})\.jpg$')
        lbl_re = re.compile('^ISIC_(\d{7})_segmentation\.png$')

        img_ids = []
        for file in os.listdir(self.img_p):
            m = img_re.match(file)
            if m:
                id = int(m.group(1))
                img_ids.append(id)

        lbl_ids = []
        for file in os.listdir(self.lbl_p):
            m = lbl_re.match(file)
            if m:
                id = int(m.group(1))
                lbl_ids.append(id)

        diff = set(img_ids) ^ set(lbl_ids)
        if diff:
            fmt = ("img and lbl directories are asymmetric. "
                   "{} found in one but not the other")
            msg = fmt.format(diff)
            raise AssertionError(msg)

        self.ids = img_ids

    def fetch_lbl(self, id):
        fmt = self.lbl_p + os.path.sep + 'ISIC_{}_segmentation.png'
        path = fmt.format(id)
        return Image.fromarray(imageio.imread(path), mode='L')

    def fetch_img(self, id):
        fmt = self.img_p + os.path.sep + 'ISIC_{}.jpg'
        path = fmt.format(id)
        return Image.fromarray(imageio.imread(path))

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        id = str(id).zfill(7)
        img = self.fetch_img(id)
        lbl = self.fetch_lbl(id)
        mask = Image.new('L', lbl.size, color=255)

        if self.global_transform:
            img, mask, lbl = self.global_transform(img, mask, lbl)

        if self.img_transform:
            img = self.img_transform(img)

        if self.lbl_transform:
            lbl = self.lbl_transform(lbl)

        mask = mask.to(torch.uint8)

        if self.normalize:
            img = self._normalize(img, mask)

        return img, mask, lbl

    def _normalize(self, img, mask):
        mean = torch.tensor([0.71, 0.57, 0.53]).reshape(3, 1, 1)
        std = torch.tensor([0.17, 0.17, 0.18]).reshape(3, 1, 1)

        demeaned = (img - mean) / std

        return torch.where(mask, demeaned, torch.tensor(0.))

if __name__ == '__main__':
    target_size = 1024, 768
    trans = Compose([
        AspectPreservingResizeTransform(target_size),
#        Lift(T.Pad(88)),
        Lift(T.ToTensor()),
    #    RandomCropTransform((564, 564))
    ])
    d = ISICDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/isic2018',
        global_transform=trans, normalize=True
    )
    import matplotlib.pyplot as plt
    for i in range(20, 50):
        img, mask, lbl = d[i]

        img = img.numpy().transpose(1, 2, 0)
        lbl = lbl.numpy()[0]
        plt.imshow(img[..., 0], vmin=-3, vmax=3)
#        plt.figure()
#        plt.imshow(lbl)
#        plt.figure()
#        plt.imshow(mask.to(torch.uint8).numpy()[0])
        plt.show()
