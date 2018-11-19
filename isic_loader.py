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


class RandomCropTransform:
    def __init__(self, target_size):
        self.w, self.h = target_size

    def __call__(self, *imgs):
        w, h = imgs[0].shape[1:3]
        assert w >= self.w and h >= self.h
        sw = random.randint(0, w-self.w)
        sh = random.randint(0, h-self.h)
        
        return [img[:, sw:sw+self.w, sh:sh+self.h] for img in imgs]



class ISICDataset:
    def __init__(self, path, img_transform=None,
                 lbl_transform=None, global_transform=None):

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

        if self.global_transform:
            img, lbl = self.global_transform(img, lbl)

        if self.img_transform:
            img = self.img_transform(img)

        if self.lbl_transform:
            lbl = self.lbl_transform(lbl)

        return img, lbl

if __name__ == '__main__':
    target_size = 1024, 768
    trans = Compose([
        ResizeTransform(target_size),
        Lift(T.ToTensor()),
        RandomCropTransform((564, 564))
    ])
    d = ISICDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/isic2018',
        global_transform=trans
    )
    img, lbl = d[172]

    import matplotlib.pyplot as plt
    img = img.numpy().transpose(1, 2, 0)
    lbl = lbl.numpy()[0]
    plt.imshow(img)
    plt.figure()
    plt.imshow(lbl)
    plt.show()
