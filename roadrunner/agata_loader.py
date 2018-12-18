import os, imageio, re, torch
import numpy as np
from PIL import Image

class AgataRoadDataset:
    def __init__(self, path, img_transform=None, mask_transform=None, training=True,
                 lbl_transform=None, global_transform=None):

        img_dir = 'train_imgs' if training else 'test_imgs'
        self.img_p = os.path.join(path, img_dir)
        self.lbl_p = os.path.join(path, 'binary_gt')

        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.mask_transform = mask_transform
        self.global_transform = global_transform

        img_re = re.compile('^(?P<name>.+?)_sat\.png$')
        lbl_re = re.compile('^(?P<name>.+?)_osm\.png$')

        img_names = []
        for file in os.listdir(self.img_p):
            m = img_re.match(file)
            if m:
                name = m.group(1)
                img_names.append(name)

        lbl_names = []
        for file in os.listdir(self.lbl_p):
            m = lbl_re.match(file)
            if m:
                name = m.group(1)
                lbl_names.append(name)

        diff = set(img_names) - set(lbl_names)
        if diff:
            raise AssertionError(f"{diff} are missing their labels")

        self.names = img_names

    def fetch_lbl(self, name):
        path = os.path.join(self.lbl_p, f'{name}_osm.png')
        lbl = imageio.imread(path)[..., 0]
        lbl = (lbl > 100).astype(np.uint8) * 255
        return Image.fromarray(lbl, mode='L')

    def fetch_img(self, name):
        path = os.path.join(self.img_p, f'{name}_sat.png')
        return Image.fromarray(imageio.imread(path)[..., :3])

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        img = self.fetch_img(name)
        lbl = self.fetch_lbl(name)
        mask = Image.new('L', lbl.size, color=255)

        if self.global_transform:
            img, mask, lbl = self.global_transform(img, mask, lbl)

        if self.img_transform:
            img = self.img_transform(img)

        if self.lbl_transform:
            lbl = self.lbl_transform(lbl)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.to(torch.uint8)

        return img, mask, lbl

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    sys.path.append('..')
    import transforms as tr

    d = AgataRoadDataset(
        '/home/jatentaki/Programs/msc/remote_home/',
        global_transform=tr.Lift(T.ToTensor())
    )

    for i in range(25, 50):
        img, mask, lbl = d[i]

        fig, (a1, a2, a3) = plt.subplots(1, 3)
        img = img.numpy().transpose(1, 2, 0)
        a1.imshow(img)
        a2.imshow(lbl.numpy()[0])
        a3.imshow(mask.numpy()[0])

        plt.show()
