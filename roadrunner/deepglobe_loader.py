import os, imageio, re, torch
import numpy as np
from PIL import Image


class DeepglobeDataset:
    def __init__(self, path, img_transform=None, mask_transform=None,
                 lbl_transform=None, global_transform=None):

        self.path = path
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.mask_transform = mask_transform
        self.global_transform = global_transform

        img_re = re.compile('^(\d+)_sat\.jpg$')
        lbl_re = re.compile('^(\d+)_mask\.png$')

        imgs, lbls = [], []
        for file in os.listdir(self.path):
            img_m = img_re.match(file)
            if img_m:
                id = int(img_m.group(1))
                imgs.append(id)

            lbl_m = lbl_re.match(file)
            if lbl_m:
                id = int(lbl_m.group(1))
                lbls.append(id)

        diff = set(imgs) ^ set(lbls)
        if diff:
            raise AssertionError(f"Diff not empty ({diff})")

        self.ids = imgs

    def fetch_lbl(self, id):
        path = os.path.join(self.path, f'{id}_mask.png')
        return Image.fromarray(imageio.imread(path)[..., 0], mode='L')

    def fetch_img(self, id):
        path = os.path.join(self.path, f'{id}_sat.jpg')
        return Image.fromarray(imageio.imread(path))

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        img = self.fetch_img(id)
        lbl = self.fetch_lbl(id)
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

import sys
sys.path.append('..')
from utils import rotated_dataset

RotatedDeepglobeDataset = rotated_dataset(DeepglobeDataset)

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    sys.path.append('..')
    import transforms as tr

    d = RotatedDeepglobeDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/roads/deepglobe/test/',
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
