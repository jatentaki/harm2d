import torch, os, imageio, re, random
import torchvision as tv
import torchvision.transforms as T
from itertools import cycle, islice
from PIL import Image

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

class ToBinary:
    def __call__(self, lbl):
        return lbl != 0

class DriveDataset:
    def __init__(self, path, training=True, img_transform=T.ToTensor(),
                 label_transform=T.ToTensor(), global_transform=None, bloat=None):
        self.training = training

        self.subdir = 'train' if training else 'test'
        self.path = os.path.join(path, self.subdir)

        self.global_transform = global_transform
        self.img_transform = img_transform
        self.label_transform = label_transform

        img_re = re.compile('^testA_(\d+)\.bmp$')
        lbl_re = re.compile('^testA_(\d+)_anno\.bmp$')

        imgs = set()
        lbls = set()
        for file in os.listdir(self.path):
            img_m = img_re.match(file)
            if img_m:
                imgs.add(img_m.group(1))

            lbl_m = lbl_re.match(file)
            if lbl_m:
                lbls.add(lbl_m.group(1))

        difference = imgs ^ lbls
        if difference:
            msg = f"Files in {difference} do not have their label/img counterparts"
            raise RuntimeError(msg)
            
        self.samples = []
        for id in sorted(list(imgs)):
            self.samples.append((self.fetch_img(id), self.fetch_lbl(id)))

        if bloat is not None:
            cycled = cycle(self.samples)
            self.samples = list(islice(cycled, bloat))

        print(f'Dataset size (post bloat and cut): {len(self.samples)}')


    def fetch_lbl(self, id):
        path = os.path.join(self.path, f'testA_{id}_anno.bmp')
        return Image.fromarray(imageio.imread(path), mode='L')


    def fetch_img(self, id):
        path = os.path.join(self.path, f'testA_{id}.bmp')
        return Image.fromarray(imageio.imread(path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, label = self.samples[idx]

        if self.global_transform:
            img, label = self.global_transform(img, label)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        return img, label 

if __name__ == '__main__':
    lbl_trans = T.Compose([T.ToTensor(), ToBinary()])
    d = DriveDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/glas',
        label_transform=lbl_trans
    )

    import matplotlib.pyplot as plt

    for i in range(5):
        img, lbl = d[i]
        fig, (a1, a2) = plt.subplots(2)

        a1.imshow(img.permute(1, 2, 0).numpy())
        a2.imshow(lbl[0].numpy())
        plt.show()
