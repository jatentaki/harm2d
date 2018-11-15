import torch, os, imageio, re
import torchvision as tv
import torchvision.transforms as T
from PIL import Image

class DriveDataset:
    def __init__(self, path, training=True, img_transform=T.ToTensor(),
                 mask_transform=T.ToTensor(), label_transform=T.ToTensor(),
                 bloat=1):
        self.training = training

        self.subdir = 'training' if training else 'test'
        self.anno_p = path + os.path.sep + self.subdir + os.path.sep + '1st_manual'
        self.img_p = path + os.path.sep + self.subdir + os.path.sep + 'images'
        self.mask_p = path + os.path.sep + self.subdir + os.path.sep + 'mask'

        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.label_transform = label_transform

        regex = re.compile('^(\d+)_' + self.subdir + '\.tif$')

        self.samples = []
        for file in os.listdir(self.img_p):
            m = regex.match(file)
            if m:
                id = int(m.group(1))
                id = id if self.training else str(id).zfill(2)

                img = self.fetch_img(id)
                mask = self.fetch_mask(id)
                label = self.fetch_label(id)

                self.samples.append((img, mask, label))

        self.samples = self.samples * bloat


    def fetch_label(self, id):
        fmt = self.anno_p + os.path.sep + '{}_manual1.gif'
        path = fmt.format(id)
        return Image.fromarray(imageio.imread(path), mode='L')

    def fetch_mask(self, id):
        fmt = self.mask_p + os.path.sep + '{}_' + self.subdir + '_mask.gif'
        path = fmt.format(id)
        return Image.fromarray(imageio.imread(path), mode='L')

    def fetch_img(self, id):
        fmt = self.img_p + os.path.sep + '{}_' + self.subdir + '.tif'
        path = fmt.format(id)
        return Image.fromarray(imageio.imread(path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, mask, label = self.samples[idx]

        if self.img_transform:
            img = self.img_transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.label_transform:
            label = self.label_transform(label)

        return img, mask, label 

if __name__ == '__main__':
    d = DriveDataset('dataset', training=False)
    print(d[5])
