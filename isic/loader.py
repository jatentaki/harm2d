import torch, os, imageio, re, random, sys
import torchvision as tv
import torchvision.transforms as T
from PIL import Image

sys.path.append('..')
import transforms as tr

class ISICDataset:
    def __init__(self, path, img_transform=T.ToTensor(), mask_transform=T.ToTensor(),
                 lbl_transform=T.ToTensor(), global_transform=None, normalize=False):

        self.normalize = normalize

        self.img_p = path + os.path.sep + 'imgs'
        self.lbl_p = path + os.path.sep + 'lbls'

        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.mask_transform = mask_transform
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

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.to(torch.uint8)

        if self.normalize:
            img = self._normalize(img, mask)

        return img, mask, lbl

    def _normalize(self, img, mask):
        mean = torch.tensor([0.71, 0.57, 0.53]).reshape(3, 1, 1)
        std = torch.tensor([0.17, 0.17, 0.18]).reshape(3, 1, 1)

        demeaned = (img - mean) / std

        return torch.where(mask, demeaned, torch.tensor(0.))

    def denormalize(self, img):
        mean = torch.tensor([0.71, 0.57, 0.53]).reshape(3, 1, 1)
        std = torch.tensor([0.17, 0.17, 0.18]).reshape(3, 1, 1)
        
        return torch.clamp(img * std + mean, 0., 1.)

ROTATE_TRANS_1024 = tr.Compose([
    tr.AspectPreservingResizeTransform((1024, 768)),
    tr.Lift(T.Pad(88)),
    tr.RandomRotate(),
])

PAD_TRANS_1024 = tr.Compose([
    tr.AspectPreservingResizeTransform((1024, 768)),
    tr.Lift(T.Pad(88)),
])

if __name__ == '__main__':
    target_size = 1024, 768
    
    d = ISICDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/isic2018',
        global_transform=ROTATE_TRANS_1024, normalize=True,
        img_transform=T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, hue=0.05),
            T.ToTensor()
        ])

    )
    import matplotlib.pyplot as plt
    for i in range(25, 50):
        img, mask, lbl = d[0]

        img = d.denormalize(img).numpy().transpose(1, 2, 0)
        lbl = lbl.numpy()[0]
#        plt.imshow(img[..., 0], vmin=-2, vmax=2)
        plt.imshow(img)
#        plt.figure()
#        plt.imshow(lbl)
#        plt.figure()
#        plt.imshow(mask.to(torch.uint8).numpy()[0])
        plt.show()
