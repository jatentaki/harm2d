import torch, os, imageio, re, random, sys
import numpy as np
import torchvision as tv
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.ndimage as ndi
from PIL import Image

sys.path.append('..')
import transforms as tr
from utils import rotated_dataset

def random_flip(*tensors):
    dims = []
    if random.randint(0, 1):
        dims.append(-1)
    if random.randint(0, 1):
        dims.append(-2)

    return [torch.flip(t, dims) for t in tensors]

class ISICDataset:
    def __init__(self, path, img_transform=T.ToTensor(),
                 lbl_transform=T.ToTensor(), mask_transform=T.ToTensor(),
                 global_transform=None, eg_size=10,
                 bg_weight=1., fg_weight=1., eg_weight=1., flip=False):
        self.img_p = os.path.join(path, 'imgs')
        self.lbl_p = os.path.join(path, 'lbls')
        self.mean_p = os.path.join(path, 'mean.npy')
        self.flip = flip

        mean = torch.from_numpy(np.load(self.mean_p).astype(np.float32))
        self.mean = F.pad(mean, (88, 88, 88, 88))

        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.lbl_transform = lbl_transform
        self.global_transform = global_transform

        total_weight = bg_weight + fg_weight + eg_weight
        self.bg_weight = bg_weight / total_weight
        self.fg_weight = fg_weight / total_weight
        self.eg_weight = eg_weight / total_weight
        self.eg_size   = eg_size

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

        self.ids = sorted(img_ids)

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

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.lbl_transform:
            lbl = self.lbl_transform(lbl)

        lbl = lbl.to(torch.uint8)
        edge = ndi.morphological_gradient(lbl.numpy()[0], size=self.eg_size)

        edge = torch.from_numpy(edge)[None]
        background = ~lbl & ~edge
        foreground =  lbl & ~edge

        weights = torch.zeros_like(edge, dtype=torch.float32)
        n = weights.numel()
        weights[edge] = n * self.eg_weight / edge.sum().item()
        weights[background] = n * self.bg_weight / background.sum().item()
        weights[foreground] = n * self.fg_weight / foreground.sum().item()

        mask = mask * weights

        img = torch.cat([img, self.mean[None]], dim=0)

        if self.flip:
            img, mask, lbl = random_flip(img, mask, lbl)

        return img, mask, lbl

ROTATE_TRANS_1024 = tr.Compose([
    tr.AspectPreservingResizeTransform((1024, 768)),
    tr.Lift(T.Pad(88)),
    tr.RandomRotate(),
])

PAD_TRANS_1024 = tr.Compose([
    tr.AspectPreservingResizeTransform((1024, 768)),
    tr.Lift(T.Pad(88)),
])

RotatedISICDataset = rotated_dataset(ISICDataset)

if __name__ == '__main__':
    target_size = 1024, 768
    
    img_transform=T.Compose([
        T.ColorJitter(0.3, 0.3, 0.3, 0.),
        T.ToTensor()
    ])

    d = ISICDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/isic2018',
        global_transform=PAD_TRANS_1024,
        eg_size=10,
        img_transform=img_transform
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4)

    for i, ax in enumerate(axes.flatten()):
        img, mask, lbl = d[0]
        ax.imshow(img.numpy().transpose(1, 2, 0)[..., :3])
        ax.set_axis_off()

    fig.tight_layout()
    plt.show()
