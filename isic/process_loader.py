import os, imageio, re
import torch.nn.functional as F
from PIL import Image

class ProcessDataset:
    def __init__(self, path):
        self.img_p = path

        img_re = re.compile('^ISIC_(\d{7})\.jpg$')
        self.ids = []
        for file in os.listdir(self.img_p):
            m = img_re.match(file)
            if m:
                id = int(m.group(1))
                self.ids.append(id)

    def fetch_img(self, id):
        fmt = self.img_p + os.path.sep + 'ISIC_{}.jpg'
        path = fmt.format(id)
        return path, Image.fromarray(imageio.imread(path))

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        id = str(id).zfill(7)
        path, img = self.fetch_img(id)

        return path, img

class APRTrans:
    def __init__(self, target_size):
        self.w, self.h = target_size

    def forward(self, img):
        self.img_size = img.size

        w_ratio = self.w / img.size[0]
        h_ratio = self.h / img.size[1]
        ratio = min(w_ratio, h_ratio)

        new_size = int(ratio * img.size[0]), int(ratio * img.size[1])
        interpolation = Image.NEAREST if img.mode == 'L' else Image.BICUBIC
        resized = img.resize(new_size, resample=interpolation)

        assert resized.size[0] <= self.w and resized.size[1] <= self.h

        padded = Image.new(img.mode, (self.w, self.h))

        self.padding_x = (padded.size[0] - resized.size[0]) // 2
        self.padding_y = (padded.size[1] - resized.size[1]) // 2

        padded.paste(resized, (self.padding_x, self.padding_y))

        return padded

    def backward(self, ten):
        if self.padding_x == 0:
            x_s = slice(None)
        else:
            x_s = slice(self.padding_x, -self.padding_x)

        if self.padding_y == 0:
            y_s = slice(None)
        else:
            y_s = slice(self.padding_y, -self.padding_y)

        unpadded = ten[..., y_s, x_s]

        return F.interpolate(unpadded[None], size=self.img_size[::-1], mode='bilinear')[0]

if __name__ == '__main__':
    d = ProcessDataset(
        '/home/jatentaki/Storage/jatentaki/Datasets/isic2018',
    )

    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    from tqdm import tqdm


    for i in tqdm(range(len(d))):
        img = d[i]
        ten = T.ToTensor()(img)

        resize = APRTrans((1024, 768))

        downsized = T.ToTensor()(resize.forward(img))
        upscaled = resize.backward(downsized[:1])

#        plt.imshow(downsized.numpy().transpose(1, 2, 0))
#        plt.figure()
#        plt.imshow(upscaled.numpy()[0])
#        plt.show()

        if ten.shape[1:] != upscaled.shape[1:]:
            print(ten.shape, upscaled.shape)
            break
