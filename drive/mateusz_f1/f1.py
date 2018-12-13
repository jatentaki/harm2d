import imageio, sys, torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

sys.path.append('../..')
from criteria import PrecRec

imgs = []
masks = []
lbls = []

for i in range(1, 21):
    nr = str(i).zfill(2)
    fname = f'{nr}_test.tif.npy'

    imgp = join('f1_mateusz', f'{nr}_test.tif.npy')
    img = np.load(imgp)
    imgs.append(img)

    maskp = join('mask', f'{nr}_test_mask.gif')
    mask = imageio.imread(maskp)
    masks.append(mask)
    
    lblp = join('1st_manual', f'{nr}_manual1.gif')
    lbl = imageio.imread(lblp)
    lbls.append(lbl)

#
#    fig, (a1, a2, a3) = plt.subplots(3)
#    a1.imshow(img)
#    a2.imshow(mask)
#    a3.imshow(lbl)
#    plt.show()

masks = np.stack(masks)
imgs = np.stack(imgs).astype(np.float32)
lbls = np.stack(lbls)

pr = PrecRec(n_thresholds=100)

pr(torch.from_numpy(imgs), torch.from_numpy(masks), torch.from_numpy(lbls))

print(pr.best_f1())
