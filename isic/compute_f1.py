import torch, os, imageio
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from isic_f1 import process_one_img

gt_path = '/cvlabdata2/cvlab/datasets_tyszkiewicz/isic2018/test/lbls'
pr_path = '/cvlabdata2/home/tyszkiew/full_size_iou_jitter_rayleigh_flip'

thresholds = np.array([0.5455])#np.linspace(0.3, 0.7, 10)
def job(file):
    pred = np.load(os.path.join(pr_path, file))
    seg_fname = file[:-4] + '_segmentation.png'
    gt = imageio.imread(os.path.join(gt_path, seg_fname)) == 255

    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt.astype(np.uint8))
    mask = torch.ones_like(gt)

    score = process_one_img(pred, mask, gt, thresholds, logit_input=False)

    return score, file

fnames = list(filter(lambda f: f.endswith('npy'), os.listdir(pr_path)))[:100]

pool = mp.Pool()
means = np.zeros_like(thresholds)

imap = pool.imap_unordered(job, fnames, chunksize=1)
scores = []
for score, fname in tqdm(imap, total=len(fnames)):
    scores.append((fname, score))
    print(f'{fname} -> {score}')
    
for p, s in scores:
    if 0.55 < s < 0.75:
        print(p)
#means = np.stack(scores).mean(axis=0)
#argmax = means.argmax()
#print('best iou', means[argmax], 'at', thresholds[argmax])
