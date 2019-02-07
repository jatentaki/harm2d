import torch, os, imageio
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from isic_f1 import process_one_img

#gt_path = '~/Storage/jatentaki/Datasets/isic2018/lbls'
gt_path = '/cvlabdata2/cvlab/datasets_tyszkiewicz/isic2018/train/lbls'
pr_path = '/cvlabdata2/home/tyszkiew/full_size_raw'

def compute_one(pname):
    gname = pname[:-4] + '_segmentation.png'
    gpath = os.path.join(gt_path, gname)

    ppath = os.path.join(pr_path, pname)

    gt = imageio.imread(gpath) == 255
    pr = np.load(ppath) 
    
    score = jaccard(pr, gt)

    return score
    #if score >= 0.65:
    #    return score
    #else:
    #    return 0.

thresholds = np.array([0.6])#np.linspace(0.2, 0.8, 10)
def job(file):
    pred = np.load(os.path.join(pr_path, file))['seg']
    seg_fname = file[:-4] + '_segmentation.png'
    gt = imageio.imread(os.path.join(gt_path, seg_fname)) == 255

    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt.astype(np.uint8))
    mask = torch.ones_like(gt)

    score = process_one_img(pred, mask, gt, thresholds)

    return score, file

fnames = list(filter(lambda f: f.endswith('npz'), os.listdir(pr_path)))[:100]

pool = mp.Pool()
means = np.zeros_like(thresholds)
n = 0

imap = pool.imap_unordered(job, fnames, chunksize=1)
scores = dict()
for score, fname in tqdm(imap, total=len(fnames)):
    #means += iou
    #n += 1
    scores[fname] = score

for name, score in scores.items():
    if score < 0.7:
        print(name, score)

#means /= n
#
#argmax = means.argmax()
#print('best iou', means[argmax], 'at', thresholds[argmax])
#print(np.stack(ious).mean(axis=0))
