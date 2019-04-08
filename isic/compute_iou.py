import torch, os, imageio
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from isic_f1 import process_one_img, postprocess

#gt_path = '/cvlabdata2/cvlab/datasets_tyszkiewicz/isic2018/test/lbls/'
pr_path = '/cvlabdata2/home/tyszkiew/harmonic_submission_validation_segmentations/'
#save_path = '/cvlabdata2/home/tyszkiew/harmonic_submission_pngs/'
save_path='~/harmonic_submission_validation/'

thresholds = np.linspace(0.3, 0.7, 10)

def calculate_job(file):
    pred = np.load(os.path.join(pr_path, file))
    seg_fname = file[:-4] + '_segmentation.png'
    gt = imageio.imread(os.path.join(gt_path, seg_fname)) == 255

    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt.astype(np.uint8))
    mask = torch.ones_like(gt)

    score = process_one_img(pred, mask, gt, thresholds, logit_input=False)

    return score, file

def save_job(file):
    pred = np.load(os.path.join(pr_path, file))
    pred = pred > 0.47777777777777775
    postprocessed = postprocess(pred)
    postprocessed = postprocessed.astype(np.uint8) * 255
    save_p = os.path.join(save_path, file[:-3] + 'png')
    
    imageio.imwrite(save_p, postprocessed)

fnames = list(filter(lambda f: f.endswith('npy'), os.listdir(pr_path)))

pool = mp.Pool()

imap = pool.imap_unordered(save_job, fnames, chunksize=1)
for _ in tqdm(imap, total=len(fnames)):
    pass

#imap = pool.imap_unordered(calculate_job, fnames, chunksize=1)
#scores = []
#for score, fname in tqdm(imap, total=len(fnames)):
#    scores.append(score)
#    
#means = np.stack(scores)
#means[means < 0.65] = 0.
#means = means.mean(axis=0)
#argmax = means.argmax()
#print('best iou', means[argmax], 'at', thresholds[argmax])
