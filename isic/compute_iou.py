import torch, os, imageio
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from isic_f1 import process_one_img, postprocess

gt_path = '/cvlabdata2/cvlab/datasets_tyszkiewicz/isic2018/test/lbls/'
pr_path = '/cvlabdata2/home/tyszkiew/full_size_iou_jitter_rayleigh_flip/'
save_path = '/cvlabdata2/home/tyszkiew/test_postprocessed/'
#save_path = '~/test_postprocessed'

thresholds = np.array([0.52])#np.linspace(0.3, 0.7, 10)

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
    pred = pred > 0.52
    postprocessed = postprocess(pred)
    postprocessed = postprocessed.astype(np.uint8) * 255
    save_p = os.path.join(save_path, file[:-3] + 'png')
    
    imageio.imwrite(save_p, postprocessed)

fnames = list(filter(lambda f: f.endswith('npy'), os.listdir(pr_path)))

#list(map(save_job, fnames))
pool = mp.Pool()

imap = pool.imap_unordered(save_job, fnames, chunksize=1)
for _ in tqdm(imap, total=len(fnames)):
    pass

#scores = []
#for score, fname in tqdm(imap, total=len(fnames)):
#    scores.append(score)
#    
#means = np.stack(scores).mean(axis=0)
#argmax = means.argmax()
#print('best iou', means[argmax], 'at', thresholds[argmax])
