import os, imageio
import multiprocessing as mp
import numpy as np

#gt_path = '~/Storage/jatentaki/Datasets/isic2018/lbls'
gt_path = '/cvlabdata2/cvlab/datasets_tyszkiewicz/isic2018/test/lbls'
pr_path = '/cvlabdata2/home/tyszkiew/full_size_iou'
#pr_path = 'full_size'

def f1(pr, gt):
    tp =  pr &  gt
    fp =  pr & ~gt
    fn = ~pr & gt

    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()

    f_1 = 2 * tp / (2 * tp + fn + fp)
    return f_1

def jaccard(pr, gt):
    tp =  pr &  gt
#    fp =  pr & ~gt
#    fn = ~pr & gt

    tp = tp.sum()
    p = pr.sum()
    t = gt.sum()

    return tp / (t + p - tp)
#    fp = fp.sum()
#    fn = fn.sum()

def compute_one(pname):
    gname = pname[:-4] + '_segmentation.png'
    gpath = os.path.join(gt_path, gname)

    ppath = os.path.join(pr_path, pname)

    gt = imageio.imread(gpath) == 255
    pr = np.load(ppath) 
    
    score = jaccard(pr, gt)

    if score >= 0.65:
        return score
    else:
        return 0.

with mp.Pool() as p:
    scores = p.map(compute_one, os.listdir(pr_path))

    print(np.array(scores).mean())
