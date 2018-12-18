import torch, os, imageio, re, sys
import numpy as np
import torchvision as tv
import torchvision.transforms as T
from PIL import Image

sys.path.append('..')
import transforms as tr

ROTATE_512 = tr.Compose([
    tr.RandomRotate(),
    tr.Lift(T.ToTensor()),
    tr.RandomCropTransform((512, 512))
])

CROP_512 = tr.Compose([
    tr.Lift(T.ToTensor()),
    tr.RandomCropTransform((512, 512)),
])
