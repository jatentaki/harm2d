import torch, argparse, functools, os, sys
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import harmonic

# local directory
from process_loader import ProcessDataset, APRTrans
from isic_f1 import postprocess

# parent directory
sys.path.append('..')
import hunet, unet
import transforms as tr
import framework

parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
# paths
parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
parser.add_argument('mean_path', metavar='DIR',
                    help='path to the mean of the dataset')
parser.add_argument('artifacts', metavar='DIR',
                    help='path to store artifacts')
parser.add_argument('load', metavar='FILE', type=str,
                    help='load an existing model')

# behavior choice
parser.add_argument('--dropout', metavar='F', default=None, type=float,
                    help='Dropout probability')
parser.add_argument('-j', '--workers', metavar='N', default=1, type=int,
                    help='number of data loader workers')
parser.add_argument('model', choices=['harmonic', 'baseline'])

# other
parser.add_argument('-s', '--early_stop', default=None, type=int,
                    help='stop early after n batches')

args = parser.parse_args()

if not os.path.isdir(args.artifacts):
    print('creating artifacts directory', args.artifacts)
    os.makedirs(args.artifacts)

data = ProcessDataset(args.data_path, args.mean_path)
loader = DataLoader(
    data, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=list
)

down = [(5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5)]
up = [(5, 5, 5), (5, 5, 5), (5, 5, 5)]
if args.model in ('harmonic', 'unconstrained'):
    setup = hunet.default_setup
    if args.dropout is not None:
        dropout = functools.partial(harmonic.d2.Dropout2d, p=args.dropout)
        setup['dropout'] = dropout

    setup['norm'] = harmonic.d2.RayleighNorm2d
    network = hunet.HUnet(in_features=4, down=down, up=up, radius=2, setup=setup)

elif args.model == 'baseline':
    if args.dropout is not None:
        dropout = functools.partial(torch.nn.Dropout2d, p=args.dropout)
        setup = {**unet.default_setup, 'dropout': dropout}
    else:
        setup = unet.default_setup

    down = [unet.repr_to_n(d) for d in down]
    up = [unet.repr_to_n(d) for d in up]
    network = unet.Unet(up=up, down=down, in_features=3, setup=setup)

cuda = torch.cuda.is_available()

n_params = 0
for param in network.parameters():
    n_params += param.numel()
print(n_params, 'learnable parameters')

if cuda:
    network = network.cuda()

checkpoint = framework.load_checkpoint(args.load)
_, _, model_dict, _ = checkpoint

network.load_state_dict(model_dict)

tentrans = tr.Compose([
    tr.Lift(T.Pad(88)),
    tr.Lift(T.ToTensor())
])
#threshold = 0.6#0.6364

with torch.no_grad():
    for i, ((path, img), ) in enumerate(tqdm(loader)):
        if i == args.early_stop:
            break

        scaler = APRTrans((1024, 768))
        downsized = scaler.forward(img)
        input = tentrans(downsized)[0]
        input = data.add_mean(input)[None]
        if cuda:
            input = input.cuda()

        prediction = network(input)

        upsized = scaler.backward(prediction[0])[0]#.cpu()
        probs = torch.sigmoid(upsized).cpu()
        #postprocessed = postprocess(probs.numpy() > threshold)

        _, fname = os.path.split(path)
        save_path = os.path.join(args.artifacts, fname[:-4] + '.npy')
        np.save(save_path, probs)
