import torch, argparse, functools, itertools, os, sys
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import harmonic

# parent directory
sys.path.append('../')
import losses, framework, hunet, unet
import criteria as criteria_mod
from utils import size_adaptive_
from criteria import PrecRec

# `drive` directory
import loader
from loader import RotatedISICDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    # paths
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR', help='path to store artifacts')
    parser.add_argument('model_path', metavar='FILE', help='path to model save file')

    # behavior choice
    parser.add_argument('model', choices=['harmonic', 'baseline', 'unconstrained'])

    parser.add_argument('-j', '--workers', metavar='N', default=1, type=int,
                        help='number of data loader workers')
    parser.add_argument('-s', '--early_stop', default=None, type=int,
                        help='stop early after n batches')
    parser.add_argument('--logdir', default=None, type=str,
                        help='TensorboardX log directory')

    args = parser.parse_args()

    if not os.path.isdir(args.artifacts):
        print('creating artifacts directory', args.artifacts)
        os.makedirs(args.artifacts)

    writer = SummaryWriter(args.logdir)

    writer.add_text('general', str(vars(args)))

    val_data = RotatedISICDataset(
        args.data_path + '/test', global_transform=loader.PAD_TRANS_1024,
        normalize=True
    )

    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=args.workers
    )

    down = [(5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5)]
    up = [(5, 5, 5), (5, 5, 5), (5, 5, 5)]

    if args.model in ('harmonic', 'unconstrained'):
        network = hunet.HUnet(in_features=3, down=down, up=up, radius=2)

    elif args.model == 'baseline':
        down = [unet.repr_to_n(d) for d in down]
        up = [unet.repr_to_n(d) for d in up]
        network = unet.Unet(up=up, down=down, in_features=3)

    cuda = torch.cuda.is_available()

    network_repr = repr(network)
    print(network_repr)
    writer.add_text('general', network_repr)

    if args.model == 'unconstrained':
        for module in network.modules():
            if hasattr(module, 'relax'):
                module.relax()
                print(f'relaxing {repr(module)}')

        network_repr = repr(network)
        print(network_repr)
        writer.add_text('general', network_repr)

        n_params = 0
        for param in network.parameters():
            n_params += param.numel()
        print(n_params, 'learnable parameters')

    if cuda:
        network = network.cuda()

    checkpoint = framework.load_checkpoint(args.model_path)
    start_epoch, best_score, model_dict, optim_dict = checkpoint

    network.load_state_dict(model_dict)


    framework.inspect(
        network, val_loader, args.artifacts, early_stop=args.early_stop
    )
