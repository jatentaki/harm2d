import torch, argparse, functools, itertools, os, warnings, imageio, sys
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import harmonic

# parent directory
sys.path.append('../')
import transforms as tr
import framework, hunet
from losses import BCE
from utils import size_adaptive_, print_dict
from criteria import PrecRec
from reg_unet import Unet, repr_to_n

# `drive` directory
import loader
from deepglobe_loader import DeepglobeDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR', help='path to store artifacts')

    parser.add_argument('model', choices=['harmonic', 'baseline'])
    parser.add_argument('action', choices=['train', 'evaluate', 'inspect'])

    parser.add_argument('-nj', '--no-jit', action='store_true',
                        help='disable jit compilation for the model')
    parser.add_argument('--parallel', action='store_true',
                        help='enable multi-gpu parallelization via nn.DataParallel')
    parser.add_argument('--optimize', action='store_true',
                        help='run optimization pass in jit')
    parser.add_argument('-tot', '--test-on-train', action='store_true',
                        help='Run evaluation and inspection on training set')
    parser.add_argument('--dropout', metavar='F', default=0.1, type=float,
                        help='Dropout probability')
    parser.add_argument('--load', metavar='FILE', default=None, type=str,
                        help='load an existing model')
    parser.add_argument('-j', '--workers', metavar='N', default=1, type=int,
                        help='number of data loader workers')
    parser.add_argument('-b', '--batch-size', metavar='N', default=1, type=int,
                        help='batch size')
    parser.add_argument('--l2', metavar='F', default=1e-5, type=float,
                        help='l2 regularization strength')
    parser.add_argument('--lr', metavar='F', default=1e-4, type=float,
                        help='learning rate (ADAM)')
    parser.add_argument('--epochs', metavar='N', default=10, type=int,
                        help='number of epochs to train for')
    parser.add_argument('-s', '--early_stop', default=None, type=int,
                        help='stop early after n batches')

    args = parser.parse_args()

    if not os.path.isdir(args.artifacts):
        print('creating artifacts directory', args.artifacts)
        os.makedirs(args.artifacts)

    writer = SummaryWriter()

    if args.action == 'inspect' and args.batch_size != 1:
        args.batch_size = 1
        print("Setting --batch-size to 1 for inspection")

    if args.action != 'train' and args.epochs is not None:
        print("Ignoring --epochs outside of training mode")

    if args.no_jit and args.optimize:
        print("Ignoring --optimize in --no-jit setting")

    writer.add_text('general', str(vars(args)))

    train_data = DeepglobeDataset(
        os.path.join(args.data_path, 'train'), global_transform=loader.ROTATE_512
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    if args.test_on_train:
        val_data = DeepglobeDataset(
            os.path.join(args.data_path, 'train'), global_transform=loader.CROP_512
        )
    else:
        val_data = DeepglobeDataset(
            os.path.join(args.data_path, 'test'),
            global_transform=loader.CROP_512
        )

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    down = [(4, 10, 10), (10, 10, 10), (10, 10, 10), (12, 12, 12)]
    up = [(10, 10, 10), (10, 10, 10), (10, 10, 10)]

    if args.model == 'harmonic':
#        dropout = functools.partial(harmonic.d2.Dropout2d, p=args.dropout)
        setup = {**hunet.default_setup, 'norm': harmonic.d2.GroupNorm2d}
        network = hunet.HUnet(in_features=3, down=down, up=up, radius=2)
    elif args.model == 'baseline':
        down = [repr_to_n(d) for d in down]
        up = [repr_to_n(d) for d in up]
        network = Unet(up=up, down=down, in_features=3)

    cuda = torch.cuda.is_available()

    network_repr = str(network)
    print(network_repr)
    writer.add_text('general', network_repr)

    if cuda:
        network = network.cuda()

    if args.parallel:
        network = torch.nn.DataParallel(network)

    loss_fn = size_adaptive_(BCE)()
    loss_fn.name = 'BCE'

    optim = torch.optim.Adam([
        {'params': network.module.l2_params(), 'weight_decay': args.l2},
        {'params': network.module.nr_params(), 'weight_decay': 0.},
    ], lr=args.lr)

    if not args.no_jit:
        example = next(iter(train_loader))[0].cuda()
        network = torch.jit.trace(
            network, example, check_trace=True, optimize=args.optimize
        )

    if args.load:
        checkpoint = framework.load_checkpoint(args.load)
        start_epoch, best_score, model_dict, optim_dict = checkpoint

        network.load_state_dict(model_dict)
        optim.load_state_dict(optim_dict)
        fmt = 'Starting at epoch {}, best score {}. Loaded from {}'
        start_epoch += 1 # skip to the next after loaded
        msg = fmt.format(start_epoch, best_score, args.load)

    if not args.load:
        print('Set start epoch and best score to 0')
        best_score = 0.
        start_epoch = 0

    if args.action == 'inspect':
        framework.inspect(
            network, val_loader, args.artifacts, early_stop=args.early_stop
        )
    elif args.action == 'evaluate':
        prec_rec = PrecRec()
        framework.test(network, val_loader, loss_fn, [],
                       logger=logger, callbacks=[prec_rec])

    elif args.action == 'train':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 'min', patience=2, verbose=True, cooldown=0
        )

        for epoch in range(start_epoch, args.epochs):
            train_loss = framework.train(
                network, train_loader, loss_fn, optim, epoch,
                writer=writer, early_stop=args.early_stop
            )

            prec_rec = PrecRec(n_thresholds=100)
            framework.test(
                network, val_loader, loss_fn, [prec_rec], epoch, writer=writer,
                early_stop=args.early_stop,
            )
            f1, f1t = prec_rec.best_f1()
            iou, iout = prec_rec.best_iou()

            writer.add_scalar('Test/f1', f1, epoch)
            writer.add_scalar('Test/f1_thres', f1t, epoch)
            writer.add_scalar('Test/iou', iou, epoch)
            writer.add_scalar('Test/iou_thres', iout, epoch)

            scheduler.step(train_loss)

            framework.save_checkpoint(
                epoch, 0., network, optim, path=args.artifacts
            )
