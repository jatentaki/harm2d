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
import transforms as tr
import framework, hunet
from losses import BCE
from utils import size_adaptive_
from criteria import PrecRec

# `drive` directory
import loader
import lbfgs_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR', help='path to store artifacts')

    parser.add_argument('model', choices=['harmonic', 'baseline'])
    parser.add_argument('action', choices=['train', 'evaluate', 'inspect'])

    parser.add_argument('-nj', '--no-jit', action='store_true',
                        help='disable jit compilation for the model')
    parser.add_argument('--rot', action='store_true',
                        help='augument input by rotations')
    parser.add_argument('--optimize', action='store_true',
                        help='run optimization pass in jit')
    parser.add_argument('-tot', '--test-on-train', action='store_true',
                        help='Run evaluation and inspection on training set')
    parser.add_argument('--bloat', type=int, default=50, metavar='N',
                        help='Process N times the dataset per epoch')
    parser.add_argument('--cut', metavar='N', default=None, type=int,
                        help='restrict training set size by N examples')
    parser.add_argument('--dropout', metavar='F', default=0.1, type=float,
                        help='Dropout probability')
    parser.add_argument('--load', metavar='FILE', default=None, type=str,
                        help='load an existing model')
    parser.add_argument('-j', '--workers', metavar='N', default=1, type=int,
                        help='number of data loader workers')
    parser.add_argument('-b', '--batch-size', metavar='N', default=1, type=int,
                        help='batch size')
    parser.add_argument('--lr', metavar='F', default=1, type=float,
                        help='learning rate (LBFGS)')
    parser.add_argument('--epochs', metavar='N', default=10, type=int,
                        help='number of epochs to train for')
    parser.add_argument('-s', '--early_stop', default=None, type=int,
                        help='stop early after n batches')
    parser.add_argument('--logdir', default=None, type=str,
                        help='TensorboardX log directory')

    args = parser.parse_args()

    if not os.path.isdir(args.artifacts):
        print('creating artifacts directory', args.artifacts)
        os.makedirs(args.artifacts)

    writer = SummaryWriter(args.logdir)

    if args.action == 'inspect' and args.batch_size != 1:
        args.batch_size = 1
        print("Setting --batch-size to 1 for inspection")

    if args.action != 'train' and args.epochs is not None:
        print("Ignoring --epochs outside of training mode")

    if args.no_jit and args.optimize:
        print("Ignoring --optimize in --no-jit setting")

    writer.add_text('general', str(vars(args)))

    transform = T.Compose([
        T.CenterCrop(644),
        T.ToTensor()
    ])

    test_global_transform = tr.Lift(T.Pad(40))

    tr_global_transform = [
#        tr.RandomRotate(),
#        tr.RandomFlip(),
        tr.Lift(T.Pad(40))
    ]
    tr_global_transform = tr.Compose(tr_global_transform)

    train_data = loader.DriveDataset(
        args.data_path, training=True, bloat=args.bloat, from_=args.cut,
        img_transform=transform, mask_transform=transform,
        label_transform=transform, global_transform=tr_global_transform
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    if args.test_on_train:
        val_data = loader.DriveDataset(
            args.data_path, training=True, img_transform=transform,
            mask_transform=transform, label_transform=transform,
            global_transform=test_global_transform
        )
    else:
        val_data = loader.DriveDataset(
            args.data_path, training=False, img_transform=transform,
            mask_transform=transform, label_transform=transform,
            global_transform=test_global_transform
        )

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    down = [(2, 3, 2), (4, 5, 4), (8, 10, 8)]
    up = [(4, 5, 4), (2, 3, 2)]
    if args.model == 'harmonic':
        dropout = functools.partial(harmonic.d2.Dropout2d, p=args.dropout)
        setup = {**hunet.default_setup, 'dropout': dropout}
        network = hunet.HUnet(in_features=3, down=down, up=up, radius=2)#, setup=setup)
    elif args.model == 'baseline':
        dropout = functools.partial(torch.nn.Dropout2d, p=args.dropout)
        setup = {**unet.default_setup, 'dropout': dropout}
        down = [unet.repr_to_n(d) for d in down]
        up = [unet.repr_to_n(d) for d in up]
        network = unet.Unet(up=up, down=down, in_features=3)#, setup=setup)

    cuda = torch.cuda.is_available()

    network_repr = str(network)
    print(network_repr)
    writer.add_text('general', network_repr)

    n_params = 0
    for param in network.parameters():
        n_params += param.numel()
    print(n_params, 'learnable parameters')

    if cuda:
        network = network.cuda()

    loss_fn = size_adaptive_(BCE)()
    loss_fn.name = 'BCE'

    optim = torch.optim.LBFGS(network.parameters(), lr=args.lr)

    criteria = [loss_fn]

    if not args.no_jit:
        example = next(iter(train_loader))[0][0:1].cuda()
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
        framework.test(network, val_loader, criteria,
                       logger=logger, callbacks=[prec_rec])

    elif args.action == 'train':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 'min', patience=2, verbose=True, cooldown=1
        )

        for epoch in range(start_epoch, args.epochs):
            train_loss = lbfgs_train.train(
                network, train_loader, loss_fn, optim, epoch,
                writer=writer, early_stop=args.early_stop
            )

            prec_rec = PrecRec(n_thresholds=100)
            framework.test(
                network, val_loader, loss_fn, [prec_rec], epoch, writer=writer,
                early_stop=args.early_stop,
            )

            results = prec_rec.get_dict()
            for key in results:
                writer.add_scalar(f'Test/{key}', results[key], epoch)

            scheduler.step(train_loss)

            framework.save_checkpoint(
                epoch, 0., network, optim, path=args.artifacts
            )
