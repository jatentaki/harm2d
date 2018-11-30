import torch, argparse, functools, itertools, os, warnings, imageio, sys
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import harmonic

# parent directory
sys.path.append('../')
import framework
from losses import BCE
from utils import size_adaptive_, maybe_make_dir, print_dict
from criteria import PrecRec
from reg_unet import Unet, repr_to_n
from hunet import HUnet

# `drive` directory
import loader

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

    with framework.Logger(args.artifacts + '/log') as logger:
        if args.action == 'inspect' and args.batch_size != 1:
            args.batch_size = 1
            print("Setting --batch-size to 1 for inspection")

        if args.action != 'train' and args.epochs is not None:
            print("Ignoring --epochs outside of training mode")

        if args.no_jit and args.optimize:
            print("Ignoring --optimize in --no-jit setting")

        logger.add_dict(vars(args))

        transform = T.Compose([
            T.CenterCrop(564),
            T.ToTensor()
        ])

        warnings.simplefilter("ignore")
        logger.add_msg('Ignoring warnings')

        if args.rot:
            train_global_transform = loader.RandomRotate()
        else:
            train_global_transform = None

        augumentation = f'augumentation: {type(train_global_transform)}'
        print(augumentation)
        logger.add_msg(augumentation)

        train_data = loader.DriveDataset(
            args.data_path, training=True, bloat=args.bloat, from_=args.cut,
            img_transform=transform, mask_transform=transform,
            label_transform=transform, global_transform=train_global_transform
        )
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

        if args.test_on_train:
            val_data = loader.DriveDataset(
                args.data_path, training=True, img_transform=transform,
                mask_transform=transform, label_transform=transform,
            )
        else:
            val_data = loader.DriveDataset(
                args.data_path, training=False, img_transform=transform,
                mask_transform=transform, label_transform=transform
            )

        val_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

        down = [(2, 5, 2), (5, 7, 5), (10, 14, 10)]
        up = [(5, 7, 5), (2, 5, 2)]
        if args.model == 'harmonic':
            network = HUnet(
                in_features=3, down=down, up=up, radius=2, gate=harmonic.d2.ScalarGate2d,
                norm=harmonic.d2.BatchNorm2d
            )
        elif args.model == 'baseline':
            down = [repr_to_n(d) for d in down]
            up = [repr_to_n(d) for d in up]
            network = Unet(up=up, down=down, in_features=3)

        cuda = torch.cuda.is_available()

        network_repr = str(network)
        print(network_repr)
        logger.add_msg(network_repr)

        if cuda:
            network = network.cuda()

        loss_fn = size_adaptive_(BCE)()
        loss_fn.name = 'BCE'

        optim = torch.optim.Adam([
            {'params': network.l2_params(), 'weight_decay': args.l2},
            {'params': network.nr_params(), 'weight_decay': 0.},
        ], lr=args.lr)

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
            print(msg)
            for module in network.modules():
                if hasattr(module, 'relax'):
                    module.relax()
                    print(f'relaxing {repr(module)}')

            print(repr(network))

        if not args.load:
            print('Set start epoch and best score to 0')
            best_score = 0.
            start_epoch = 0

        if args.action == 'inspect':
            framework.inspect(
                network, val_loader, args.artifacts,
                criteria=criteria, early_stop=args.early_stop
            )
        elif args.action == 'evaluate':
            prec_rec = PrecRec()
            framework.test(network, val_loader, criteria,
                           logger=logger, callbacks=[prec_rec])
            f1, thres = prec_rec.best_f1()
            print('F1', f1, 'at', thres)

        elif args.action == 'train':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, 'min', patience=2, verbose=True, cooldown=1
            )

            for epoch in range(start_epoch, args.epochs):
                train_loss = framework.train(
                    network, train_loader, loss_fn, optim, epoch,
                    early_stop=args.early_stop, logger=logger
                )

                score = framework.test(
                    network, val_loader, criteria,
                    early_stop=args.early_stop, logger=logger
                )
                scheduler.step(train_loss)
                framework.save_checkpoint(
                    epoch, score, network, optim, path=args.artifacts
                )

                if score > best_score:
                    best_score = score 
                    fname = 'model_best_{:.2f}.pth.tar'.format(best_score)
                    framework.save_checkpoint(
                        epoch, best_score, network, optim,
                        path=args.artifacts, fname=fname
                    )
