import torch, argparse, functools, itertools, os, warnings, sys
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import harmonic.d2 as d2

# local directory
import loader

# parent directory
sys.path.append('..')
import losses, framework
import criteria as criteria_mod
from utils import size_adaptive_, maybe_make_dir, print_dict
from reg_unet import Unet, repr_to_n
from hunet import HUnet
from scheduler import MultiplicativeScheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    # paths
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR',
                        help='path to store artifacts')

    # behavior choice
    parser.add_argument('model', choices=['harmonic', 'baseline'])
    parser.add_argument('action', choices=['train', 'evaluate', 'inspect'])

    # framework control
    parser.add_argument('--load', metavar='FILE', default=None, type=str,
                        help='load an existing model')
    parser.add_argument('-j', '--workers', metavar='N', default=1, type=int,
                        help='number of data loader workers')
    parser.add_argument('-nj', '--no-jit', action='store_true',
                        help='disable jit compilation for the model')
    parser.add_argument('--optimize', action='store_true',
                        help='run optimization pass in jit')

    # learning paramters
    parser.add_argument('-b', '--batch-size', metavar='N', default=1, type=int,
                        help='batch size')
    parser.add_argument('-bm', '--batch-multiplier', metavar='N', default=1, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--l2', metavar='F', default=1e-5, type=float,
                        help='l2 regularization strength')
    parser.add_argument('--lr', metavar='F', default=1e-4, type=float,
                        help='learning rate (ADAM)')
    parser.add_argument('--epochs', metavar='N', default=10, type=int,
                        help='number of epochs to train for')

    # other
    parser.add_argument('-s', '--early_stop', default=None, type=int,
                        help='stop early after n batches')
    parser.add_argument('-tot', '--test-on-train', action='store_true',
                        help='Run evaluation and inspection on training set')

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

        warnings.simplefilter("ignore")
        logger.add_msg('Ignoring warnings')

        train_data = loader.ISICDataset(
            args.data_path + '/train', global_transform=loader.ROTATE_TRANS_1024,
            img_transform=T.Compose([
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                T.ToTensor()
            ]),
            normalize=True
        )
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers
        )

        if args.test_on_train:
            val_data = loader.ISICDataset(
                args.data_path + '/train', global_transform=loader.PAD_TRANS_1024,
                normalize=True
            )
        else:
            val_data = loader.ISICDataset(
                args.data_path + '/test', global_transform=loader.PAD_TRANS_1024,
                normalize=True
            )

        val_loader = DataLoader(
            val_data, shuffle=False, batch_size=args.batch_size,
            num_workers=args.workers
        )

        down = [(5, 5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5)]
        up = [(5, 5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5)]
        if args.model == 'baseline':
            down = [repr_to_n(d) for d in down]
            up = [repr_to_n(u) for u in up]
            network = Unet(
                up=up, down=down, in_features=3
            )
        elif args.model == 'harmonic':
            network = HUnet(
                in_features=3, down=down, up=up, size=5, radius=2
            )

        cuda = torch.cuda.is_available()

        network_repr = repr(network)
        logger.add_msg(network_repr)
        print(network_repr)
        n_params = 0
        for param in network.parameters():
            n_params += param.numel()
        print(n_params, 'learnable parameters')

        if cuda:
            network = network.cuda()

        loss_fn = size_adaptive_(losses.BCE)()
        loss_fn.name = 'BCE'

        optim = torch.optim.Adam([
            {'params': network.l2_params(), 'weight_decay': args.l2},
            {'params': network.nr_params(), 'weight_decay': 0.},
        ], lr=args.lr)

        criteria = [loss_fn]

        if args.load:
            checkpoint = framework.load_checkpoint(args.load)
            start_epoch, best_score, model_dict, optim_dict = checkpoint

            network.load_state_dict(model_dict)
            optim.load_state_dict(optim_dict)
            fmt = 'Starting at epoch {}, best score {}. Loaded from {}'
            start_epoch += 1 # skip to the next after loaded
            msg = fmt.format(start_epoch, best_score, args.load)
            print(msg)

#            for module in network.modules():
#                if hasattr(module, 'relax'):
#                    module.relax()
#                    print(f'relaxing {repr(module)}')
#            print(repr(network))

        if not args.load:
            print('Set start epoch and best score to 0')
            best_score = 0.
            start_epoch = 0

        if not args.no_jit:
            example = next(iter(train_loader))[0][0:1].cuda()
            network = torch.jit.trace(
                network, example, check_trace=True, optimize=args.optimize
            )

        if args.action == 'inspect':
            framework.inspect(
                network, val_loader, args.artifacts,
                criteria=criteria, early_stop=args.early_stop
            )
        elif args.action == 'evaluate':
            iou = criteria_mod.ISICIoU(n_thresholds=100)
            callbacks = [iou]
            framework.test(
                network, val_loader, criteria, logger=logger,
                callbacks=callbacks, early_stop=args.early_stop
            )
            best_iou, iou_thres = iou.best_iou()
            print('IoU', best_iou, 'at', iou_thres)

        elif args.action == 'train':
            scheduler = MultiplicativeScheduler(
                optim, 'min', patience=3, verbose=True, cooldown=0,
                cmd_args=args, factor=0.2
            )

            for epoch in range(start_epoch, args.epochs):
                framework.train(
                    network, train_loader, loss_fn, optim, epoch,
                    early_stop=args.early_stop, logger=logger,
                    batch_multiplier=args.batch_multiplier
                )

                iou = criteria_mod.ISICIoU(n_thresholds=100)
                callbacks = [iou]
                score = framework.test(
                    network, val_loader, criteria, callbacks=callbacks,
                    early_stop=args.early_stop, logger=logger
                )
                best_iou, iou_thres = iou.best_iou()
                iou_msg = f'best iou {best_iou} at {iou_thres}'
                print(iou_msg)
                logger.add_msg(iou_msg)
                scheduler.step(score)
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
