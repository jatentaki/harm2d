import torch, argparse, functools, itertools, os, sys
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import harmonic

# local directory
import loader

# parent directory
sys.path.append('..')
import losses, framework, hunet, unet
import criteria as criteria_mod
from utils import size_adaptive_
from criteria import PrecRec
from isic_f1 import IsicIoU

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    # paths
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR',
                        help='path to store artifacts')

    # behavior choice
    parser.add_argument('model', choices=['harmonic', 'baseline', 'unconstrained'])
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
    parser.add_argument('--dropout', metavar='F', default=None, type=float,
                        help='Dropout probability')

    # other
    parser.add_argument('-s', '--early_stop', default=None, type=int,
                        help='stop early after n batches')
    parser.add_argument('-tot', '--test-on-train', action='store_true',
                        help='Run evaluation and inspection on training set')
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

    loader_weights = {
        'bg_weight': 1., 'fg_weight': 1., 'eg_weight': 1.
    }

    train_data = loader.ISICDataset(
        args.data_path + '/train', global_transform=loader.PAD_TRANS_1024,
        img_transform=T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.ToTensor()
        ]),
        **loader_weights
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers
    )

    if args.test_on_train:
        val_data = loader.ISICDataset(
            args.data_path + '/train', global_transform=loader.PAD_TRANS_1024,
            **loader_weights
        )
    else:
        val_data = loader.ISICDataset(
            args.data_path + '/test', global_transform=loader.PAD_TRANS_1024,
            **loader_weights
        )

    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.workers
    )

    down = [(5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5)]
    up = [(5, 5, 5), (5, 5, 5), (5, 5, 5)]
    if args.model in ('harmonic', 'unconstrained'):
        if args.dropout is not None:
            dropout = functools.partial(harmonic.d2.Dropout2d, p=args.dropout)
            setup = {**hunet.default_setup, 'dropout': dropout}
        else:
            setup = hunet.default_setup

        network = hunet.HUnet(in_features=4, down=down, up=up, radius=2, setup=setup)

    elif args.model == 'baseline':
        if args.dropout is not None:
            dropout = functools.partial(torch.nn.Dropout2d, p=args.dropout)
            setup = {**unet.default_setup, 'dropout': dropout}
        else:
            setup = unet.default_setup

        down = [unet.repr_to_n(d) for d in down]
        up = [unet.repr_to_n(d) for d in up]
        network = unet.Unet(up=up, down=down, in_features=4, setup=setup)

    cuda = torch.cuda.is_available()

    network_repr = repr(network)
    print(network_repr)
    writer.add_text('general', network_repr)

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

    if args.load:
        checkpoint = framework.load_checkpoint(args.load)
        start_epoch, best_score, model_dict, optim_dict = checkpoint

        network.load_state_dict(model_dict)
        optim.load_state_dict(optim_dict)
        fmt = 'Starting at epoch {}, best score {}. Loaded from {}'
        start_epoch += 1 # skip to the next after loaded
    else:
        print('Set start epoch and best score to 0')
        best_score = 0.
        start_epoch = 0

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

    if not args.no_jit:
        example = next(iter(train_loader))[0][0:1].cuda()
        network = torch.jit.trace(
            network, example, check_trace=True, optimize=args.optimize
        )

    if args.action == 'inspect':
        framework.inspect(
            network, val_loader, args.artifacts, early_stop=args.early_stop
        )
    elif args.action == 'evaluate':
        callbacks = [PrecRec(), IsicIoU()]

        framework.test(
            network, val_loader, loss_fn, callbacks, start_epoch, writer=writer,
            early_stop=args.early_stop,
        )

        for callback in callbacks:
            results = callback.get_dict()
            print(results)
            for key in results:
                writer.add_scalar(f'Evaluation/{key}', results[key], start_epoch)

    elif args.action == 'train':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 'min', patience=2, verbose=True, cooldown=1
        )

        for epoch in range(start_epoch, args.epochs):
            train_loss = framework.train(
                network, train_loader, loss_fn, optim, epoch,
                writer=writer, early_stop=args.early_stop
            )

            callbacks = [PrecRec(), IsicIoU()]
            framework.test(
                network, val_loader, loss_fn, callbacks, epoch, writer=writer,
                early_stop=args.early_stop,
            )

            for callback in callbacks:
                results = callback.get_dict()
                for key in results:
                    writer.add_scalar(f'Test/{key}', results[key], epoch)

            scheduler.step(train_loss)

            framework.save_checkpoint(
                epoch, 0., network, optim, path=args.artifacts
            )
