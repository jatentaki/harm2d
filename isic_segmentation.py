import torch, argparse, functools, itertools, os, warnings
import torch.nn.functional as F
import torchvision as tv
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

import isic_loader, losses

from utils import size_adaptive_, maybe_make_dir, print_dict
from criteria import PrecRec
from framework import train, test, Logger
from reg_unet import Unet, repr_to_n
from hunet import HUnet

def load_checkpoint(path):
    cp = torch.load(path)
    return cp['epoch'], cp['score'], cp['model'], cp['optim']

def save_checkpoint(epoch, score, model, optim, path=None, fname=None):
    if path is None:
        raise ValueError("No path specified for save_checkpoint")

    cp = {
        'epoch': epoch,
        'score': score,
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }

    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directory {}".format(path))
    else:
        if not os.path.isdir(path):
            fmt = "{} is not a directory"
            msg = fmt.format(path)
            raise IOError(msg)
    
    fname = fname if fname else 'epoch{}.pth.tar'.format(epoch)
    torch.save(cp, path + os.path.sep + fname)

def load_model(network, path):
    if path is not None:
        load = torch.load(path)
        network.load_state_dict(load)
        print('Loaded from {}'.format(path))


def inspect(network, loader, path, early_stop=None, criteria=[]):
    network.eval()
    path += os.path.sep
    maybe_make_dir(path)

    with tqdm(total=len(loader), dynamic_ncols=True) as progress, torch.no_grad():
        for i, args in enumerate(loader):
            if i == early_stop:
                break

            if torch.cuda.is_available():
                input = args[0].cuda()

            directory = path + os.path.sep + str(i)

            # prediction
            prediction = network(input)
            heatmap = torch.sigmoid(prediction)
            heatmap = heatmap.cpu()[0, 0].numpy()
            hm_path = directory + os.path.sep + 'heatmap.npy'
            maybe_make_dir(hm_path, silent=True)
            np.save(hm_path, heatmap)

            # image 
            image = input.cpu()[0].numpy().transpose(1, 2, 0)
            np.save(directory + os.path.sep + 'image.npy', image)

            # ground truth 
            g_truth = args[-1][0, 0].numpy()
            np.save(directory + os.path.sep + 'g_truth.npy', g_truth)

            progress.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR',
                        help='path to store artifacts')

    parser.add_argument('model', choices=['harmonic', 'baseline'])
    parser.add_argument('action', choices=['train', 'evaluate', 'inspect'])

    parser.add_argument('-tot', '--test-on-train', action='store_true',
                        help='Run evaluation and inspection on training set')
    parser.add_argument('--bloat', type=int, default=50, metavar='N',
                        help='Process N times the dataset per epoch')
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
    parser.add_argument('--restart-checkpoint', action='store_true',
                        help=('consider a loaded checkpoint to be epoch '
                        '0 with 0 best performance'))

    args = parser.parse_args()
    
    if not os.path.isdir(args.artifacts):
        print('creating artifacts directory', args.artifacts)
        os.makedirs(args.artifacts)

    with Logger(args.artifacts + '/log') as logger:
        if args.action == 'inspect' and args.batch_size != 1:
            args.batch_size = 1
            print("Setting --batch-size to 1 for inspection")

        if args.action != 'train' and args.epochs is not None:
            print("Ignoring --epochs outside of training mode")

        logger.add_dict(vars(args))

        warnings.simplefilter("ignore")
        logger.add_msg('Ignoring warnings')

        resize_size = (1024, 768)
        crop_size = (584, 584)#(920, 584)
        resize = isic_loader.ResizeTransform(resize_size)

        train_transform = isic_loader.Compose([
            resize,
            isic_loader.Lift(tv.transforms.ToTensor()),
            isic_loader.RandomCropTransform(crop_size)
        ])

        test_transform = isic_loader.Compose([
            resize,
            isic_loader.Lift(tv.transforms.ToTensor()),
            isic_loader.CenterCropTransform(crop_size)
        ])

        train_data = isic_loader.ISICDataset(
            args.data_path + '/train', global_transform=train_transform
        )
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers
        )

        if args.test_on_train:
            val_data = isic_loader.ISICDataset(
                args.data_path + '/train', global_transform=test_transform
            )
        else:
            val_data = isic_loader.ISICDataset(
                args.data_path + '/test', global_transform=test_transform
            )

        val_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers
        )

        down = [(5, 7, 5), (5, 5, 5), (3, 5, 3), (3, 5, 3)]
        up = [(5, 5, 5), (3, 5, 3), (9,)]
        if args.model == 'baseline':
            down = [repr_to_n(d) for d in down]
            up = [repr_to_n(u) for u in up]
            network = Unet(
                up=up, down=down, in_features=3
            )
        elif args.model == 'harmonic':
            network = HUnet(in_features=3, down=down, up=up, size=5, radius=2)

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

        loss_fn = size_adaptive_(losses.BCE)(masked=False)
        loss_fn.name = 'BCE'

        optim = torch.optim.Adam([
            {'params': network.l2_params(), 'weight_decay': args.l2},
            {'params': network.nr_params(), 'weight_decay': 0.},
        ], lr=args.lr)

        criteria = [loss_fn]

        # jit the model
        example = next(iter(train_loader))[0].cuda()
        network = torch.jit.trace(network, example)

        if args.load:
            start_epoch, best_score, model_dict, optim_dict = load_checkpoint(args.load)
            network.load_state_dict(model_dict)
            optim.load_state_dict(optim_dict)
            fmt = 'Starting at epoch {}, best score {}. Loaded from {}'
            start_epoch += 1 # skip to the next after loaded
            msg = fmt.format(start_epoch, best_score, args.load)
            print(msg)

        if not args.load:
            print('Set start epoch and best score to 0')
            best_score = 0.
            start_epoch = 0

        if args.action == 'inspect':
            inspect(
                network, val_loader, args.artifacts,
                criteria=criteria, early_stop=args.early_stop
            )
        elif args.action == 'evaluate':
            prec_rec = PrecRec(masked=False, n_thresholds=100)
            test(
                network, val_loader, criteria, logger=logger,
                callbacks=[prec_rec]
            )
            f1, thres = prec_rec.best_f1()
            print('F1', f1, 'at', thres)

        elif args.action == 'train':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, 'min', patience=2, verbose=True, cooldown=1
            )

            for epoch in range(start_epoch, args.epochs):
                train(
                    network, train_loader, loss_fn, optim, epoch,
                    early_stop=args.early_stop, logger=logger
                )

                score = test(
                    network, val_loader, criteria,
                    early_stop=args.early_stop, logger=logger
                )
                scheduler.step(score)
                save_checkpoint(
                    epoch, score, network, optim, path=args.artifacts
                )

                if score > best_score:
                    best_score = score 
                    fname = 'model_best_{:.2f}.pth.tar'.format(best_score)
                    save_checkpoint(
                        epoch, best_score, network, optim,
                        path=args.artifacts, fname=fname
                    )
