import torch, functools, os, sys
from tensorboardX import SummaryWriter
import harmonic

# parent directory
sys.path.append('../')
import framework, hunet
from losses import BCE
from utils import size_adaptive_
from criteria import PrecRec

class Experiment:
    def __init__(self, setup, train_loader, test_loader, artifacts, args):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = size_adaptive_(BCE)()
        self.loss_fn.name = 'BCE'
        self.writer = SummaryWriter()
        self.writer.add_text('general', str(vars(args)))

        if not os.path.isdir(artifacts):
            print('Creating artifacts directory', artifacts)
            os.makedirs(artifacts)

        self.artifacts = artifacts

    def train(self):
        dropout = functools.partial(harmonic.d2.Dropout2d, p=setup['dropout'])
        setup = {**hunet.default_setup, 'dropout': dropout}
        network = hunet.HUnet(
            in_features=3, down=setups['down'], up=setup['up'], radius=setup['radius'],
            setup=setup
        ).cuda()

        self.writer.add_text('general', str(networl))

        optim = torch.optim.Adam([
            {'params': network.l2_params(), 'weight_decay': args.l2},
            {'params': network.nr_params(), 'weight_decay': 0.},
        ], lr=args.lr)

        if not args.no_jit:
            example = next(iter(self.train_loader))[0][0:1].cuda()
            network = torch.jit.trace(
                network, example, check_trace=True, optimize=args.optimize
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 'min', patience=3, verbose=True, cooldown=0
        )

        best_f1 = 0.

        for epoch in range(start_epoch, args.epochs):
            train_loss = framework.train(
                network, self.train_loader, self.loss_fn, optim, epoch,
                writer=writer, early_stop=args.early_stop
            )

            prec_rec = PrecRec(n_thresholds=100)
            framework.test(
                network, self.val_loader, self.loss_fn, [prec_rec], epoch, writer=writer,
                early_stop=args.early_stop,
            )
            f1, f1t = prec_rec.best_f1()
            iou, iout = prec_rec.best_iou()

            best_f1 = max(best_f1, f1)

            writer.add_scalar('Test/f1', f1, epoch)
            writer.add_scalar('Test/f1_thres', f1t, epoch)
            writer.add_scalar('Test/iou', iou, epoch)
            writer.add_scalar('Test/iou_thres', iout, epoch)

            scheduler.step(train_loss)

            framework.save_checkpoint(
                epoch, f1, network, optim, path=args.artifacts
            )

        return best_f1
