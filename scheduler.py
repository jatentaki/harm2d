from torch.optim.lr_scheduler import ReduceLROnPlateau

class MultiplicativeScheduler(ReduceLROnPlateau):
    def __init__(self, *args, cmd_args=None, **kwargs):
        super(MultiplicativeScheduler, self).__init__(*args, **kwargs)
        self.cmd_args = cmd_args

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

        if self.cmd_args is not None:
            self.cmd_args.batch_multiplier *= 2
            if self.verbose:
                print(f'Epoch {epoch}: increasing batch multiplier to '
                      f'{self.cmd_args.batch_multiplier}')
