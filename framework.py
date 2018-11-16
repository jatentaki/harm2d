import torch, os
from utils import AvgMeter, open_file, print_dict, size_adaptive_, fmt_value
import criteria
from tqdm import tqdm

def train(network, dataset, loss_fn, optimizer, epoch, early_stop=None, logger=None):
    def optimization_step(input, mask, target):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            mask = mask.cuda()

        prediction = network(input)
        optimizer.zero_grad()
        loss = loss_fn(prediction, mask, target)
        loss.backward()
        optimizer.step()

        return loss

    loss_meter = AvgMeter()
    network.train()
    msg = 'Train epoch {}'.format(epoch)
    print(msg)
    if logger is not None:
        logger.add_msg(msg)

    with tqdm(total=len(dataset), dynamic_ncols=True) as progress:
        for i, (img, mask, target) in enumerate(dataset):
            if i == early_stop:
                return

            loss = optimization_step(img, mask, target)

            loss_meter.update(loss.item())

            progress.update(1)
            progress.set_postfix(loss=loss_meter.last, mean=loss_meter.avg)

            if logger is not None:
                logger.add_dict({'loss': loss_meter.last})

    return loss_meter.avg


def test(network, dataset, criteria, early_stop=None, logger=None, callbacks=[]):
    if criteria == []:
        raise ValueError("Empty criterion list")

    network.eval()
    meters = [AvgMeter() for c in criteria]

    if logger is not None:
        logger.add_msg('Validation')

    with tqdm(total=len(dataset), dynamic_ncols=True) as progress, torch.no_grad():
        for i, (input, mask, target) in enumerate(dataset):
            if i == early_stop:
                break

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                mask = mask.cuda()

            prediction = network(input)

            for callback in callbacks:
                callback(prediction, mask, target)

            for criterion, meter in zip(criteria, meters):
                perf = criterion(prediction, mask, target)
                if isinstance(perf, torch.Tensor):
                    meter.update(perf.item())
                else:
                    meter.update(perf)

            progress.update(1)

            criteria_dict = {
                c.name: m.last for c, m in zip(criteria, meters)
            }

            if logger is not None:
                logger.add_dict(criteria_dict)

    means = print_dict({c.name: m.avg for c, m in zip(criteria, meters)})
    print('Validation averages\n\t' + means)
    if logger is not None:
        logger.add_msg('Validation averages' + means)
    return meters[0].avg


class Logger:
    def __init__(self, path):
        self.path = path
        self.file = None


    def __enter__(self):
        self.open()
        return self


    def __exit__(self, *errs):
        self.close()
    

    def open(self):
        # find a unique log name
        i = 0
        while True:
            proposed_name = self.path + str(i) + '.log'
            if os.path.isfile(proposed_name):
                print('Found previous log', proposed_name)
                i += 1
            else:
                break
            
        self.path = proposed_name
        self.file = open_file(self.path, 'w', buffering=512)
        print('Logging to', self.path)


    def close(self):
        self.file.close()
        self.file = None


    def add_msg(self, msg):
        self.file.write(msg + '\n')


    def add_dict(self, dict_):
        self.add_msg(print_dict(dict_, prec=6))
