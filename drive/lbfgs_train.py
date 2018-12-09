import torch, os
from utils import AvgMeter, open_file, print_dict, size_adaptive_, fmt_value
from tqdm import tqdm

def train(network, dataset, loss_fn, optimizer, epoch,
          early_stop=None, logger=None):
    loss_meter = AvgMeter()
    network.train()
    msg = 'Train epoch {}'.format(epoch)
    print(msg)
    if logger is not None:
        logger.add_msg(msg)

    with tqdm(total=len(dataset), dynamic_ncols=True) as progress:
        for i, args in enumerate(dataset):
            if i == early_stop:
                break

            if torch.cuda.is_available():
                args = [a.cuda() for a in args]

            def evaluation():
                network.zero_grad()
                prediction = network(args[0])
                loss = loss_fn(prediction, *args[1:])
                loss.backward()
                loss_meter.update(loss.item())
                if logger is not None:
                    logger.add_dict({'loss': loss_meter.last})

                return loss

            optimizer.step(evaluation)
            progress.update(1)
            progress.set_postfix(loss=loss_meter.last, mean=loss_meter.avg)

            if logger is not None:
                logger.add_dict({'loss': loss_meter.last})

    return loss_meter.avg
