import torch, os
from utils import AvgMeter, open_file, print_dict, size_adaptive_, fmt_value
from tqdm import tqdm

def train(network, dataset, loss_fn, optimizer, epoch, writer,
          early_stop=None):
    loss_meter = AvgMeter()
    network.train()
    msg = 'Train epoch {}'.format(epoch)

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

                writer.add_scalar('Train/loss', loss.item(), epoch)

                return loss

            optimizer.step(evaluation)
            progress.update(1)
            progress.set_postfix(loss=loss_meter.last, mean=loss_meter.avg)


    writer.add_scalar('Train/loss_mean', loss_meter.avg, epoch)
    return loss_meter.avg
